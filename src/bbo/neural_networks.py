from typing import Callable

import numpy as np
from scipy.stats import qmc
import torch
import torch.nn as nn
import torch.optim as optim

from bbo.random import create_rng_seed

ACTIVATION_FUNCTIONS_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


class Net(nn.Module):
    """General class for neural networks."""
    def __init__(
        self, n_inputs: int, neurons: list[int], activations: list[str]
    ):
        """Initialise neural network.

        Args:
            n_inputs (int): number of neurons in input layer.
            neurons (list[int]): number of neurons in each hidden layer. Length
              of list is the number of hidden layers.
            activations (list[str]): activation functions after each hidden
              layer as defined in ACTIVATIONS_FUNCTIONS_MAP.
        """
        super().__init__()
        layers = []
        n_neurons = n_inputs  # number of neurons in input layer
        for neuron, act_func in zip(neurons, activations):
            layers.append(nn.Linear(n_neurons, neuron))
            layers.append(ACTIVATION_FUNCTIONS_MAP[act_func]())
            n_neurons = neuron  # number of input neurons for next layer
        layers.append(nn.Linear(n_neurons, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Run forward pass."""
        return self.net(x)


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    neurons: list[int],
    activations: list[str],
    learning_rate: float = 0.01,
    weight_decay: float = 0.01,
    epochs: int = 500,
    return_decay: bool = False,
    seed_input: str | None = None,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
):
    """Train neural network. Use ADaptive Moment Estimation (ADAM) algorithm
    for training network.

    Args:
        x (np.ndarray): input data.
        y (np.ndarray): target values.
        neurons (list[int]): number of neurons in each hidden layer. Length of
          list is the number of hidden layers.
        activations (list[str]): activation functions after each hidden layer
          as defined in ACTIVATIONS_FUNCTIONS_MAP.
        learning_rate (float): step size used by optimiser to update model
          weights.
        weight_decay (float): L2 regularisation coefficient applied to model
          weights.
        epochs (int): number of training cycles.
        return_decay (bool): return loss function decay curve.
        seed_input (str): string used to generate random seed.
        x_val (np.ndarray): optional validation set input data.
        y_val (np.ndarray): optional validation set target values.

    Returns:
        trained neural network model and optionally loss function decay curves.
    """
    # Set random seed
    if seed_input is not None:
        torch.manual_seed(create_rng_seed(seed_input))

    n_inputs = x.shape[1]  # number of input features
    model = Net(n_inputs, neurons, activations)
    opt = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()  # Mean Square Error as loss function

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    if x_val is not None:
        x_val_t = torch.tensor(x_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        # Training
        model.train()
        opt.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
        train_losses.append(loss.item())

        # Validation
        if x_val is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val_t)
                val_loss = loss_fn(val_pred, y_val_t)
            val_losses.append(val_loss.item())

    if return_decay:
        if x_val is not None:
            return model, train_losses, val_losses
        return model, train_losses

    return model


def loocv_score(
    x: np.ndarray,
    y: np.ndarray,
    neurons: list[int],
    activations: list[str],
    learning_rate: float = 0.01,
    weight_decay: float = 0.01,
    seed_input: str | None = None,
):
    """Conduct Leave One Out Cross Validation (LOOCV) for a neural network.

    Args:
        x (np.ndarray): input data.
        y (np.ndarray): target values.
        neurons (list[int]): number of neurons in each hidden layer. Length of
          list is the number of hidden layers.
        activations (list[str]): activation functions after each hidden layer
          as defined in ACTIVATIONS_FUNCTIONS_MAP.
        learning_rate (float): step size used by optimiser to update model
          weights.
        weight_decay (float): L2 regularisation coefficient applied to model
          weights.
        seed_input (str): string used to generate random seed.

    Returns:
        Mean Root Mean Square Error (RMSE) and standard deviation of RMSE
          across all LOOCV folds.
    """
    n = len(x)
    losses = []

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        if seed_input is not None:
            seed_input = f"{seed_input} LOOCV {i}"

        model = train_model(
            x=x[mask],
            y=y[mask],
            neurons=neurons,
            activations=activations,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=5000,
            seed_input=seed_input,
        )

        with torch.no_grad():
            pred = model(torch.tensor(x[i:i + 1], dtype=torch.float32))
            loss = (pred.item() - y[i]) ** 2
            losses.append(loss)

    return np.sqrt(np.mean(losses)), np.std(np.sqrt(losses))


def train_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    neurons: list[int],
    activations: list[str],
    learning_rate: float = 0.01,
    weight_decay: float = 0.01,
    n_models: int = 20,
    seed_input: str | None = None,
):
    """Train ensemble of neural network models.

    Args:
        x (np.ndarray): input data.
        y (np.ndarray): target values.
        neurons (list[int]): number of neurons in each hidden layer. Length of
          list is the number of hidden layers.
        activations (list[str]): activation functions after each hidden layer
          as defined in ACTIVATIONS_FUNCTIONS_MAP.
        learning_rate (float): step size used by optimiser to update model
          weights.
        weight_decay (float): L2 regularisation coefficient applied to model
          weights.
        n_models (int): number of neural network models to train.
        seed_input (str): string used to generate random seed.

    Returns:
        list of trained neural network models.
    """
    models = []
    n = len(x)

    for mdl in range(n_models):
        if seed_input is not None:
            seed_input = f"{seed_input} Model {mdl}"

        model = train_model(
            x=x,
            y=y,
            neurons=neurons,
            activations=activations,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=5000,
            seed_input=seed_input,
        )
        models.append(model)

    return models


def ensemble_predict(models: list[Net], x: np.ndarray):
    """Predict output values for given inputs from each neural network model in
    ensemble and calculate the mean and standard deviation across all models.

    Args:
        models (list[Net]): list of trained neural network models.
        x (np.ndarray): input data.

    Returns:
        means and standard deviations of output values across all models.
    """
    x_t = torch.tensor(x, dtype=torch.float32)

    preds = []

    for m in models:
        with torch.no_grad():
            preds.append(m(x_t).numpy())

    preds = np.stack(preds)

    mean = preds.mean(0).flatten()
    std = preds.std(0).flatten()

    return mean, std


def propose_point(
    models: list[Net],
    n_dimensions: int,
    seed_input: str,
    acq_func: Callable,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    n_candidates: int = 2000,
    **acq_kwargs,
):
    """Generate candidates using Latin Hypercube algorithm and find candidate
    that maximises acquisition function.

    Args:
        models (list[Net]): list of trained neural network models.
        n_dimensions (int): number of dimensions (i.e. input features).
        seed_input (str): string used to generate random seed.
        acq_func (Callable): acquisition function.
        bounds (tuple): tuple of lower and upper boundaries in each dimension.
        n_candidates (int): number of candidates to generate.
        acq_kwargs: keyword arguments for acquisition function.
    """
    sampler = qmc.LatinHypercube(
        d=n_dimensions, seed=np.random.default_rng(create_rng_seed(seed_input))
    )
    candidates = sampler.random(n_candidates)

    if bounds is None:  # default boundaries cover entire space
        bounds = (np.zeros(n_dimensions), np.ones(n_dimensions))
    lower, upper = bounds
    candidates = lower + (upper - lower) * candidates

    mean, std = ensemble_predict(models, candidates)

    acq = acq_func(mean, std, **acq_kwargs)

    best_idx = np.argmax(acq)

    return candidates[best_idx]
