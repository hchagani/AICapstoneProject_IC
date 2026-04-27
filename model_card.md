# Model Card for Black Box Optimisation Capstone Project OctoBBO

This model card contains information on the machine learning models used for the Black Box Optimisation (BBO) capstone project that forms part of the requirements for the Professional Certificate in Machine Learning and Artificial Intelligence at Imperial College London.

## Overview

The __OctoBBO__ approach derives its name from the search for the global maxima of 8 (_octo_) functions that are hidden behind Black Box Optimisation (_BBO_) problems. As a unified approach is not employed and each function is treated as a separate problem, and the word "function" is derived from Latin, the octo- (eight in Latin) prefix is applied.

Rather than presenting a single model, a range of strategies is documented in the [notebooks](./notebooks/) within this repository. These strategies involve the reassessment of employed models and the implementation of new models as more data is revealed. Functions are evaluateed by submitting queries consisting of input values, which are derived using the strategies documented in this repository. The evolution of the strategy for each function on a weekly basis is presented here.

Each week corresponds to an update in version number as strategies change. The final version number is 14, derived from 13 weekly analyses and one analysis of the final result.

## Intended Use

The model is primarily presented as part of the requirements for [Imperial College London's Professional Certificate in Machine Learning and Artificial Intelligence](https://execed-online.imperial.ac.uk/professional-certificate-ml-ai). Additionally, the model can be used for educational purposes as an illustration of how to maximise functions hidden behind a black box with few evaluations. Therefore, it is presented as an evolution of strategies over time rather than simply the final result.

The strategies presented here cannot be applied directly to maximise other BBO problems. These strategies are presented to invoke ideas and suggest avenues to pursue when presented with other BBO challenges.

## Details of Employed Strategies

A variety of strategies were implemented and no uniform model was used across all functions. The strategies employed for each function evolved in response to the outputs as queries were processed. These outputs dictated the direction of the subsequent analyses.

A high-level summary is presented here. Further details can be found in the individual [notebooks](./notebooks/).

With the exception of functions 5 & 6, which were deemed to have clear regions that could immediately be exploited, an initial phase of exploration was conducted. Typically, the initial 3-6 queries were dedicated to this phase, which either employed Bayesian Optimisation using a Gaussian Process surrogate model or a spatial sampling method. Once a promising region was identified, the strategy evolved into one of exploitation, which typically lasted until the final submission. This latter phase included applying a range of different machine learning models to the data in an attempt to find one that best described the most promising region.

An exception to this was function 4, where the exploitation phase was followed by another exporation phase, returning to exploitation for the final query.

In the cases of functions 5 & 6, the process was reversed. The initial exploitation phase employed Bayesian Optimisation with a Gaussian Process surrogate model and Probability of Improvement (PI) acquisition function. Once it was deemed that a point of diminishing returns had been reached, a policy of exploration was adopted. In the case of function 5, this was short-lived with a return to exploitation a couple of weeks later. For function 6, the exploratory phase lasted until the final three weeks, which saw a return to exploitation.

The table below summarises the methods employed for each submission to determine the next point to submit as a query. Further details can be found in the [individual function notebooks](./notebooks/):

| Function | Exploration Techniques | Exploitation Techniques |
| :-: | --- | --- |
| 1 | Sample farthest point from observed data points and corners, BO with GP surrogate model and UCB acquisition function to assess candidates from grid search. | BO with GP surrogate model and PI acquisition function to assess candidates from grid search. |
| 2 | Sample farthest point from observed data points and corners, BO with GP surrogate model and UCB acqusition function to assess candidates from grid search | Find centroid of points in promising band, RF ensemble model with UCB acquisition function to assess candidates from grid search, BO with global GP surrogate model and UCB or EI acqusition functions to assess candidates from random search within regions defined by decision tree model. |
| 3 | BO with GP surrogate model and standard deviation only or UCB acquisition function to assess candidates from grid search. | BO with global GP surrogate model and PI acquisition function to assess candidates from local grid search. |
| 4 | Sample points from midpoints of largest empty spaces in each dimension, BO with GP surrogate model and UCB acqusition function to assess candidates from grid search, ensemble of NN models with variety of acquisition functions to assess candidates from random search, NN model to assess candidates from random search. | Find predicted peak of quadratic linear regression model. |
| 5 | Probing of above-below mean boundary defined by SVM model, BO with GP surrogate model and UCB acquisition function to assess candidates from medians of clusters extracted from random searches. | BO with GP surrogate model and PI acquisition function to assess candidates from grid search, RF ensemble model with UCB, EI and PI acqusiition functions to assess candidates from grid search. |
| 6 | BO with GP surrogate model and UCB acquisition function to assess candidates from grid seach and medians of clusters extracted from random searches. | BO with GP surrogate model and PI and EI acquisition function to assess candidates from grid search, RF ensemble model with EI acquisition function. |
| 7 | Sample points from midpoints of largest empty spaces in each dimension, BO with GP surrogate model and UCB acquisition function to assess candidates from grid search and from medians of clusters extracted from random searches, RF ensemble model with UCB acquisition function, ensemble of NN models with variety of acquisition functions to assess candidates from random search. | BO with GP surrogate model and PI acquisition function to assess candidates from local grid search. |
| 8 | Sample points from midpoints of largest empty spaces in each dimension, BO with GP surrogate model and UCB acquisition function to assess candidates from grid search. | BO with GP surrogate model and PI acquisition function to assess candidates from local grid search. |

__Key:__ _BO_: Bayesian Optimisation, _EI_: Expected Improvement, _GP_: Gaussian Process, _NN_: Neural Network, _PI_: Probabilty of Improvement, _RF_: Random Forests, _SVM_: Support Vector Machines, _UCB_: Upper Confidence Bound

## Performance

As the goal of the challenge is to find the global maximum for each function, the maximum observed output is the primary metric employed to assess the performance of the strategies. As there is no overarching strategy applied to all function, the performance is measured on a function-by-function basis.

A secondary goal is to increase knowledge of the underlying functions. In this respect, other metrics such as the Euclidean distance between candidates and observed data points and/or domain boundaries are used to select points to submit in weekly queries. Linear regression and neural network ensemble models use the mean square error and root mean square error as metrics to assess their performance. Accuracy is the primary metric for the support vector machines model.

For all functions, at least one query returned a data point with output greater than that in the initial data set. By this metric, the strategies employed were successful. The table below is a summary of these results:

| Function | Best point | Output | Absolute improvement over initial data set |
| :-: | --- | :-: | :-: |
| 1 | (0.690812, 0.723212) | 1.519374628E-9 | 1.519373857E-9 |
| 2 | (0.719125, 0.079963) | 0.692164 | 0.080959 |
| 3 | (0.392581, 0.611593, 0.426556) | -0.017109 | 0.017727 |
| 4 | (0.402835, 0.399055, 0.376248, 0.397277) | 0.199695 | 4.225237 |
| 5 | (0.865263, 0.722000, 0.991167, 0.958600) | 3315.259755 | 2226.400137 |
| 6 | (0.568700, 0.374362, 0.385750, 0.832375, 0.152625) | -0.451271 | 0.262994 |
| 7 | (0.007427, 0.221733, 0.174203, 0.280670, 0.353335, 0.680910) | 2.319813 | 0.954845 |
| 8 | (0.216733, 0.235980, 0.161700, 0.137167, 0.715833, 0.472566, 0.196443, 0.729239) | 9.956185 | 0.357703 |

where the absolute improvement over the initial data set $I$ is defined as the difference in outputs between the best point from the final $f(x_{\mathrm{final}})$ and initial $f(x_{\mathrm{initial}})$ data sets:

$$
I = f(x_{\mathrm{final}}) - f(x_{\mathrm{initial}})
$$

By definition, $I = 0$ if there has been no improvement.

## Assumptions & Limitations

### Assumptions

- __Presence of local maxima__: for all functions except 5 & 6, it was assumed that there were regions of local maxima. For some functions, such as function 8, this assumption was upheld. However, given the limited query budget, it has not been possible to verify this assumption for other functions.
- __Smoothness of landscape__: in most cases, it was assumed that the landscape was smooth and therefore could be approximated with a Radial Basis Function kernel when using Bayesian Optimisation with a Gaussian Process surrogate model. Occasionally, a Matern kernel was used but the smoothness parameter was always set to 1.5. There was not enough data to make a definitive conclusion about the evenness of the landscape.
- __No or low levels of stochastic noise__: it was assumed that there was little to no stochastic noise. This is almost certain not to be the case, and therefore some peaks may have been misidentified. However, with the exception of function 1, promising regions have been identified, within which multiple high output points have been found. Therefore, while there may be some noise reported in a peak's output, it can be stated with a reasonable certainty that a promising region that has been identified contains a peak.

### Limitations

- __Small number of data points__: for all functions, the size of initial data set was quite small relative to the number of features. Coupled with the limied query budget making each evaluation relatively expensive, the most ideal strategy was Bayesian Optimisation. However, other machine learning models were also trained on the data and used to propose points to query. In some of these cases, such as with neural network models, the number of data points was too low to build a useful model.
- __Limited query budget__: similarly to the above, a query budget of 13 evaluations was not sufficiently large enough for any rigorous testing of machine learning models. Coupled with the restriction of one evaluation per function per query, it was not possible to say with certainty that a particular model was a good representation of the underlying function.
- __Training models on global data to predict local phenomena__: as relatively few data points were available, there was not enough data in promising regions to train a local model. Therefore, the global data set was used, thus producing a global model. However, in many of the functions it was established that a single global model was not sufficient to describe the underlying function.
- __Computational power__: the analysis has been performed on a laptop, and therefore settings for computationally intensive methods, such as grid searches and neural network ensemble models, have been relatively modest. For example, the grid resolution for searches at higher dimensions has been very low. This has been mitigated somewhat with a recursive grid search, where a new grid of higher resolution is built around the most promising point and the search is repeated until a suitable resolution has been achieved. However, because of the relatively poor resolution of the initial grid that spans the domain, some promising regions may not have been identified.

## Ethical Considerations

The [data set](./data/) used in this strategy, the [notebooks](./notebooks/) containing details of the rationale and models employed in this strategy, and the [source code](./src/bbo/) are available in this repository. This allows for full transparency of the analysis and for it to be reproduced in part or in full. Errors made in the original analysis have been identified.

The strategy, as detailed here, is intended for educational purposes, and to invoke ideas and suggest avenues to pursue for other BBO challenges. Should the same functions be used in a future Capstone project for [Imperial College London's Professional Certificate in Machine Learning and Artificial Intelligence](https://execed-online.imperial.ac.uk/professional-certificate-ml-ai), the information in this repository can be used by students to gain additional information extra to the initial data set, and therefore is potentially an unfair advantage. While the stated aim of this project is to find the global maximum for each function, the secondary goal is to gain practical experience using a wide variety of machine learning models. The data set is available for all to use removing any unfair advantages and it is hoped that the notebooks will encourage experimentation with other models.
