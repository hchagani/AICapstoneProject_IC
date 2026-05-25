# AICapstoneProject_IC

## Project Overview
The Black Box Optimisation (BBO) capstone project consists of 8 optimisation problems where the underlying functions are unknown. The goal is to find the global maxima through limited evaluations of each function. This mirrors the limitations in many real world machine learning challenges, where evaluations can be expensive and extensive searches across the entire phase space are not possible. By identifying regions of interest, and focusing evaluations there, much of the relevant structure of the underlying functions can be inferred and maxima can be identified.

This project exhibits problem-solving skills, the ability to work with limited data and within constraints, the extraction of information from data and the flexibility of changing approach in light of new evidence, and documentation and communication skills.

Further details can be found in the [datasheet](./datasheet.md) and [model card](./model_card.md).

## Setup instructions
From the command line:
```bash
# Clone the repo and navigate to the home directory
$ git clone https://github.com/hchagani/AICapstoneProject_IC.git
$ cd AICapstoneProject_IC

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install project (this will install all dependencies too)
pip install -e .

# Launch jupyter lab
jupyter lab
```
This should launch a browser. Navigate to the `notebooks` directory and select the notebook you wish to view.

## Inputs & Outputs
The functions consist of a different number of inputs, or in other words features. The number of features for each function ranges from two to 8. Although the names of the features are unknown, each feature has been normalised to lie in the range [0.0, 1.0). Outputs are not normalised and can be of any value.

Points to query are submitted weekly via a portal. Inputs can be submitted to a precision of 6 decimal places.

For example, the output of function 6 is a combined score of flavour, consistency, calories, waste and cost of a cake recipe as ranked by an expert tester given five ingredient inputs. The names of the ingredients are unknown. Submitting the coordinates of a point to query for this function involves submitting a value for each ingredient delimited by a hyphen as follows:

0.728186-0.154693-0.732552-0.693997-0.056401

Once processed, the outputs for each function are evaluated. In the case of function 6, this would be the combined score detailed above.

## Challenge Objectives
The objectives for all the optimisation problems is to find the global maximum for each function. While some background information for each function has been provided through which some assumptions can be inferred, the majority of input features have not been identified, making it difficult to use domain knowledge. Additionally, the underlying shape of each function is unknown.

For each function an initial data set has been provided and one point can be queried each week. It can take up to 48 hours for a query to be processed so there can be a delay in receiving the output. As the project lasts for a finite number of weeks, there are a limited number of queries that can be submitted. The delay in response and the limited number of queries mimic real world constraints.

## Technical Approach
As there are 8 different functions that are independent from each other, it would be prudent to treat them as eight different optimisation problems.

### Function 1
The input features are the axes of an area with contamination sources. The output is the strength of contamination (e.g. radiation) at the coordinates.

Initial observations of the data indicate there are negative values, which make little physical sense and therefore may be noise. The positive values are rather flat, and there does not appear to be a clear peak.

#### Strategy
1. Initial exploration (Weeks 1-3):
   - Sampled points far away from existing observations and boundaries.
   - No clear structure emerged.
1. Initial Bayesian Optimisation (Weeks 4-6):
   - Adopted Gaussian Process (GP) surrogate models with Radial Basis Function (RBF) and Matern kernels.
   - Used an Upper Confidence Bound (UCB) acquisition function to balance exploration with exploitation. This tended to suggest boundary points as these areas were where the global GP models were least certain.
   - Attempts at linear regression with leave one out and 5-fold cross validation were not fruitful.
1. Combination of classification & regression GP surrogate models (Weeks 7-13):
   - Introduced two GP models:
     1. Classificaton GP to predict probability of positive output.
     1. Regression GP trained on logarithmic values of positive outputs to predict magnitude of output.
   - Acquisition function was product of probability from classification GP and either UCB or Probability of Improvement (PI) acquisition functions from regression GP.
   - Generated equidistant candidate points along the circumference of a circle around the best observed point, with radius equal to the midpoint between this point and its nearest neighbour with negative output. This allowed for exploitation in promising regions.
   - Successfully identified a point with an output several orders of magnitude higher than the best point in the initial data set.
   - The landscape appears jagged with many local maxima. Identified at least 2 promising regions for further queries.

### Function 2
The input features are two machine learning model parameters. The output is the log-likelihood score.

Initial observations of the data suggest that there are two promising regions along the 0.6 < `x0` < 0.8 band.

#### Strategy
1. Initial exploration & Bayesian Optimisation (Weeks 1-6):
   - Adopted Gaussian Process (GP) surrogate models with Radial Basis Function (RBF) and Matern kernels.
   - Used Upper Confidence Bound (UCB) function to balance exploration with exploitation. Occasionally this tended to suggest points close to the boundary, although some promising points in more central regions were identified.
   - The underlying function appears to be less sensitive to changes in `x1`, and the two promising regions identified in the initial data set merge into a promising band between 0.6 <`x0` < 0.8.
   - Projecting points onto the `x0` axis, and submitting queries in this promising band reveal a complicated landscape that may consist of many sharp peaks.
   - Attempts at linear regression with leave one out and 5-fold cross validation were not fruitful.
1. Region-based analysis with decision trees and GP surrogate models (Weeks 7-13):
   - Introduced decision tree models to partition domain into regions based on observed output values.
   - Generated candidate points in each region. More candidates were generated in regions with a higher mean output using softmax weighting.
   - Global GP surrogate model and Expected Improvement (EI) acquisition function were used to assess and select candidate points to query.
   - The landscape appears complex in region 0.6 < `x0` < 0.8 with the potential presence of many local maxima.
   - A promising region in the band where `x0` is approximately 0.9 has not been explored.
   - Random forests and extra trees ensembles were investigated as surrogate model replacements for the GP. However, they were found to be biased towards high density areas and therefore may miss promising unexplored regions that were picked up by the GP.

### Function 3
The input features are three compounds in a drug discovery project. The output is the severity of the side effects from different combinations of these compounds.

The outputs have been inverted so they are all negative. Therefore, a higher number corresponds to smaller side effects. Of the intial data points, two high value ones lie near each other indicating the presence of a promising region to explore.

#### Strategy
1. Initial exploration & Bayesian Optimisation (Weeks 1-6):
   - Adopted Gaussian Process (GP) surrogate models with Radial Basis Function (RBF) kernel.
   - Used Upper Confidence Bound (UCB) function to balance exploration with exploitation. This either tended to suggest new regions to explore when model uncertainty was high or points in promising areas to exploit.
   - Identified two promising regions.
1. Exploitation with the Probability of Improvement (PI) acquisition function (Weeks 7-13):
   - Continued with GP surrogate models with RBF kernel.
   - Used PI acquisition function, adopting a strategy of exploitation in promsing regions.
   - Constructed small grids around points with the highest output and assessed PI scores to determine next query. The grids had no more than two values in any dimension, and represented discretised small perturbations from the observed data points. The bounds for the grid are equal to half the GP's length scales in each dimension, with an upper bound of 0.05.
   - Identified an additional promising region, which was exploited. A local maximum was found here.

### Function 4
The input features are four machine learning model hyperparameters. The machine learning model approximates the optimal placing of products across warehouses for a business with high online sales. The output is the difference from the expensive baseline.

All initial data points have a negative output, implying that none of them perform better than the expensive baseline. There is one point that performs signficantly better than the others, around which there may be a promising region to explore.

#### Strategy
1. Initial exploration (Weeks 1-3):
   - Sampled points from midpoints of largest empty spaces in each dimension, assuming that independence between features.
   - No other promising regions identified.
1. Quadratic linear regression (Weeks 4, 6, 7 & 13):
   - Introduced global linear regression models to explain observed data.
   - Quadratic linear regression model provided good fit to data, verified through leave one out and 10-fold cross validation.
   - Querying the peaks of fitted global quadratic linear model consistently led to higher outputs. The final query led to the only point found with a positive output (i.e. it beat the expensive baseline).
   - Eigenvalues from the Hessian matrix showed that the model's peak was fairly round with relatively large curvature. The matrix also showed a relatively low amount of interaction between features.
   - Investigating points along the flattest direction of the model's peak indicated it was likely steep and well-centred.
1. Bayesian Optimisation (Weeks 5 & 8):
   - Adopted global Gaussian Process (GP) surrogate models with Radial Basis Function (RBF) and Matern kernels.
   - Upper Confidence Bound (UCB) acqusition function were used to assess and select candidate points to query to balance exploration with exploitation.
   - Recursive grid search performed to improve resolution.
   - No other promising regions identified.
1. Neural network models and ensembles (Weeks 9-12):
   - A variety of neural network layouts and acquisition functions assessed by comparing training and validation loss function outputs with epoch number. Root Mean Square Error (RMSE) was the chosen loss function metric.
   - After selecting a layout, the ensemble of neural network models was trained on all the data.
   - Generated candidate points according to Latin Hypercube algorithm to provide better coverage of domain with fewer samples.
   - Results from ensemble used to select candidate point for a query. Subsequent queries used the results from the best model in an ensemble.
   - Generalisation was found to be poor as the data set was too small for neural networks and the predicted outputs were unreliable.
   - No other promising regions identified.

### Function 5
The input features are four chemical inputs. The output is the yield of a chemical process in a factory.

There is one point with an output that is significantly higher than its surroundings. Based on domain knowledge of the process, one global maximum is expected. Therefore, a policy of exploitation in the region around this point was followed.

#### Strategy
1. Initial exploitation & Bayesian Optimisation (Weeks 1-4):
   - Adopted Gaussian Process (GP) surrogate models with Radial Basis Function (RBF) kernel.
   - Used Probability of Improvement (PI) acquisition function to exploit region around best observed data point.
   - Outputs increased with successive queries. A promising region was identified as having `x2` and `x3` values that lie close to their upper boundaries.
1. Support Vector Machine (SVM) guided boundary probing (Week 5):
   - Use SVM classification model to identify boundary as mean output.
   - The model separated high and low data points with 100% accuracy.
   - Moved along gradient of decision function in `x2` and `x3` direction to find output had increased from that of the most influential support vector. This confirmed the presence of a promising region at high `x2` and `x3` values.
1. Combination of GP surrogate with k-nearest neighbours models (Week 6):
   - Generated random sets of candidates across the domain which were assessed with a GP surrogate model with RBF kernel and Upper Confidence Bound (UCB) acquisition function. The UCB acquisition function had an additional term to penalise points that lay close to the current best observed point to encourage exploration.
   - Best 5 candidates from each iteration were selected and divided into clusters using the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm. The distance threshold was determined by plotting the distances of each point to its 5th nearest neighbour and estimating the point at which this distance changed significantly (indicated by the position of the elbow).
   - One of the positional medians from each cluster was chosen as the next point to query.
   - Length scales from the GP models suggested that the output was less sensitive to changes in `x0` and `x1`.
   - Output was low due to high GP uncertainty in unexplored regions. Given that a promising region had been identified, a single peak was expected and the relatively small number of remaining queries, this policy of exploration was discontinued.
1. Exploitation with Random Forests (RF) ensemble models (Weeks 7-13):
   - RF and extra trees ensembles were investigated as surrogate model replacements for the GP.
   - RF ensemble models and a combination of Upper Confidence Bound (UCB), Probability of Improvement (PI) and Expected Improvement (EI) acquisition functions were used to assess and select candidate points to query.
   - Means and standard deviations were calculated from the trees that comprised the RF ensemble models.
   - Used hierarchical agglomerative clustering algorithm with distance threshold of $\frac{1}{3}$ to identify clusters. Recursive grid search performed for each cluster to improve resolution.
   - Found best observed point and consistently high outputs at combined high `x2` and `x3` values, in line with conclusions from GP. In this region, the output appears to be insensitive to changes in `x0` and `x1`.
   - Potential second region identified at high `x2` and low `x3` values, which was not investigated further because of limited query budget.

### Function 6
The input fatures are ingredients for a cake recipe. The output is the combined score based on flavour, consistency, calories, waste and cost as judged by an expert taster.

#### Strategy
1. Initial exploitation & Bayesian Optimisation (Weeks 1-5):
   - Adopted Gaussian Process (GP) surrogate models with Radial Basis Function (RBF) kernel.
   - Used Probability of Improvement (PI) acquisition function to exploit region around best observed data point.
   - Recursive grid search performed to improve resolution.
   - Found best observed point.
   - Adopted a more exploratory policy in week 5 as Upper Confidence Bound (UCB) acquisition function replaced PI as assessor of candidate points.
   - Length scales from the GP models suggested that the output was less sensitive to changes in `x0` and `x2`.
1. Combination of GP surrogate with k-nearest neighbours models (Week 6):
   - Generated random sets of candidates across the domain which were assessed with a GP surrogate model with RBF kernel and UCB acquisition function. The UCB acquistiion function had an additional term to penalise points that lay close to the current best observed point to encourage exploration.
   - Best 5 candidates from each iteration were selected and divided into clusters using the Density-Base Spatial Clustering of Applications with Noise (DBSCAN) algorithm. The distance threshold was determined by plotting the distances of each point to its 5th nearest neighbour and estimating the point at which this distance changed significantly (indicated by the position of the elbow).
   - The candidate points formed a single cluster and the candidate with the greatest uncertainty was chosen as the next point to query.
   - Output was low due to high GP uncertainty in explored regions.
1. Exploration with Extremely Randomised Trees (Extra Trees - ET) and Random Forests (RF) ensemble models (Weeks 7-10):
   - ET and RF ensembles were investigated as surrogate model replacements for the GP.
   - The UCB acquisition function was used to assess and select candidate points to query from a recursive grid search.
   - Investigations of the promising region that had previously been identified indicated that the output may be sensitive to changes in all dimensions, in contrast to the conclusion reached at the end of week 5.
   - No other promising regions were identified.
1. Region-based anlaysis with decision trees and GP surrogate models (Week 11):
   - Returned to a policy of exploitation.
   - Used decision tree model to partition domain into regions based on observed output values.
   - Generated candidate points in each region. More candidates were generated in regions with a higher mean output using softmax weighting.
   - Global GP surrogate model and Expected Improvement (EI) acquisition function were used to assess and select a candidate point to query.
   - Found to be a promising method for future queries.
1. Exploitation with RF ensemble models (Weeks 12-13):
   - Returned to training RF ensembles with the more exploitative PI acquisition function as an assessor of candidate points.
   - Candidate points chosen using a recursive grid search.
   - Discovered the instability in the random forests ensemble model as the proposed points varied significantly depending on the random seed used during initialisation. This is likely because of the relatively small number of data points in the dataset relative to the number of dimensions, which can amplify small modelling differences that arise when using different random seeds.
   - Despite this, a steady improvement in output was witnessed.
   - The promising region lies in the space around (0.55, 0.4, 0.4, 0.8, 0.15).
