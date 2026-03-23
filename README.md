# AICapstoneProject_IC

## Project Overview
The Black Box Optimisation (BBO) capstone project consists of 8 optimisation problems where the underlying functions are unknown. The goal is to find the global maxima through limited evaluations of each function. This mirrors the limitations in many real world machine learning challenges, where evaluations can be expensive and extensive searches across the entire phase space are not possible. By identifying regions of interest, and focusing evaluations there, much of the relevant structure of the underlying functions can be inferred and maxima can be identified.

This project exhibits problem-solving skills, the ability to work with limited data and within constraints, the extraction of information from data and the flexibility of changing approach in light of new evidence, and documentation and communication skills.

## Setup instructions
From the command line:
```
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

Initial observations of the data indicate there are negative values, which make little physical sense and therefore may be noise. The positive values are rather flat, and there does not appear to be a clear peak. Therefore, it was decided to focus on exploration over exploitation over the first 3 weeks, querying coordinates farthest away from any known points or corners.

No clear patterns emerged from this initial phase. Attempts to fit linear regression models with linear and quadratic functions was attempted during week 4. As there are few data points, k-fold cross-validation was performed with a validation set of size 1. The results were not promising. Therefore, Bayesian Optimisation was implemented in week 4, using a Gaussian Process surrogate model with a Radial Basis Function kernel. An Upper Confidence Bound acquisition function with an exploration parameter of 1.96 was used to select the next point to query.

it may be worth revisiting the linear regression model in the future. Additionally, using a Support Vector Machine model to identify promising regions may yield dividends. For the time being, an exploration strategy seems to be better.
