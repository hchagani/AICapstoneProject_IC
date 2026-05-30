# Datasheet for Black Box Optimisation Capstone Project Data Set

This datasheet contains information on the data set used for the Black Box Optimisation (BBO) capstone project that forms part of the requirements for the Professional Certificate in Machine Learning and Artificial Intelligence at Imperial College London.

## Motivation

The data set was created for a BBO challenge that forms part of the requirements for [Imperial College London's Professional Certificate in Machine Learning and Artificial Intelligence](https://execed-online.imperial.ac.uk/professional-certificate-ml-ai). The data set consists of 8 optimisation problems where the underlying functions are unknown. The functions mimic real world problems. The goal is to find the global maxima through limited evaluation of each function.

Each function consists of two to 8 features, termed inputs. Functions are evaluated by submitting queries consisting of input values. A total of 13 queries, one for each week of the project, can be submitted for each function. It can take up to 48 hours for a query to be processed and for the results, known as outputs, to be delivered. The limited number of queries and delay in response mimics real world constraints.

After each submission, the data set is analysed to find a combination of inputs that will either yield more information about the underlying function or search for the global maximum. Various machine learning models are employed in attempts to model the underlying function. A single, unified model is not implemented across all functions, and the models may change in light of new information gained after each submission. Ultimately, this is an exercise in gaining experience using a wide range of models.

## Composition

The data set is located in the [data](./data/) directory and consists of 16 files covering the inputs and outputs for the eight underlying functions. The naming convention for each file is as follows:

```
function_{function_ID}_{I/O}.npy
```
where `function_ID` corresponds to the eight functions, and `I/O` is either `inputs` or `outputs`. The files are in [NumPy binary file format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html).

The inputs are matrices of size `(N, d)` where `N` is the total number of data points and `d` is the number of features. The values of each element are real numbers normalised to lie in the range [0, 1). The outputs are vectors of length `N`, that can consist of any real number.

The composition of the data set is summarised in the table below:

| Function ID | Number of Features | Initial Data Set Size | Input Description | Output Description |
| :-: | :-: | :-: | --- | --- |
| 1 | 2 | 10 | Axes of an area with contamination sources. | Strength of contamination. |
| 2 | 2 | 10 | Machine learning model parameters. | Log likelihood score. |
| 3 | 3 | 15 | Strength of compounds in a drug discovery project. | Inverted severity of side-effects. |
| 4 | 4 | 30 | Hyperparameters for machine learning model that approximates optimal placing of products across warehouses for a business with high online sales. | Difference from expensive baseline. |
| 5 | 4 | 20 | Strength of chemical inputs. | Yield of chemical process in factory. |
| 6 | 5 | 20 | Quantity of ingredients for a cake recipe. | Combined score of flavour, consistency, calories, waste and cost as evaluated by an expert taster. |
| 7 | 6 | 30 | Hyperparameters for a machine learning model. | Model's performance score. |
| 8 | 8 | 40 | Hyperparameters for a machine learning model. | Model's accuracy score. |

After each submission is processed, the inputs and outputs are appended to their corresponding files. Therefore the aforementioned data files contain the complete data set after all evaluations. The notebook for each function, located in the [notebooks](./notebooks/) directory, presents the evolving analysis on a weekly basis by slicing the data set.

The points that comprise the initial data set are not uniformly distributed across any of the function domains. The main purpose of this exercise is to find the global maximum, and given the limited number of queries, no attempt is made outside of the exploratory phases to map out the domain. Therefore, the data set can consist of large regions of the domain that have not been evaluated.

Additionally, given the limited number of queries, only one attempt was made to submit an observed data point for re-evaluation. This was in function 1. Therefore, little is known about stochastic noise in the data set.

The data set is anonymised and feature names have been removed. Therefore, there are no privacy concerns nor sensitive information contained within it.

## Collection Process

A variety of machine learning techniques were employed to generate candidate points for weekly queries. Although either Bayesian Optimisation with a Gaussian Process Surrogate model or a spatial sampling method was initially used to explore each function's domain, these later evolved to other machine learning techniques with each successive query. There was no uniform model used across all functions, neither was there a uniform strategy over time. The weekly outputs from each model dictated the direction of the subsequent analysis.

A high-level summary is presented here. Further details can be found in the individual [notebooks](./notebooks/).

With the exception of functions 5 & 6, which were deemed to have clear regions that could be immediately exploited, an initial process of exploration was conducted. This exploratory phase lasted between 3 and six weeks, and employed either Bayesian Optimisation with a Gaussian Process surrogate model and Upper Confidence Bound (UCB) acquisition function with exploration parameter `k` = 1.96, or a spatial sampling method. This later evolved to either using Bayesian Optimisation with a more exploitative acquisition function such as Probability of Improvement (PI), or employing a different machine learning model.

Other machine learning models that were employed include the combination of classification and regression Gaussian Process surrogate models (function 1), linear regression (function 4), neural networks (functions 4 & 7), decision trees (function 2), random forests ensemble models (functions 5, 6 & 7), and support vector machines to identify low and high output regions (function 5). These had varying degrees of success, but were all used to generate candidate queries at some point during the analysis.

The BBO challenge took place over a period of 13 weeks. One query per function could be submitted every week. Late submissions that were submitted after the end of the week were not penalised and were processed within 48 hours. Once a late submission was processed, another query could be submitted.

The data set is anonymised and feature names have been removed. At the time of writing, the programme allows future students to use the full data set that is presented in this repository. This may change, and future students are requested to seek guidance from the programme facilitators and support staff at Imperial College London before using it.

## Preprocessing & Uses

The data included is in the same format that was supplied. While the format has been maintained, new inputs and outputs have been appended to the files after each submission has been processed. Any transformations, such as converting the outputs to a logarithmic scale or separating data points into high and low output regions, are conducted during the analysis as and when needed, and are not preserved in the data set.

The data set is supplied with the intention to reproduce the strategies employed during this BBO challenge. In this context, it is used in the notebooks for each function, which detail a week-by-week evolution of the analysis of the data after each submission.

The data set is not substantially large enough, nor does it cover enough of the domain to be described as representative. Therefore, it is not suitable for determining the shape of the underlying function to any degree of certainty.

## Distribution & Maintenance

The data set is available in the [data](./data/) directory within this repository and is maintained by the owner of the repository, Hassan Chagani. The data remains available in this repository for all users unless its removal is requested by Imperial College London or any other party. Requests for removal will be considered on a case-by-case basis.

Although the data set is maintained here to illustrate the strategies employed during the BBO challenge, it is not envisaged that there will be any updates to it in the future as the project has been completed. Therefore, it is in its final form.
