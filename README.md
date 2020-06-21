# Credit-Risk Analysis Using Machine Learning

## Objective

Create machine learning process to access and evaluate credit risk, and determine the best learners between ENSEMBLE and RESAMPLE for identifying risk levels.

## Resources

python, Jupyter notebook, machine learning, github

## Results

### Ensemble Learner

First we look at the ENSEMBLE learner to identify and catching fraudulant credit applications and risk levels. 

##### Accuracy Score 

0.49797771814131836

##### Confusion Matrix
  
  | Positive  | Negative |
| ------------- | ------------- |
| 93            | 8             |
| 983           | 16121         |

   
##### Classification Report
                  pre       rec       spe        f1       geo       iba       sup

  high_risk       0.09      0.92      0.94      0.16      0.93      0.87       101
   low_risk       1.00      0.94      0.92      0.97      0.93      0.87     17104

avg / total       0.99      0.94      0.92      0.97      0.93      0.87     17205

##### Obersvation

The precision on detecting high risk is low at 0.09, yet Recall is very optimal at 0.92.
The precision on detecting low is 100% and recall is also 94%
Our F1 performed poorly on High risk 0.16 but did well with low risk at 0.97

This learning method did not yeild the best accuracy results for detecting high risk. Many false positives were captured leading to skewed results.


### Resample Learner

Next we look at Resampling method on the same dataset

##### Accuracy Score

0.6557032574164807

##### Confusion Matrix
|Positive| Negative|
|-------------|-------------|
 |  73, |   28|
  |7036| 10068|

##### Classification Report

|              | pre           |rec            |spe              |f1           |geo |iba |sup |
| ------------- | ------------- |------------- | ------------- |------------- | ------------- |------------- | ------------- |
|  high_risk     |  0.01    |  0.72    |  0.59    |  0.02   |   0.65 |     0.43  |     101|
 |  low_risk   |    1.00   |   0.59   |   0.72   |   0.74   |   0.65  |    0.42  |   17104|
|avg / total  |     0.99  |    0.59   |   0.72   |   0.74  |    0.65   |   0.42  |   17205|

### Recommendation : ENSEMBLE LEARNING

With the available data set and the learning capabilities of the two learning methods we use, ENSEMBLE yields better results than that of RESAMPLE. The ability to detect (RECALL) high risk is better. Also our F1 score which is the weighted average of the true positive rate is better is ENSEMBLE


