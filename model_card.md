# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
– Person or organization developing model
    - Name: Ahmed Zareef
– Model date
    - Date: 02-March-2022
– Model version
    - Version: 1.0
– Model type: Classification
- Model name : Logistic Regression
- Algorithm: Logistic Regression
- Features used:
    - age
    - workclass
    - fnlwgt
    - education
    - education_num
    - marital_status
    - occupation
    - relationship

## Intended Use
– Primary intended uses 
    - Use: Predictive Analytics
– Primary intended users
    - User: Government Agencies or HR and Finance Departments 
    

## Training Data
– Training data source: https://archive.ics.uci.edu/ml/datasets/census+income
## Evaluation Data
– Evaluation data source: https://archive.ics.uci.edu/ml/datasets/census+income
## Metrics
Precision: 0.8550295857988166
Recall: 0.1829113924050633
Fbeta: 0.30135557872784147

## Ethical Considerations
Data privacy should be maintained and assured to prevent personal data from being exposed


## Caveats and Recommendations
Users who are going to use the output of this model should understand that scoring and predictions are based on technical scoring formulas. Probability of having mistakes in scoring should be clarified. And understanding that automated scoring doesn't consider any human external factors into consideration. Recommended to be used in alignment with human review at least in the beginning