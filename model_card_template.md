# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
– Person or organization developing model
    - Name: Hazem
– Model date
    - Date: 2020-08-21
– Model version
    - Version: 1.0
– Model type
    - Type: Classification
    - Model name : Logistic Regression
– Information about training algorithms, parameters, fair-
ness constraints or other applied approaches, and features
used in the model
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
– Training data source
    - Source: https://archive.ics.uci.edu/ml/datasets/census+income
## Evaluation Data
– Evaluation data source
    - Source: https://archive.ics.uci.edu/ml/datasets/census+income
## Metrics
- precision : 0.738944365192582
- recall : 0.2664609053497942
- fbeta1 : 0.3916824196597353

## Ethical Considerations
Please this is a very sensitive data such information should not be shared with any one.
Encrypt the used features since it is a private data.

## Caveats and Recommendations
This model is not recommended to be used on the full employees base since it has low recall, however it can be used in campanes on a subset of the base employees.