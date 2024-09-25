# Churn Prediction
data_D.csv contains records of clients from a bank, which consists of 14 features:
- id
- CustomerId
- Surname
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- churn

The problem is to classify a client's churn, using either Random Forest, or XGBoost.
After data is pre-processed and the models are evaluated, XGBoost is chosen and pickled, along with the encoders and scalers (file format .pkl).
The entirety of the machine learning process is documented in the jupyter notebook file.
Prediction code is then made using python (.py file) and deployed on streamlit. It can be accessed from the following link: https://mduts2602073076.streamlit.app/
