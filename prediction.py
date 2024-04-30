import streamlit as st
import pickle as pkl
import pandas as pd


with open('RandomForest.pkl', "rb") as model_file:
    model = pkl.load(model_file)
with open('oneHot_encode.pkl', "rb") as oneHot_file:
    oneHot_encoder = pkl.load(oneHot_file)
with open('gender_encode.pkl', "rb") as binary_file:
    gender_encoder = pkl.load(binary_file)
with open('scaler.pkl', "rb") as scaler_file:
    scaler = pkl.load(scaler_file)


def main():
    st.title('Churn Prediction')

    creditScore = st.number_input("Credit score", 300,850)
    geo = st.selectbox("Geography",["France","Germany","Spain"])
    gender = st.radio("Gender",["Male","Female"])
    age = st.number_input("Age",0,100)
    tenure = st.number_input("Tenure (in years)",0,100)
    balance = st.number_input("Balance",0,10000000)
    num_products = st.number_input("Number of products",0,100)
    cc = st.radio("Do you have a credit card?",["Yes","No"])
    active = st.radio("Are you an active member?",["Yes","No"])
    salary = st.number_input("Estimated salary",0,10000000)

    data = {'Credit Score': float(creditScore), 'Geography': geo, 'Gender': gender, 'Age': int(age),
           'Tenure': int(tenure), 'Balance': float(balance), 'Number of Products': int(num_products),
           'HasCreditCard': (1 if cc == "Yes" else 0), 'IsActive': (1 if active == "Yes" else 0), 'Estimated Salary': float(salary)}
    
    df = pd.DataFrame([list(data.values())], columns=['Credit Score','Geography', 'Gender', 'Age','Tenure', 
                                               'Balance', 'Number of Products','HasCreditCard', 
                                               'IsActive', 'Estimated Salary'])
    

    df = df.replace(gender_encoder)

    cat = df[['Geography']]
    cat_enc=pd.DataFrame(oneHot_encoder.transform(cat).toarray(),columns=oneHot_encoder.get_feature_names_out())
    df = df.reset_index(drop=True)
    df = pd.concat([df,cat_enc], axis=1)
    df = df.drop(['Geography'], axis=1)
    
    num = ['Credit Score', 'Balance', 'Estimated Salary']
    for col in num:
        df[col] = scaler.transform(df[[col]])

    if st.button('Make Prediction'):
        prediction = model.predict(df)[0]
        if prediction == 1:
            output = "CHURN"
        else:
            output = "NOT CHURN"
        st.success(f'The prediction is: {output}')


if __name__ == '__main__':
    main()
