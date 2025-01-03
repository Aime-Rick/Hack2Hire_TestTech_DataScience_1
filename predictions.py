import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import xgboost 

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("processing.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, preprocessing = load_model()

st.title("Credit Scoring Prediction App")
st.write("""
This app predicts the credit scoring class (e.g., good/bad) for a user based on their input data.
""")

st.sidebar.header("User Input Features")
def user_input_features():
    age = st.sidebar.slider("Age", 18, 80, 35)
    sex = st.sidebar.selectbox("Sex", options=["male", "female"])
    job = st.sidebar.slider("Job", 0, 3, 1)
    housing = st.sidebar.selectbox("Housing", options=["own", "rent", "free"])
    saving_accounts = st.sidebar.selectbox("Saving Accounts", options=["little", "moderate","quite rich", "rich"])
    checking_account = st.sidebar.selectbox("Checking Account", options=["little", "moderate","rich"])
    credit_amount = st.sidebar.number_input("Credit Amount (in $)", 250, 50000, 5000)
    duration = st.sidebar.slider("Duration (in month)", 0, 200, 20)
    purpose = st.sidebar.selectbox("Purpose", options=["car", "radio/TV", "furniture/equipment", "business", "education", "repairs", "domestic appliances", "vacation/others"])
    
    data = [age, sex, job, housing, saving_accounts, checking_account, credit_amount, duration, purpose]        
    
    return pd.DataFrame(np.array(data).reshape(1, -1), columns=['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose'])

input_df = user_input_features()

st.subheader("User Input Features")
st.write(input_df)

scaled_input = preprocessing.transform(input_df)

if st.button("Predict Credit Score"):
    prediction = model.predict(scaled_input)[0]
    prediction_prob = model.predict_proba(scaled_input)

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("The user is classified as **Good Credit Risk**.")
    else:
        st.error("The user is classified as **Bad Credit Risk**.")
    
    st.write("Prediction probabilities:")
    st.write(f"Bad Credit: {prediction_prob[0][0]:.2f}, Good Credit: {prediction_prob[0][1]:.2f}")


