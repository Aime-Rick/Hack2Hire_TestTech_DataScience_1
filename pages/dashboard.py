import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Model Performance Dashboard",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("processing.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, preprocessing = load_model()

@st.cache_resource
def load_data():
    with open("c1.pkl", "rb") as f:
        roc_curve = pickle.load(f)
    with open("c2.pkl", "rb") as f:
        confusion_matrix = pickle.load(f)
    with open("c3.pkl", "rb") as f:
        bar_chart = pickle.load(f)
    return roc_curve, confusion_matrix, bar_chart

roc_curve_plot, confusion_matrix_plot, bar_chart  = load_data()

st.title("Interactive Model Performance Dashboard")
st.markdown(
    """
    Welcome to the **Model Performance Dashboard**. This dashboard helps you understand 
    the performance of the credit scoring model with metrics like ROC Curve, Confusion Matrix, 
    and Credit Score Distribution. 
    """
)

st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Go to", ["ROC Curve", "Confusion Matrix", "Credit Score Distribution"])

if section == "ROC Curve":
    st.subheader("📉 ROC Curve")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(roc_curve_plot, use_container_width=True)

    with col2:
        st.write("### AUC Score")
        st.metric(label="ROC AUC", value=0.71)

elif section == "Confusion Matrix":
    st.subheader("🧮 Confusion Matrix")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(confusion_matrix_plot, use_container_width=True)
        
    with col2:
        st.write("### Classification Metrics")
        st.metric(label="F1 score", value=0.81)
        st.metric(label="Precision", value=0.82)
        st.metric(label="Recall", value=0.81)
        st.metric(label="F-beta score (beta=0.5)", value=0.82)

elif section == "Credit Score Distribution":
    st.subheader("📊 Credit Score Distribution")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(bar_chart, use_container_width=True)

    with col2:
        st.write("### Summary")
        st.write("Total Predictions: 200")
        st.dataframe([{"Risk Class": "Good Credit", "Count":139},{"Risk Class": "Bad Credit", "Count":61} ], use_container_width=True)

