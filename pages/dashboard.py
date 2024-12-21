import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import auc, confusion_matrix, roc_curve, precision_score, recall_score, f1_score, fbeta_score
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

st.title("Interactive Model Performance Dashboard")
st.markdown(
    """
    Welcome to the **Model Performance Dashboard**. This dashboard helps you understand 
    the performance of the credit scoring model with metrics like ROC Curve, Confusion Matrix, 
    and Credit Score Distribution. 
    """
)

@st.cache_data
def load_test_data():
    X_test = pd.read_csv("data/X_test.csv")  
    y_test = pd.read_csv("data/y_test.csv") 
    return X_test, y_test

X_test, y_test = load_test_data()

X_test_1 = X_test.copy()
X_test = preprocessing.transform(X_test)
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Go to", ["ROC Curve", "Confusion Matrix", "Credit Score Distribution"])

if section == "ROC Curve":
    st.subheader("📉 ROC Curve")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve',
            line=dict(color='green', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Guess',
            line=dict(color='gray', dash='dash')
        ))

        fig.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend_title="Legend",
            width=600,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("### AUC Score")
        st.metric(label="ROC AUC", value=f"{roc_auc:.2f}")

elif section == "Confusion Matrix":
    st.subheader("🧮 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"]
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.imshow(
            cm_df,
            text_auto=True,  
            color_continuous_scale="Blues",  
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        fig.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=600, height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("### Confusion Matrix Values")
        st.dataframe(cm_df, use_container_width=True)

        st.write("### Classification Metrics")
        st.write("f1-score: ", f1_score(y_test, y_pred).round(2))
        st.write("Precision: ", precision_score(y_test, y_pred).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred).round(2))
        st.write("F-beta Score (beta=0.5): ", fbeta_score(y_test, y_pred, beta=0.5).round(2))

elif section == "Credit Score Distribution":
    st.subheader("📊 Credit Score Distribution")

    counts = pd.Series(y_pred).value_counts().rename_axis('Risk Class').reset_index(name='Count')
    counts["Risk Class"] = counts["Risk Class"].map({0: "Bad Credit", 1: "Good Credit"})

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(
            counts,
            x="Risk Class",
            y="Count",
            color="Risk Class",
            title="Credit Score Distribution",
            text="Count", 
            color_discrete_map={"Bad Credit": "red", "Good Credit": "green"},  
        )

        fig.update_layout(
            xaxis_title="Risk",
            yaxis_title="Count",
            showlegend=False,  
            height=600,
            width=600
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')  # Text above bars

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("### Summary")
        st.write(f"Total Predictions: {len(y_pred)}")
        st.dataframe(counts, use_container_width=True)

