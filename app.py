import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries [cite: 32-39]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Metrics 
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
)

st.set_page_config(page_title="BITS Machine Learning Assignment 2", layout="wide")

st.title("Classification Model Performance Dashboard")
st.write("M.Tech (AIML/DSE) - Machine Learning Assignment 2 [cite: 4]")

# Step 1: Dataset Upload [cite: 91]
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type="csv")

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # --- DATA CLEANING (Fixes your ValueError) ---
    # 1. Drop rows with missing values
    df = df.dropna()
    
    # 2. Convert text columns to numbers automatically (Encoding)
    # Models require numeric input [cite: 32-39]
    df = pd.get_dummies(df, drop_first=True)
    
    # Define X and y (Assumes last column is Target)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Ensure Target is numeric for XGBoost/AUC
    if not np.issubdtype(y.dtype, np.number):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Model Selection [cite: 92]
    st.sidebar.header("2. Model Settings")
    model_choice = st.sidebar.selectbox(
        "Choose Model", 
        ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost")
    )

    # Initialize selected model [cite: 34-39]
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "KNN":
        model = KNeighborsClassifier()
    elif model_choice == "Naive Bayes":
        model = GaussianNB()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "XGBoost":
        model = XGBClassifier(eval_metric='logloss')

    # Step 3: Train and Display Metrics [cite: 93, 94]
    if st.sidebar.button("Run Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate All 6 Required Metrics [cite: 41-46]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # AUC needs probability scores
        try:
            y_prob = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        except:
            auc = 0.0  # Fallback if model doesn't support predict_proba

        # Display Metrics in Columns [cite: 93]
        st.subheader(f"Evaluation Metrics: {model_choice}")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Accuracy", f"{acc:.4f}")
        m_col1.metric("AUC Score", f"{auc:.4f}")
        m_col2.metric("Precision", f"{prec:.4f}")
        m_col2.metric("Recall", f"{rec:.4f}")
        m_col3.metric("F1 Score", f"{f1:.4f}")
        m_col3.metric("MCC Score", f"{mcc:.4f}")

        # Confusion Matrix [cite: 94]
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)

else:
    st.info("Please upload a CSV dataset to begin. (Ensure it has >12 features and >500 rows) [cite: 30]")
