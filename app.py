import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
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

st.set_page_config(page_title="BITS Assignment 2", layout="wide")
st.title("Classification Performance Dashboard")

st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type="csv") 

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # BITS Requirement Check: 500 rows, 12 features
    if df.shape[0] < 500 or df.shape[1] < 12:
        st.error(f"Dataset too small! Found {df.shape[0]} rows and {df.shape[1]} columns.")
    else:
        # --- SAFE CLEANING STEP ---
        
        # 1. Drop rows with missing TARGET (Last column)
        target_name = df.columns[-1]
        df = df.dropna(subset=[target_name])
        
        # 2. Safe Fill for features
        for col in df.columns[:-1]:
            # If the column is entirely empty, drop it
            if df[col].isnull().all():
                df = df.drop(columns=[col])
                continue
                
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                # Fix for your KeyError: Check if mode exists before accessing [0]
                col_mode = df[col].mode()
                if not col_mode.empty:
                    df[col] = df[col].fillna(col_mode[0])

        # 3. Final Features and Target separation
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # 4. Encoding
        X = pd.get_dummies(X, drop_first=True)
        if not np.issubdtype(y.dtype, np.number):
            y = pd.factorize(y)[0]
        else:
            y = y.astype(int)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.sidebar.header("2. Model Selection")
        model_name = st.sidebar.selectbox("Choose Model", 
            ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"))

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(eval_metric='logloss')
        }

        if st.sidebar.button("Run Evaluation"):
            model = models[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Display Metrics
            st.subheader(f"Results: {model_name}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            c2.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
            c3.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")
            
            # Confusion Matrix
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='viridis')
            st.pyplot(fig)
else:
    st.info("Awaiting CSV upload...")
