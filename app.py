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
    f1_score, matthews_corrcoef, confusion_matrix
)

st.set_page_config(page_title="BITS Assignment 2 Dashboard", layout="wide")
st.title("Classification Performance Dashboard")

st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type="csv") 

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Validation Check (BITS Requirement)
        if df.shape[0] < 500 or df.shape[1] < 12:
            st.error(f"Dataset too small! Found {df.shape[0]} rows and {df.shape[1]} columns.")
        else:
            # --- ULTIMATE SAFE CLEANING ---
            
            # 1. Fill ALL missing values in the entire dataframe
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        df[col] = df[col].fillna("Unknown")

            # 2. Separate Features and Target
            target_name = df.columns[-1]
            X = df.drop(columns=[target_name])
            y = df[target_name]

            # 3. Encoding
            X = pd.get_dummies(X, drop_first=True)
            
            # Use factorize to ensure y is numeric and has no NaNs
            y_encoded, _ = pd.factorize(y)
            y = pd.Series(y_encoded)

            # 4. Final Safety Check before splitting
            if len(X) > 0:
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
                    
                    # Metrics
                    st.subheader(f"Results: {model_name}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                    c2.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
                    c3.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")
                    
                    # Confusion Matrix
                    fig, ax = plt.subplots()
                    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='plasma')
                    st.pyplot(fig)
            else:
                st.error("Data processing resulted in zero features. Check your CSV format.")

    except Exception as e:
        st.error(f"Execution Error: {e}")
else:
    st.info("Awaiting CSV upload...")
