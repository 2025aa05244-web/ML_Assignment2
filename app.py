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
from sklearn.impute import SimpleImputer 

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="BITS Assignment 2", layout="wide")
st.title("ML Model Evaluation Dashboard")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type="csv") 

if uploaded_file is not None:
    # Load raw data
    df = pd.read_csv(uploaded_file)
    
    # Check assignment requirements: 500 rows, 12 features
    if df.shape[0] < 500 or df.shape[1] < 12:
        st.error(f"Dataset too small! Found {df.shape[0]} rows, {df.shape[1]} columns.")
    else:
        # --- ROBUST CLEANING FOR BOTH X AND Y ---
        
        # 1. Drop rows where the TARGET is missing (Crucial to fix your current error)
        # We assume the last column is the target
        target_col = df.columns[-1]
        df = df.dropna(subset=[target_col])
        
        # 2. Separate Features and Target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # 3. Handle Missing Values in Features (X)
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns

        if len(num_cols) > 0:
            X[num_cols] = SimpleImputer(strategy='mean').fit_transform(X[num_cols])
        if len(cat_cols) > 0:
            X[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[cat_cols])

        # 4. Convert text to numbers
        X = pd.get_dummies(X, drop_first=True)
        
        # Ensure y is numeric (Label Encoding)
        if not np.issubdtype(y.dtype, np.number):
            y = pd.factorize(y)[0]
        else:
            # Even if numeric, cast to int to be safe for classifiers
            y = y.astype(int)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.sidebar.header("Model Selection")
        model_name = st.sidebar.selectbox("Choose a Model", 
            ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"))

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(eval_metric='logloss')
        }

        if st.sidebar.button("Run Model Evaluation"):
            model = models[model_name]
            # This is where the error was happening; it is now fixed by df.dropna(subset=[target_col])
            model.fit(X_train, y_train) 
            y_pred = model.predict(X_test)
            
            # Display Metrics
            st.subheader(f"Performance: {model_name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            c2.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
            c3.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")
            
            # Confusion Matrix
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)
else:
    st.info("Please upload your CSV file in the sidebar.")
