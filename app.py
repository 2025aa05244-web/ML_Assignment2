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

# Metrics [cite: 40-46]
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="BITS Assignment 2", layout="wide")
st.title("ML Model Evaluation Dashboard")

# sidebar for file upload [cite: 91]
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type="csv") 

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # 1. Validation Check: Instance and Feature Size [cite: 30]
    num_rows, num_cols = df.shape
    
    if num_rows < 500 or num_cols < 12:
        st.error(f"Dataset too small! Found {num_rows} rows and {num_cols} columns. (Need 500+ rows and 12+ features).")
    else:
        st.success(f"Dataset Loaded: {num_rows} instances, {num_cols} features.")
        
        # 2. Robust Data Preprocessing (Fixes the Shape Mismatch)
        # Separate Target [cite: 29]
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Fill missing values: Numeric with mean, Categorical with most_frequent
        num_cols_list = X.select_dtypes(include=[np.number]).columns
        cat_cols_list = X.select_dtypes(exclude=[np.number]).columns

        if len(num_cols_list) > 0:
            imputer_num = SimpleImputer(strategy='mean')
            X[num_cols_list] = imputer_num.fit_transform(X[num_cols_list])
        
        if len(cat_cols_list) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            X[cat_cols_list] = imputer_cat.fit_transform(X[cat_cols_list])

        # Convert categorical text to numbers [cite: 32]
        X = pd.get_dummies(X, drop_first=True)
        
        # Ensure y is numeric [cite: 39]
        if not np.issubdtype(y.dtype, np.number):
            y = pd.factorize(y)[0]

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Model Selection [cite: 92]
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

        # 4. Run Analysis [cite: 40]
        if st.sidebar.button("Run Model Evaluation"):
            model = models[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Metrics [cite: 41-46, 93]
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Display
            st.subheader(f"Results for {model_name}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("Precision", f"{prec:.4f}")
            c3.metric("F1 Score", f"{f1:.4f}")
            
            # Confusion Matrix [cite: 94]
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu')
            st.pyplot(fig)
else:
    st.info("Please upload your CSV file in the sidebar to begin.")
