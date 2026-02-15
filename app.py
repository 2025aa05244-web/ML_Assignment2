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

# All 6 Required Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
)

st.set_page_config(page_title="BITS Assignment 2 - Comprehensive ML", layout="wide")
st.title("Machine Learning Classification Dashboard")
st.write("M.Tech (AIML/DSE) | Evaluation of Multiple Classification Models")

# Sidebar for file upload
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Wine Quality or similar CSV", type="csv") 

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validation Check (BITS Requirement: 500+ rows, 12+ features)
        if df.shape[0] < 500 or df.shape[1] < 12:
            st.warning(f"Note: Dataset has {df.shape[0]} rows and {df.shape[1]} columns. Ensure this meets your specific project selection criteria.")
        
        # --- DATA PRE-PROCESSING ---
        target_col = df.columns[-1]
        
        # Clean missing values
        df = df.fillna(df.median(numeric_only=True))
        for col in df.select_dtypes(exclude=[np.number]).columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        # Separate Features and Target
        X = df.drop(columns=[target_col])
        X = pd.get_dummies(X, drop_first=True)
        
        # Factorize target for consistent numeric classes (0, 1, 2...)
        y_encoded, y_cats = pd.factorize(df[target_col])
        y = pd.Series(y_encoded)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.sidebar.header("2. Model Selection")
        model_name = st.sidebar.selectbox("Choose Model", 
            ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"))

        # Initialize Models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(
                                        use_label_encoder=False, 
                                        eval_metric='logloss',
                                        objective='binary:logistic' if len(np.unique(y)) == 2 else 'multi:softprob'
                                    )
        }


        if st.sidebar.button("Run Detailed Evaluation"):
            model = models[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # XGBoost specific AUC handling
            try:
                if len(np.unique(y)) == 2:
                    # Binary classification needs the probability of the positive class
                    y_prob = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_prob)
                else:
                    # Multi-class needs the 'ovr' (One-vs-Rest) strategy
                    y_prob = model.predict_proba(X_test)
                    auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            except Exception as e:
                st.warning(f"AUC could not be calculated for this model: {e}")
                auc = 0.0


            # --- DISPLAY ALL 6 METRICS ---
            st.subheader(f"Results for {model_name}")
            
            # First Row: Core Accuracy & AUC
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")
            
            # Second Row: Precision, Recall, F1
            col4, col5, col6 = st.columns(3)
            col4.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.4f}")
            col5.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")
            col6.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
            
            # --- VISUALIZATION ---
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a CSV file to begin. Ensure your target is the last column.")

