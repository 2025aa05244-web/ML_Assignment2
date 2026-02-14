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
from sklearn.impute import SimpleImputer # Added for robustness

# Metrics [cite: 40-46]
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
)

st.title("Classification Model Performance Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type="csv") [cite: 91]

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # --- ROBUST DATA CLEANING ---
    # 1. Separate Features and Target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 2. Handle Categorical Data (Encoding)
    X = pd.get_dummies(X, drop_first=True)

    # 3. Handle Missing Values (Imputation instead of dropping)
    # This ensures you don't end up with 0 rows
    imputer = SimpleImputer(strategy='mean')
    if not X.empty:
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns)
    
    # 4. Safety Check: Ensure we meet assignment minimums 
    if len(df) < 500 or len(X.columns) < 12:
        st.error(f"Dataset too small! Current rows: {len(df)}, Features: {len(X.columns)}. Requirement: 500 rows, 12 features.") [cite: 30]
    else:
        # Proceed only if data is valid
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_choice = st.sidebar.selectbox("Choose Model", 
            ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost")) [cite: 92]

        # Model Mapping [cite: 34-39]
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(eval_metric='logloss')
        }

        if st.sidebar.button("Run Model"):
            model = models[model_choice]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics Calculation [cite: 40-46, 93]
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Display metrics [cite: 93]
            st.success(f"Model {model_choice} executed successfully!")
            cols = st.columns(3)
            cols[0].metric("Accuracy", f"{acc:.4f}")
            cols[1].metric("F1 Score", f"{f1:.4f}")
            cols[2].metric("MCC Score", f"{mcc:.4f}")

            # Visualization [cite: 94]
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
            st.pyplot(fig)
else:
    st.info("Please upload a CSV file to begin.")
