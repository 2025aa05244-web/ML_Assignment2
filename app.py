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
from sklearn.impute import SimpleImputer 

# Metrics [cite: 40-46]
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)

# This must come AFTER imports
st.set_page_config(page_title="BITS Assignment 2", layout="wide")
st.title("ML Model Evaluation Dashboard")

# sidebar for file upload 
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type="csv") 

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    
    # 1. Validation Check: Instance and Feature Size 
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    
    if num_rows < 500 or num_cols < 12:
        st.error(f"Dataset does not meet requirements! Found {num_rows} rows and {num_cols} columns. (Need 500+ rows and 12+ features).") [cite: 30]
    else:
        st.success(f"Dataset Loaded: {num_rows} instances, {num_cols} features.") [cite: 30]
        
        # 2. Data Preprocessing
        # Handling missing values via Imputation to prevent "Empty Data" errors
        imputer = SimpleImputer(strategy='most_frequent')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        # Convert categorical text to numbers (One-Hot Encoding)
        df_encoded = pd.get_dummies(df_imputed, drop_first=True)
        
        # Define X and y (Assuming last column is the Target)
        X = df_encoded.iloc[:, :-1]
        y = df_encoded.iloc[:, -1].astype(int) # Ensure target is integer for classifiers

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Model Selection Dropdown [cite: 92]
        st.sidebar.header("Model Selection")
        model_name = st.sidebar.selectbox("Choose a Model", 
            ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost")) [cite: 34-39]

        # Initialize Models [cite: 34-39]
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

        # 4. Run Analysis
        if st.sidebar.button("Run Model Evaluation"):
            model = models[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate Metrics [cite: 40-46, 93]
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # AUC Score handling
            try:
                y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else model.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            except:
                auc = 0.0

            # Display Metrics [cite: 93]
            st.subheader(f"Results for {model_name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.4f}")
            col1.metric("AUC Score", f"{auc:.4f}")
            col2.metric("Precision", f"{prec:.4f}")
            col2.metric("Recall", f"{rec:.4f}")
            col3.metric("F1 Score", f"{f1:.4f}")
            col3.metric("MCC Score", f"{mcc:.4f}")

            # 5. Confusion Matrix & Classification Report [cite: 94]
            st.subheader("Visualizations & Reports")
            c1, c2 = st.columns(2)
            
            with c1:
                st.write("Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
                st.pyplot(fig)
            
            with c2:
                st.write("Classification Report")
                st.text(classification_report(y_test, y_pred))

else:
    st.info("Waiting for CSV file upload. Please use the sidebar.") [cite: 91]


