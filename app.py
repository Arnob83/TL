import sqlite3
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os
import shap  # For SHAP explanations
from lime.lime_tabular import LimeTabularExplainer  # Import for LIME

# URLs for the model file in your GitHub repository
model_url = "https://raw.githubusercontent.com/Arnob83/TL/main/Logistic_Regression_model.pkl"

# Download the model file and save it locally
model_response = requests.get(model_url)
with open("Logistic_Regression_model.pkl", "wb") as file:
    file.write(model_response.content)

# Load the trained model
with open("Logistic_Regression_model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("loan_data.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS loan_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gender TEXT,
        married TEXT,
        dependents INTEGER,
        self_employed TEXT,
        loan_amount REAL,
        property_area TEXT,
        credit_history TEXT,
        education TEXT,
        applicant_income REAL,
        coapplicant_income REAL,
        loan_amount_term REAL,
        result TEXT
    )
    """
    )
    conn.commit()
    conn.close()

# Save prediction data to the database
def save_to_database(gender, married, dependents, self_employed, loan_amount, property_area, 
                     credit_history, education, applicant_income, coapplicant_income, 
                     loan_amount_term, result):
    conn = sqlite3.connect("loan_data.db")
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO loan_predictions (
        gender, married, dependents, self_employed, loan_amount, property_area, 
        credit_history, education, applicant_income, coapplicant_income, loan_amount_term, result
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (gender, married, dependents, self_employed, loan_amount, property_area, 
          credit_history, education, applicant_income, coapplicant_income, 
          loan_amount_term, result))
    conn.commit()
    conn.close()

# Prediction function
@st.cache_data
def prediction(Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term):
    # Map user inputs to numeric values (if necessary)
    Education_1 = 0 if Education_1 == "Graduate" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1

    # Create input data (all user inputs)
    input_data = pd.DataFrame(
        [[Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term]],
        columns=["Credit_History", "Education_1", "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]
    )

    # Filter to only include features used by the model
    trained_features = classifier.feature_names_in_  # Features used in model training
    input_data_filtered = input_data[trained_features]

    # Model prediction (0 = Rejected, 1 = Approved)
    prediction = classifier.predict(input_data_filtered)
    probabilities = classifier.predict_proba(input_data_filtered)  # Get prediction probabilities

    pred_label = 'Approved' if prediction[0] == 1 else 'Rejected'
    return pred_label, input_data, probabilities, input_data_filtered

# Explain prediction using LIME
def show_lime_explanation(input_data_filtered):
    st.subheader("LIME Explanation")
    
    # Initialize LIME explainer
    explainer = LimeTabularExplainer(
        training_data=input_data_filtered.values,
        feature_names=input_data_filtered.columns,
        class_names=['Rejected', 'Approved'],
        mode='classification'
    )

    # Explain a specific prediction
    explanation = explainer.explain_instance(
        data_row=input_data_filtered.iloc[0],
        predict_fn=classifier.predict_proba
    )
    
    # Visualize explanation
    st.pyplot(explanation.as_pyplot_figure())

# Explain prediction using SHAP
def show_shap_explanation(input_data_filtered):
    st.subheader("SHAP Explanation")
    
    # Initialize SHAP explainer
    explainer = shap.LinearExplainer(classifier, input_data_filtered, feature_perturbation="interventional")
    shap_values = explainer.shap_values(input_data_filtered)

    # Visualize SHAP values using a bar plot
    st.write("SHAP values show the impact of each feature on the model's prediction:")
    shap.summary_plot(shap_values, input_data_filtered, plot_type="bar", show=False)
    st.pyplot(plt.gcf())

# Main Streamlit app
def main():
    # Initialize database
    init_db()

    # User inputs
    Gender = st.selectbox("Gender", ("Male", "Female"))
    Married = st.selectbox("Married", ("Yes", "No"))
    Dependents = st.selectbox("Dependents", (0, 1, 2, 3, 4, 5))
    Self_Employed = st.selectbox("Self Employed", ("Yes", "No"))
    Loan_Amount = st.number_input("Loan Amount", min_value=0.0)
    Property_Area = st.selectbox("Property Area", ("Urban", "Rural", "Semi-urban"))
    Credit_History = st.selectbox("Credit History", ("Unclear Debts", "Clear Debts"))
    Education_1 = st.selectbox('Education', ("Under_Graduate", "Graduate"))
    ApplicantIncome = st.number_input("Applicant's yearly Income", min_value=0.0)
    CoapplicantIncome = st.number_input("Co-applicant's yearly Income", min_value=0.0)
    Loan_Amount_Term = st.number_input("Loan Term (in months)", min_value=0.0)

    if st.button("Predict"):
        result, input_data, probabilities, input_data_filtered = prediction(
            Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term
        )
        
        # Save data to database
        save_to_database(
            Gender, Married, Dependents, Self_Employed, Loan_Amount, Property_Area, 
            Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term, result
        )

        # Display the prediction
        if result == "Approved":
            st.success(f"Your loan is Approved! (Probability: {probabilities[0][1]:.2f})", icon="✅")
        else:
            st.error(f"Your loan is Rejected! (Probability: {probabilities[0][0]:.2f})", icon="❌")

        st.write(input_data)

        # Show explanations
        show_shap_explanation(input_data_filtered)
        show_lime_explanation(input_data_filtered)

if __name__ == "__main__":
    main()
