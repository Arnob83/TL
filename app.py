import sqlite3
import pickle
import streamlit as st
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os
import numpy as np

# URL to the raw Logistic Regression model
url = "https://raw.githubusercontent.com/Arnob83/TL/main/Logistic_Regression_model.pkl"

# Download the Logistic Regression model file and save it locally
response = requests.get(url)
with open("Logistic_Regression_model.pkl", "wb") as file:
    file.write(response.content)

# Load the trained model
with open("Logistic_Regression_model.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)

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
    """)
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
    pred_label = 'Approved' if prediction[0] == 1 else 'Rejected'
    return pred_label, input_data_filtered

# Explanation function using LIME
def explain_prediction_with_lime(input_data):
    # Get the training data and feature names
    feature_names = classifier.feature_names_in_

    # Convert input_data to numpy array
    input_array = input_data.values

    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=input_data.values,  # Use the input data as a proxy for the training data
        feature_names=feature_names,
        class_names=['Rejected', 'Approved'],
        mode='classification'
    )

    # Generate explanation for the prediction
    explanation = explainer.explain_instance(
        data_row=input_array[0],  # Use the first row of the input data
        predict_fn=classifier.predict_proba
    )

    # Plot the explanation
    fig = explanation.as_pyplot_figure()
    plt.tight_layout()
    return fig

# Main Streamlit app
def main():
    # Initialize database
    init_db()

    # App layout
    st.markdown(
        """
        <style>
        .main-container {
            background-color: #f4f6f9;
            border: 2px solid #e6e8eb;
            padding: 20px;
            border-radius: 10px;
        }
        .header {
            background-color: #4caf50;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .header h1 {
            color: white;
        }
        </style>
        <div class="main-container">
        <div class="header">
        <h1>Loan Prediction ML App</h1>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

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

    # Prediction and database saving
    if st.button("Predict"):
        result, input_data = prediction(
            Credit_History,
            Education_1,
            ApplicantIncome,
            CoapplicantIncome,
            Loan_Amount_Term
        )

        # Save data to database
        save_to_database(Gender, Married, Dependents, Self_Employed, Loan_Amount, Property_Area, 
                         Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, 
                         Loan_Amount_Term, result)

        # Display the prediction
        if result == "Approved":
            st.success(f'Your loan is {result}', icon="✅")
        else:
            st.error(f'Your loan is {result}', icon="❌")

        # Explain the prediction with LIME
        st.header("Explanation of Prediction")
        lime_fig = explain_prediction_with_lime(input_data)
        st.pyplot(lime_fig)

    # Download database button
    if st.button("Download Database"):
        if os.path.exists("loan_data.db"):
            with open("loan_data.db", "rb") as f:
                st.download_button(
                    label="Download SQLite Database",
                    data=f,
                    file_name="loan_data.db",
                    mime="application/octet-stream"
                )
        else:
            st.error("Database file not found.")

if __name__ == '__main__':
    main()
