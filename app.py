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

def explain_prediction_with_lime(input_data, prediction_label):
    # Define a proxy dataset (representative data for perturbations)
    proxy_data = pd.DataFrame({
        "Credit_History": [1, 0, 1, 0],
        "Education_1": [0, 1, 0, 1],
        "ApplicantIncome": [3000, 4000, 5000, 2000],
        "CoapplicantIncome": [0, 1500, 0, 1000],
        "Loan_Amount_Term": [360, 180, 120, 240]
    })

    # Ensure input data features are in the correct order
    feature_names = classifier.feature_names_in_
    input_data = input_data[feature_names]

    # Initialize LIME explainer with proxy data
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=proxy_data[feature_names].values,
        feature_names=feature_names,
        class_names=['Rejected', 'Approved'],
        mode='classification'
    )

    # Convert input_data to numpy array
    input_array = input_data.values

    # Determine the index for the predicted class
    class_index = 1 if prediction_label == "Approved" else 0

    # Define a predict function to ensure input has valid feature names
    def predict_fn(x):
        # Convert x (numpy array) to DataFrame with correct feature names
        x_df = pd.DataFrame(x, columns=feature_names)
        return classifier.predict_proba(x_df)

    # Generate explanation for the prediction
    explanation = explainer.explain_instance(
        data_row=input_array[0],  # Use the first row of the input data
        predict_fn=predict_fn,
        labels=[class_index]  # Specify the predicted class index
    )

    # Extract feature contributions for the predicted class
    contributions = explanation.local_exp[class_index]

    # Sort contributions by importance
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    # Prepare data for plotting
    sorted_features = [feature_names[i] for i, _ in contributions]
    sorted_values = [val for _, val in contributions]

    # Create a bar plot
    plt.figure(figsize=(8, 5))
    colors = ['green' if val > 0 else 'red' for val in sorted_values]
    plt.barh(sorted_features, sorted_values, color=colors)
    plt.xlabel("Feature Contribution to Prediction")
    plt.ylabel("Features")
    plt.title(f"Local Explanation for Class {prediction_label}")
    plt.tight_layout()

    return plt.gcf()




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
lime_fig = explain_prediction_with_lime(input_data, prediction_label=result)
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
