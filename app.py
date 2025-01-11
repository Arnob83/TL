import sqlite3
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os
from pygam import LogisticGAM, s

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
def prediction(_gam_model, input_data):
    probabilities = _gam_model.predict_proba(input_data)
    prediction = _gam_model.predict(input_data)
    pred_label = 'Approved' if prediction[0] == 1 else 'Rejected'
    return pred_label, probabilities

# Function to create and display the GAM feature importance graph
def show_gam_feature_importance(gam_model, input_data, feature_names):
    st.subheader("Feature Contributions")

    # Partial dependence for each feature
    contributions = []
    for i in range(len(feature_names)):
        partial_dependence = gam_model.partial_dependence(i, input_data)
        contributions.append(partial_dependence[0])

    # Create a DataFrame for visualization
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Contribution': contributions
    }).sort_values(by="Contribution", ascending=False)

    # Highlight positive and negative contributions
    feature_df['Impact'] = feature_df['Contribution'].apply(
        lambda x: 'Positive' if x >= 0 else 'Negative'
    )

    # Plot feature contributions
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = feature_df['Impact'].map({'Positive': 'green', 'Negative': 'red'})
    ax.barh(feature_df['Feature'], feature_df['Contribution'], color=colors)
    ax.set_xlabel("Contribution Magnitude")
    ax.set_ylabel("Features")
    ax.set_title("Feature Contributions to Loan Decision (GAM)")
    plt.gca().invert_yaxis()
    st.pyplot(fig)

    # Add explanations for each feature
    st.subheader("Feature Contribution Explanations")
    for _, row in feature_df.iterrows():
        if row['Impact'] == 'Positive':
            explanation = f"The feature '{row['Feature']}' positively contributed to loan approval."
        else:
            explanation = f"The feature '{row['Feature']}' negatively impacted the loan approval."
        st.write(f"- {explanation} (Contribution: {row['Contribution']:.4f})")

# Main Streamlit app
def main():
    # Initialize database
    init_db()

    # Train a GAM model (for demonstration, use LogisticGAM with dummy smoothing terms)
    gam_model = LogisticGAM(s(0) + s(1) + s(2) + s(3) + s(4))

    # Feature names (replace these with the actual feature names from your dataset)
    feature_names = ["Credit_History", "Education_1", "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]

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

    # Prepare input data for prediction
    input_data = pd.DataFrame(
        [[Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term]],
        columns=feature_names
    )

    # Prediction and database saving
    if st.button("Predict"):
        result, probabilities = prediction(gam_model, input_data)

        # Save data to database
        save_to_database(Gender, Married, Dependents, Self_Employed, Loan_Amount, Property_Area, 
                         Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, 
                         Loan_Amount_Term, result)

        # Display the prediction
        if result == "Approved":
            st.success(f"Your loan is Approved! (Probability: {probabilities[0][1]:.2f})", icon="✅")
        else:
            st.error(f"Your loan is Rejected! (Probability: {probabilities[0][0]:.2f})", icon="❌")

        # Show prediction values
        st.subheader("Prediction Value")
        st.write(input_data)

        # Show feature importance graph and explanations
        show_gam_feature_importance(gam_model, input_data, feature_names)

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
