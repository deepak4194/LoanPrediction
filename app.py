import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Function 1: Load and preprocess the dataset
def load_data():
    df = pd.read_csv('train.csv')
    df = df[['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Gender', 'Loan_Status']]
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})  # Convert Gender to numeric
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})  # Convert Loan_Status to numeric
    df = df.dropna()  # Drop missing values
    return df

# Function 2: Train the Random Forest model
def train_model(df):
    X = df[['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Gender']]
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    return model

# Function 3: Make prediction
def predict(model, income, loan_amount, credit_history, gender):
    prediction = model.predict([[income, loan_amount, credit_history, gender]])
    return prediction

# Function 4: User Interface
def loan_prediction_app():
    # UI elements
    st.markdown(""" 
        <style>
        body { font-family: 'Arial', sans-serif; background-color: #f0f2f6; }
        .main { background-color: white; padding: 10px; border-radius: 8px; max-width: 600px; margin: auto; }
        .header { text-align: center; padding: 10px; background-color: #4CAF50; color: white; }
        </style>
        <div class="main">
            <div class="header">
                <h1>Loan Eligibility Prediction</h1>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Call function to get user inputs
    income = st.number_input('Applicant Income', min_value=0)
    loan_amount = st.number_input('Loan Amount', min_value=0)
    credit_history = st.selectbox('Credit History (1: Good, 0: Bad)', (1, 0))
    gender = st.selectbox('Gender (1: Male, 0: Female)', (1, 0))

    # Step 1: Call load_data() to load and preprocess the data
    df = load_data()  # This will load and clean the data

    # Step 2: Call train_model(df) to train the Random Forest model
    model = train_model(df)  # This will train the model using the preprocessed data

    # Step 3: When the user clicks the 'Predict' button, call the predict() function
    if st.button('Predict'):
        result = predict(model, income, loan_amount, credit_history, gender)  # Make prediction based on input
        
        # Step 4: Display the result based on the prediction
        if result == 1:
            st.markdown(""" 
                <div class="main">
                    <h3 style="color:green;">Loan Approved!</h3>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(""" 
                <div class="main">
                    <h3 style="color:red;">Loan Rejected!</h3>
                </div>
            """, unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == '__main__':
    loan_prediction_app()   # Call the main app function to run the interface
