# Loan Eligibility Prediction

## Project Overview

This project is a web application that predicts loan eligibility based on the applicant's income, loan amount, credit history, and gender. The application uses a Random Forest classifier trained on a dataset containing historical loan data.

## Features

- Predict loan eligibility (Approved/Rejected)
- User-friendly interface for inputting data
- Built using Streamlit and scikit-learn

## Technologies Used

- Python
- Streamlit
- scikit-learn
- pandas
- Random Forest Classifier

## Dataset

The dataset used for this project is `train.csv`, which includes the following columns:
- `ApplicantIncome`: Income of the applicant
- `LoanAmount`: Amount of loan requested
- `Credit_History`: Credit history of the applicant (1: Good, 0: Bad)
- `Gender`: Gender of the applicant (Male/Female)
- `Loan_Status`: Loan status (Y/N)

## How to Run the Application

1. Clone the repository:
   ```bash
   git clone https://github.com/deepak4194/LoanPrediction.git
   cd LoanPrediction
