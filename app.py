import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Loan Eligibility Checker", layout="centered")
st.title("Loan Eligibility Prediction App")
st.write("Provide details in the sidebar to check your loan approval status.")

@st.cache_data
def load_data():
    df = pd.read_csv('loan.csv')
    df.dropna(inplace=True)
    return df

@st.cache_resource
def train_model(df):
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    X = df[['ApplicantIncome', 'LoanAmount']]
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

df = load_data()
model = train_model(df)

# Sidebar Inputs
st.sidebar.header("User Input")
income = st.sidebar.number_input("Applicant Income", min_value=0, max_value=100000, value=5000, step=500)
loan_amt = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0, max_value=1000, value=150, step=10)

# Prediction
if st.button("Predict"):
    prediction = model.predict([[income, loan_amt]])
    result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
    st.success(f"Loan Status: {result}")

    # Visualization
    fig = px.scatter(df, x='ApplicantIncome', y='LoanAmount',
                     color=df['Loan_Status'].map({1: 'Approved', 0: 'Rejected'}),
                     title="Loan Data Scatter Plot")
    fig.add_scatter(x=[income], y=[loan_amt], mode='markers',
                    marker=dict(color='red', size=12), name='Your Input')
    st.plotly_chart(fig)
