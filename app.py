import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load saved model and scaler
with open("rf_ltv_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.title("Predict Customer Lifetime Value (LTV)")

st.markdown("""
Enter customer details below to predict their LTV:
""")

# Collect input from user
age = st.number_input("Age", min_value=18, max_value=100, value=30)
location = st.number_input("Location (encoded as integer)", value=1)
income_level = st.number_input("Income Level", min_value=0, max_value=10, value=5)
total_transactions = st.number_input("Total Transactions", min_value=0, value=10)
avg_transaction_value = st.number_input("Average Transaction Value", min_value=0, value=1000)
max_transaction_value = st.number_input("Max Transaction Value", min_value=0, value=1500)
min_transaction_value = st.number_input("Min Transaction Value", min_value=0, value=500)
total_spent = st.number_input("Total Spent", min_value=0, value=10000)
active_days = st.number_input("Active Days", min_value=0, value=200)
last_transaction_days_ago = st.number_input("Last Transaction Days Ago", min_value=0, value=10)
loyalty_points = st.number_input("Loyalty Points Earned", min_value=0, value=500)
referral_count = st.number_input("Referral Count", min_value=0, value=2)
cashback_received = st.number_input("Cashback Received", min_value=0, value=200)
app_usage_freq = st.number_input("App Usage Frequency", min_value=0, value=50)
preferred_payment_method = st.number_input("Preferred Payment Method (encoded)", min_value=0, value=1)
support_tickets = st.number_input("Support Tickets Raised", min_value=0, value=1)
issue_resolution_time = st.number_input("Issue Resolution Time", min_value=0, value=2)
satisfaction_score = st.number_input("Customer Satisfaction Score", min_value=0, max_value=10, value=8)

# Prepare input for prediction
input_data = np.array([age, location, income_level, total_transactions, avg_transaction_value,
                       max_transaction_value, min_transaction_value, total_spent, active_days,
                       last_transaction_days_ago, loyalty_points, referral_count, cashback_received,
                       app_usage_freq, preferred_payment_method, support_tickets,
                       issue_resolution_time, satisfaction_score]).reshape(1, -1)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
predicted_ltv = model.predict(input_scaled)

st.success(f"Predicted Customer LTV: {predicted_ltv[0]:.2f}")
