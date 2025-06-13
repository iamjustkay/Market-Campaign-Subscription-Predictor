import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Market Campaign Predictor")
st.title("💰 Market Campaign Subscription Predictor")

# Input fields
fields = {
    'age': st.number_input("Age", min_value=18, max_value=100, value=30),
    'job': st.selectbox("Job", ["admin.", "technician", "blue-collar", "services", "management", "retired"]),
    'marital': st.selectbox("Marital Status", ["married", "single", "divorced"]),
    'education': st.selectbox("Education Level", ["secondary", "tertiary", "primary", "unknown"]),
    'default': st.selectbox("Credit in Default?", ["yes", "no"]),
    'balance': st.number_input("Average Yearly Balance (EUR)", value=1000.0),
    'housing': st.selectbox("Housing Loan?", ["yes", "no"]),
    'loan': st.selectbox("Personal Loan?", ["yes", "no"]),
    'contact': st.selectbox("Contact Communication Type", ["cellular", "telephone"]),
    'day': st.number_input("Last Contact Day of the Month", min_value=1, max_value=31, value=15),
    'month': st.selectbox("Last Contact Month", 
                          ["jan", "feb", "mar", "apr", "may", "jun", 
                           "jul", "aug", "sep", "oct", "nov", "dec"]),
    'duration': st.number_input("Last Contact Duration (seconds)", value=120),
    'campaign': st.number_input("Number of Contacts During Campaign", value=1),
    'pdays': st.number_input("Days Since Last Contact", value=-1),
    'previous': st.number_input("Previous Contacts", value=0),
    'poutcome': st.selectbox("Previous Campaign Outcome", ["unknown", "success", "failure", "other"])
}

if st.button("Predict Subscription"):
    with st.spinner("Sending data to model..."):
        try:
            # POST request to FastAPI backend
            response = requests.post("http://localhost:8000/predict/", json=fields)
            response.raise_for_status()  # raise exception for 4xx/5xx errors

            result = response.json()

            st.success("✅ Prediction complete!")
            st.markdown(f"**Subscribed:** {'✅ Yes' if result['subscribed'] else '❌ No'}")
            st.markdown(f"**Probability of Subscription:** `{result['probability']:.2%}`")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error: Could not connect to prediction server.\n{e}")
