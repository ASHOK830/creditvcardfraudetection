import streamlit as st
import pandas as pd
import pickle

# Load model and column names
model = pickle.load(open("fraud_model.pkl", "rb"))
model_cols = pickle.load(open("model_columns.pkl", "rb"))

st.title("💳 Online Fraud Detection App")
st.write("Upload a CSV file with transaction data to check for fraud.")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is not None:
    # Read uploaded CSV
    data = pd.read_csv(uploaded)
    st.subheader("Uploaded Data (first 10 rows)")
    st.write(data.head(10))

    # Use only the columns model was trained on
    try:
        X = data[model_cols]
        preds = model.predict(X)
        data["Prediction"] = preds
        data["Label"] = data["Prediction"].map({0: "Safe", 1: "Fraud"})

        st.subheader("Prediction Results (first 20 rows)")
        st.write(data.head(20))

        fraud_count = int((data["Prediction"] == 1).sum())
        total = len(data)
        st.success(f"🚨 Fraud rows: {fraud_count} out of {total} transactions")

    except KeyError:
        st.error("Your CSV does not have the correct columns expected by the model.")
else:
    st.info("Please upload a CSV file to start.")
