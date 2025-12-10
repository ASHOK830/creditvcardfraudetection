import streamlit as st
import pandas as pd
import pickle

# Load model and column names
model = pickle.load(open("fraud_model.pkl", "rb"))
model_cols = pickle.load(open("model_columns.pkl", "rb"))

st.title("💳 Online Fraud Detection App")

st.write(
    """
This app uses a **Logistic Regression model** trained on the Kaggle Credit Card Fraud dataset.

You can:
- 📁 Upload a CSV file of transactions to check many rows at once  
- ⚡ Or enter a **single transaction manually** for real-time fraud prediction
"""
)

# -----------------------------
# MODE 1: CSV UPLOAD (BATCH)
# -----------------------------
st.header("📁 1. Upload CSV file (Batch Prediction)")

uploaded = st.file_uploader("Upload CSV file with transactions", type=["csv"])

if uploaded is not None:
    # Read uploaded CSV
    data = pd.read_csv(uploaded)
    st.subheader("Uploaded Data (first 10 rows)")
    st.write(data.head(10))

    try:
        # Use only the columns the model was trained on
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
        st.error("❌ Your CSV does not have the correct columns expected by the model.")
else:
    st.info("Upload a CSV file above for batch prediction (optional).")

# -----------------------------
# MODE 2: SINGLE TRANSACTION (REAL-TIME)
# -----------------------------
st.header("⚡ 2. Real-Time Single Transaction Prediction")

st.write(
    """
Enter the feature values for **one transaction** below.
These should match the columns used in the dataset (Time, V1–V28, Amount).
"""
)

# Create input fields dynamically for all model columns
user_input = {}
with st.form("single_transaction_form"):
    st.subheader("Enter Transaction Details")

    for col in model_cols:
        # All features are numeric in creditcard.csv
        user_input[col] = st.number_input(
            f"{col}",
            value=0.0,
            format="%.5f"
        )

    submitted = st.form_submit_button("🔍 Predict Fraud for this Transaction")

if submitted:
    # Convert dict to DataFrame with one row
    input_df = pd.DataFrame([user_input])
    # Ensure columns are in correct order
    input_df = input_df[model_cols]

    # Make prediction
    pred_class = model.predict(input_df)[0]  # 0 or 1
    pred_proba = model.predict_proba(input_df)[0][1]  # probability of class 1 (fraud)

    if pred_class == 1:
        st.error(f"🚨 This transaction is predicted as **FRAUD**.\n\nFraud probability: **{pred_proba:.4f}**")
    else:
        st.success(f"✅ This transaction is predicted as **SAFE**.\n\nFraud probability: **{pred_proba:.4f}**")
