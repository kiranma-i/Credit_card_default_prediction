import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

st.set_page_config(page_title="Credit Card Default Prediction", layout="wide")

@st.cache_data
def load_data():
    csv_path = Path("Credit Card Defaulter Prediction.csv")
    if not csv_path.exists():
        st.error("CSV file not found. Make sure 'Credit Card Defaulter Prediction.csv' exists in the project folder.")
        st.stop()

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def get_summary(df):
    return {
        "Shape": df.shape,
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
    }

def make_charts(df):
    st.subheader("Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="default", data=df, ax=ax)
    ax.set_title("Default Distribution")
    st.pyplot(fig)

    st.subheader("Feature Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(df["AGE"], bins=30, kde=True, ax=ax1)
        ax1.set_title("Age Distribution")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.histplot(df["LIMIT_BAL"], bins=30, kde=True, ax=ax2)
        ax2.set_title("Credit Limit Distribution")
        st.pyplot(fig2)

def load_model():
    model_path = Path("model.pkl")
    if not model_path.exists():
        st.warning("No trained model found yet. Save your trained model as 'model.pkl' to enable prediction.")
        return None
    return joblib.load(model_path)

def prediction_form():
    st.subheader("Prediction Inputs")

    user_input = {
        "LIMIT_BAL": st.number_input("LIMIT_BAL", min_value=0, value=50000),
        "SEX": st.selectbox("SEX", [1, 2]),
        "EDUCATION": st.selectbox("EDUCATION", [1, 2, 3, 4]),
        "MARRIAGE": st.selectbox("MARRIAGE", [0, 1, 2, 3]),
        "AGE": st.slider("AGE", 18, 90, 35),
        "PAY_0": st.slider("PAY_0", -2, 9, 0),
        "PAY_2": st.slider("PAY_2", -2, 9, 0),
        "PAY_3": st.slider("PAY_3", -2, 9, 0),
        "PAY_4": st.slider("PAY_4", -2, 9, 0),
        "PAY_5": st.slider("PAY_5", -2, 9, 0),
        "PAY_6": st.slider("PAY_6", -2, 9, 0),
        "BILL_AMT1": st.number_input("BILL_AMT1", value=0),
        "BILL_AMT2": st.number_input("BILL_AMT2", value=0),
        "BILL_AMT3": st.number_input("BILL_AMT3", value=0),
        "BILL_AMT4": st.number_input("BILL_AMT4", value=0),
        "BILL_AMT5": st.number_input("BILL_AMT5", value=0),
        "BILL_AMT6": st.number_input("BILL_AMT6", value=0),
        "PAY_AMT1": st.number_input("PAY_AMT1", value=0),
        "PAY_AMT2": st.number_input("PAY_AMT2", value=0),
        "PAY_AMT3": st.number_input("PAY_AMT3", value=0),
        "PAY_AMT4": st.number_input("PAY_AMT4", value=0),
        "PAY_AMT5": st.number_input("PAY_AMT5", value=0),
        "PAY_AMT6": st.number_input("PAY_AMT6", value=0),
    }

    return pd.DataFrame([user_input])

def main():
    st.title("Credit Card Default Prediction")
    df = load_data()

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    summary = get_summary(df)
    st.write(summary)

    st.subheader("Data Quality")
    st.write("Missing values:", df.isnull().sum())
    st.write("Duplicate rows:", df.duplicated().sum())

    make_charts(df)

    model = load_model()
    if model is not None:
        input_df = prediction_form()
        if st.button("Predict"):
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            st.success(f"Prediction: {'Default' if prediction == 1 else 'Non-Default'}")
            st.write(f"Probability of Default: {probability:.2%}")

if __name__ == "__main__":
    main()