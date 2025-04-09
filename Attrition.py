import streamlit as st
import pandas as pd
import pickle

# Load models and encoders
with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("decision_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Page config
st.set_page_config(page_title="Attrition Predictor", layout="centered")

# Sidebar navigation
st.sidebar.title("🔀 Navigation")
page = st.sidebar.radio("Choose model:", ["KNN Model", "Decision Tree Model"])

# --- KNN PAGE ---
if page == "KNN Model":
    st.title("🧠 KNN Attrition Predictor")
    st.markdown("**Features:** BusinessTravel, JobInvolvement, MaritalStatus")

    # Inputs
    bt = st.selectbox("Business Travel", label_encoders["BusinessTravel"].classes_, key="bt")
    ji_knn = st.selectbox("Job Involvement (1 = Low, 4 = High)", options=[1, 2, 3, 4], index=2, key="ji_knn")
    ms = st.selectbox("Marital Status", label_encoders["MaritalStatus"].classes_, key="ms")

    # Prepare input
    input_knn = pd.DataFrame([{
        "BusinessTravel": label_encoders["BusinessTravel"].transform([bt])[0],
        "JobInvolvement": ji_knn,
        "MaritalStatus": label_encoders["MaritalStatus"].transform([ms])[0]
    }])

    if st.button("Predict with KNN"):
        pred = knn_model.predict(input_knn)[0]
        proba = knn_model.predict_proba(input_knn)[0][1]
        if pred == 1:
            st.error(f"⚠️ Likely to leave (Probability: {proba:.2f})")
        else:
            st.success(f"✅ Likely to stay (Probability of leaving: {proba:.2f})")

elif page == "Decision Tree Model":
    st.title("🌳 Decision Tree Attrition Predictor")
    st.markdown("**Features:** OverTime, JobInvolvement, YearsAtCompany")

    # Inputs
    ot = st.selectbox("OverTime", label_encoders["OverTime"].classes_, key="ot")
    ji_dt = st.selectbox("Job Involvement (1 = Low, 4 = High)", options=[1, 2, 3, 4], index=2, key="ji_dt")
    yac = st.number_input("Years at Company", min_value=0, max_value=40, value=3, key="yac")

    # Prepare input
    input_dt = pd.DataFrame([{
        "OverTime": label_encoders["OverTime"].transform([ot])[0],
        "JobInvolvement": ji_dt,
        "YearsAtCompany": yac
    }])

    if st.button("Predict with Decision Tree"):
        pred = dt_model.predict(input_dt)[0]
        proba = dt_model.predict_proba(input_dt)[0][1]  # fixed this line
        if pred == 1:
            st.error(f"⚠️ Likely to leave (Probability: {proba:.2f})")
        else:
            st.success(f"✅ Likely to stay (Probability of leaving: {proba:.2f})")
