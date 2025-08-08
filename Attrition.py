import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- LOAD SAVED MODELS & PREPROCESSORS ----------------
knn_model = pickle.load(open("knn_model.pkl", "rb"))
knn_scaler = pickle.load(open("knn_scaler.pkl", "rb"))
knn_encoder = pickle.load(open("onehot_encode.pkl", "rb"))

rf_model = pickle.load(open("knn_mod.pkl", "rb"))
rf_scaler = pickle.load(open("knn_scal.pkl", "rb"))
rf_encoder = pickle.load(open("onehot_encoder.pkl", "rb"))

# ---------------- STREAMLIT APP ----------------
st.sidebar.title("📊 Employee Prediction App")
page = st.sidebar.radio("Select Page", ["Attrition Prediction", "Performance Rating Prediction"])

# ---------------- ATTRITION PAGE ----------------
if page == "Attrition Prediction":
    st.header("🧠 Attrition Predictor")

    # Inputs
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    jl = st.number_input("Job Level", min_value=1, max_value=5, value=2)
    mi = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
    twy = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
    ycr = st.number_input("Years In Current Role", min_value=0, max_value=20, value=2)
    ywcm = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=2)
    ms = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    ot = st.selectbox("OverTime", ["Yes", "No"])

    if st.button("Predict Attrition"):
        # Categorical dataframe
        cat_df = pd.DataFrame([[ms, ot]], columns=['MaritalStatus', 'OverTime'])
        cat_encoded = knn_encoder.transform(cat_df)

        # Numerical dataframe
        num_df = pd.DataFrame([[age, jl, mi, twy, ycr, ywcm]],
                              columns=['Age', 'JobLevel', 'MonthlyIncome',
                                       'TotalWorkingYears', 'YearsInCurrentRole',
                                       'YearsWithCurrManager'])

        # SCALE only numeric features
        num_scaled = knn_scaler.transform(num_df)

        # Combine scaled numeric and encoded categorical (do NOT scale combined)
        final_features = np.hstack((num_scaled, cat_encoded))

        # Prediction
        pred = knn_model.predict(final_features)[0]
        proba = knn_model.predict_proba(final_features)[0][1]

        #st.write(f"Predicted class: {pred}")
        #st.write(f"Probability array: {knn_model.predict_proba(final_features)[0]}")


        if pred == "Yes":
            st.error(f"⚠️ Likely to Leave (Probability: {proba:.2f})")
        else:
            st.success(f"✅ Likely to Stay (Probability of Leaving: {1 - proba:.2f})")

# ---------------- PERFORMANCE PAGE ----------------
elif page == "Performance Rating Prediction":
    st.header("🏆 Performance Rating Predictor")

    # Inputs
    es = st.slider("Environment Satisfaction", 1, 4, 3)
    psh = st.number_input("Percent Salary Hike", min_value=0, max_value=100, value=15)
    rs = st.slider("Relationship Satisfaction", 1, 4, 3)
    ycr = st.number_input("Years In Current Role", min_value=0, max_value=20, value=5)
    dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    jr = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                   "Manager", "Human Resources", "Manufacturing Director", "Healthcare Representative"])

    if st.button("Predict Performance Rating"):
        # Categorical dataframe
        cat_df = pd.DataFrame([[dept, jr]], columns=['Department', 'JobRole'])
        cat_encoded = rf_encoder.transform(cat_df)

        # Numerical dataframe
        num_df = pd.DataFrame([[es, psh, rs, ycr]],
                              columns=['EnvironmentSatisfaction', 'PercentSalaryHike',
                                       'RelationshipSatisfaction', 'YearsInCurrentRole'])

        # SCALE only numeric features
        num_scaled = rf_scaler.transform(num_df)

        # Combine scaled numeric and encoded categorical (do NOT scale combined)
        final_features = np.hstack((num_scaled, cat_encoded))

        # Predict
        pred = rf_model.predict(final_features)[0]
        st.success(f"Predicted Performance Rating: {pred}")
