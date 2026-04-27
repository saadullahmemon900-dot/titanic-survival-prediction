import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Titanic Dashboard", page_icon="🚢", layout="centered")

# ---------------------------
# LOAD MODEL
# ---------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------------------
# TITLE UI
# ---------------------------
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🚢 Titanic Survival Dashboard</h1>", unsafe_allow_html=True)
st.write("Predict survival probability using Machine Learning + beautiful UI 🎯")

# ---------------------------
# SIDEBAR INPUTS
# ---------------------------
st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox("Pclass", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 1, 80, 25)
sibsp = st.sidebar.slider("SibSp", 0, 8, 0)
parch = st.sidebar.slider("Parch", 0, 8, 0)
fare = st.sidebar.slider("Fare", 0, 500, 50)
embarked = st.sidebar.selectbox("Embarked", ["S", "C", "Q"])

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
sex = 0 if sex == "male" else 1

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
fare_log = np.log1p(fare)

embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_data = np.array([[pclass, sex, age, family_size,
                        is_alone, fare_log, embarked_Q, embarked_S]])

input_data_scaled = scaler.transform(input_data)

# ---------------------------
# PREDICTION BUTTON
# ---------------------------
if st.button("🎯 Predict Survival"):

    prediction = model.predict(input_data_scaled)[0]
    prob = model.predict_proba(input_data_scaled)[0][1]

    # ---------------------------
    # RESULT DISPLAY
    # ---------------------------
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("🎉 Passenger SURVIVED")
    else:
        st.error("💀 Passenger DID NOT SURVIVE")

    # ---------------------------
    # PROBABILITY GAUGE
    # ---------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Survival Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "lightgreen"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # INPUT SUMMARY CHART
    # ---------------------------
    st.subheader("📌 Input Summary")

    features = ["Pclass", "Sex", "Age", "FamilySize", "IsAlone", "FareLog"]
    values = [pclass, sex, age, family_size, is_alone, fare_log]

    st.bar_chart(dict(zip(features, values)))

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("🚢 Built with ❤️ using Streamlit | Titanic ML Project")