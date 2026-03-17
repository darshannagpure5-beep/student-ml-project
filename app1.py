import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Student Predictor",
                   page_icon="🎓", layout="centered")

# Load model
model = joblib.load("models/model.pkl")

# Title
st.markdown("<h1 style='text-align: center;'>🎓 Student Pass Predictor</h1>",
            unsafe_allow_html=True)
st.write("### Enter student details below:")

# Sidebar (for better UX)
st.sidebar.header("📊 Input Features")

hours = st.sidebar.slider("Hours Studied", 0, 12, 4)
sleep = st.sidebar.slider("Sleep Hours", 0, 12, 6)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 60)

# Show inputs nicely
st.write("### 🧾 Your Inputs")
st.write(f"📘 Study Hours: **{hours} hrs**")
st.write(f"😴 Sleep: **{sleep} hrs**")
st.write(f"🏫 Attendance: **{attendance}%**")

# Prediction button
if st.button("🚀 Predict Result"):

    input_data = np.array([[hours, sleep, attendance]])
    prediction = model.predict(input_data)[0]

    st.write("---")

    if prediction == 1:
        st.success("✅ Student will PASS 🎉")
        st.balloons()
    else:
        st.error("❌ Student may FAIL 😢")

# Footer
st.write("---")
st.caption("Built by Darshan 🚀")
