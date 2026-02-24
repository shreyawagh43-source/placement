import streamlit as st
import numpy as np
import joblib

model = joblib.load('resume_screening_model.pkl')

coding_map = {
    'Basic': 0,
    'Intermediate': 1,
    'Advanced': 2
}

st.title("AI Placement Prediction System")

internships = st.number_input("Number of Internships", 0, 5, 1)
skills = st.number_input("Skills Count", 0, 20, 5)
projects = st.number_input("Projects", 0, 10, 3)
coding = st.selectbox("Coding Level", ['Basic', 'Intermediate', 'Advanced'])
cgpa = st.slider("CGPA", 0.0, 10.0, 7.5)

if st.button("Predict Placement"):
    candidate = np.array([[
        internships,
        skills,
        projects,
        coding_map[coding],
        cgpa
    ]])

    pred = model.predict(candidate)[0]
    prob = model.predict_proba(candidate)[0][1]

    if pred == 1:
        st.success(f"✅ Likely Placed (Confidence: {prob*100:.2f}%)")
    else:
        st.error(f"❌ Not Placed (Confidence: {prob*100:.2f}%)")
