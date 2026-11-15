import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Mental Health", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
model = joblib.load(ROOT / "models" / "mental_health_pipeline.joblib")
encoder = joblib.load(ROOT / "models" / "label_encoder.joblib")
metrics = json.loads((ROOT / "models" / "evaluation_metrics.json").read_text())

st.title("Student Mental Health Monitor")
tab1, tab2 = st.tabs(["Model Analysis", "New Prediction"])

with tab1:
    st.subheader(f"Model: {metrics['model_name']}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Precision", f"{metrics['precision']:.1%}")
    c3.metric("Recall", f"{metrics['recall']:.1%}")
    c4.metric("F1 Score", f"{metrics['f1_score']:.1%}")

    st.write("### Confusion Matrix")
    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=metrics["classes"],
        columns=metrics["classes"]
    )
    st.dataframe(cm_df, use_container_width=True)

with tab2:
    prep = model.named_steps["preprocessor"]
    cats = {}
    for _, trans, cols in prep.transformers_:
        if hasattr(trans, "named_steps") and "encoder" in trans.named_steps:
            for col, values in zip(cols, trans.named_steps["encoder"].categories_):
                cats[col] = list(values)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 15, 100, 22)
            gender = st.selectbox("Gender", cats["Gender"])
            year = st.number_input("Year of Study", 1, 5, 2)
            sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
            study = st.number_input("Study Hours / Week", 0.0, 168.0, 20.0)
            gpa = st.number_input("GPA", 0.0, 4.0, 3.0)
            social = st.number_input("Social Media Hours", 0.0, 24.0, 3.0)
            screen = st.number_input("Screen Time Hours", 0.0, 24.0, 7.5)

        with col2:
            activity = st.selectbox("Physical Activity", cats["Physical_Activity"])
            stress = st.slider("Financial Stress", 0.0, 10.0, 5.0)
            support = st.slider("Family Support", 0.0, 10.0, 5.0)
            attend = st.slider("Attendance Rate (%)", 0.0, 100.0, 80.0)
            job = st.selectbox("Part Time Job", cats["Part_Time_Job"])
            diet = st.selectbox("Diet Quality", cats["Diet_Quality"])
            history = st.selectbox("Previous Mental Health History", cats["Previous_Mental_Health_History"])

        if st.form_submit_button("Predict"):
            data = pd.DataFrame([{
                "Age": age, "Gender": gender, "Year_of_Study": year,
                "Sleep_Hours": sleep, "Study_Hours_Per_Week": study, "GPA": gpa,
                "Social_Media_Hours": social, "Screen_Time_Hours": screen,
                "Physical_Activity": activity, "Financial_Stress": stress,
                "Family_Support": support, "Attendance_Rate": attend,
                "Part_Time_Job": job, "Diet_Quality": diet,
                "Previous_Mental_Health_History": history
            }])

            pred = encoder.inverse_transform(model.predict(data))[0]
            probs = model.predict_proba(data)[0]
            
            st.success(f"Predicted Risk Level: **{pred}**")
            st.bar_chart(pd.Series(probs, index=encoder.classes_))