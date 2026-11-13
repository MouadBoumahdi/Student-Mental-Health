import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "mental_health_pipeline.joblib"
ENCODER_PATH = ROOT_DIR / "models" / "label_encoder.joblib"
METRICS_PATH = ROOT_DIR / "models" / "evaluation_metrics.json"

st.set_page_config(
    page_title="Student Mental Health Monitor",
    page_icon="🎓",
    layout="wide",
)


def load_model_artifacts() -> tuple[Any, Any]:
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)


def load_metrics() -> dict[str, Any]:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def get_category_options(model: Any) -> dict[str, list[str]]:
    preprocessor = model.named_steps["preprocessor"]
    options: dict[str, list[str]] = {}

    for _, transformer, columns in preprocessor.transformers_:
        if not hasattr(transformer, "named_steps"):
            continue
        encoder = transformer.named_steps.get("encoder")
        if encoder is None or not hasattr(encoder, "categories_"):
            continue
        for column, categories in zip(columns, encoder.categories_):
            options[str(column)] = [str(category) for category in categories]

    return options


required_files = [MODEL_PATH, ENCODER_PATH, METRICS_PATH]
missing_files = [path.name for path in required_files if not path.is_file()]
if missing_files:
    st.error(
        "Fichiers du modèle manquants : "
        + ", ".join(missing_files)
        + ". Exécutez d'abord le notebook de prétraitement."
    )
    st.stop()

model, label_encoder = load_model_artifacts()
metrics = load_metrics()

st.title("Student Mental Health Monitor")
st.caption(
    "Consultez les performances du modèle ou estimez le niveau de risque "
    "pour une nouvelle observation."
)

analysis_tab, prediction_tab = st.tabs(
    ["Analyse du modèle", "Tester une nouvelle donnée"]
)

with analysis_tab:
    st.subheader(f"Modèle évalué : {metrics['model_name']}")
    st.caption("Métriques calculées sur le jeu de test.")

    metric_definitions = [
        ("Exactitude", "accuracy"),
        ("Précision pondérée", "precision"),
        ("Rappel pondéré", "recall"),
        ("F1 pondéré", "f1_score"),
    ]
    metric_columns = st.columns(len(metric_definitions))
    for column, (label, key) in zip(metric_columns, metric_definitions):
        column.metric(label, f"{metrics[key]:.1%}")

    chart_column, matrix_column = st.columns(2)
    with chart_column:
        st.markdown("#### Comparaison des métriques")
        metric_chart = pd.DataFrame(
            {
                "Score": [metrics[key] for _, key in metric_definitions],
            },
            index=[label for label, _ in metric_definitions],
        )
        st.bar_chart(metric_chart, horizontal=True, x_label="Score")

    classes = [str(label) for label in metrics["classes"]]
    confusion = np.asarray(metrics["confusion_matrix"], dtype=int)
    if confusion.shape != (len(classes), len(classes)):
        st.error("La matrice de confusion ne correspond pas à la liste des classes.")
        st.stop()

    with matrix_column:
        st.markdown("#### Matrice de confusion")
        normalized = st.checkbox("Afficher les pourcentages par classe réelle")
        display_matrix = confusion
        fmt = "d"
        if normalized:
            display_matrix = display_matrix.astype(float)
            row_totals = display_matrix.sum(axis=1, keepdims=True)
            display_matrix = np.divide(
                display_matrix,
                row_totals,
                out=np.zeros_like(display_matrix),
                where=row_totals != 0,
            ) * 100
            fmt = ".1f"

        figure, axis = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            display_matrix,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=axis,
            cbar=normalized,
        )
        axis.set_xlabel("Classe prédite")
        axis.set_ylabel("Classe réelle")
        st.pyplot(figure)
        plt.close(figure)

    true_positives = np.diag(confusion).astype(float)
    predicted_totals = confusion.sum(axis=0)
    actual_totals = confusion.sum(axis=1)
    class_precision = np.divide(
        true_positives,
        predicted_totals,
        out=np.zeros_like(true_positives),
        where=predicted_totals != 0,
    )
    class_recall = np.divide(
        true_positives,
        actual_totals,
        out=np.zeros_like(true_positives),
        where=actual_totals != 0,
    )
    class_f1 = np.divide(
        2 * class_precision * class_recall,
        class_precision + class_recall,
        out=np.zeros_like(class_precision),
        where=(class_precision + class_recall) != 0,
    )
    class_metrics = pd.DataFrame(
        {
            "Précision": class_precision,
            "Rappel": class_recall,
            "F1": class_f1,
            "Observations": actual_totals,
        },
        index=classes,
    )

    st.markdown("#### Résultats par classe")
    class_column, recall_column = st.columns(2)
    with class_column:
        st.dataframe(class_metrics.round(3), width="stretch")
    with recall_column:
        st.bar_chart(class_metrics[["Rappel"]], horizontal=True, x_label="Score")

with prediction_tab:
    st.subheader("Tester une nouvelle observation")
    st.info(
        "Cette estimation est un outil pédagogique et ne constitue pas un "
        "diagnostic médical. Elle ne remplace pas l'avis d'un professionnel."
    )

    category_options = get_category_options(model)
    with st.form("student_prediction_form"):
        first_column, second_column, third_column = st.columns(3)

        with first_column:
            age = st.number_input("Âge", min_value=15, max_value=100, value=22)
            gender = st.selectbox("Genre", category_options["Gender"])
            year_of_study = st.number_input(
                "Année d'étude", min_value=1, max_value=5, value=2
            )
            sleep_hours = st.number_input(
                "Heures de sommeil par jour",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.1,
            )
            study_hours = st.number_input(
                "Heures d'étude par semaine",
                min_value=0.0,
                max_value=168.0,
                value=20.0,
                step=0.5,
            )

        with second_column:
            gpa = st.number_input(
                "GPA",
                min_value=0.0,
                max_value=4.0,
                value=3.0,
                step=0.01,
            )
            social_media_hours = st.number_input(
                "Réseaux sociaux (heures par jour)",
                min_value=0.0,
                max_value=24.0,
                value=3.0,
                step=0.1,
            )
            screen_time_hours = st.number_input(
                "Temps d'écran (heures par jour)",
                min_value=0.0,
                max_value=24.0,
                value=7.5,
                step=0.1,
            )
            physical_activity = st.selectbox(
                "Activité physique", category_options["Physical_Activity"]
            )
            financial_stress = st.number_input(
                "Stress financier (échelle 0–10)",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1,
            )

        with third_column:
            family_support = st.number_input(
                "Soutien familial (échelle 0–10)",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1,
            )
            attendance_rate = st.number_input(
                "Taux de présence (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=0.1,
            )
            part_time_job = st.selectbox(
                "Emploi à temps partiel", category_options["Part_Time_Job"]
            )
            diet_quality = st.selectbox(
                "Qualité de l'alimentation", category_options["Diet_Quality"]
            )
            previous_history = st.selectbox(
                "Antécédents de santé mentale",
                category_options["Previous_Mental_Health_History"],
            )

        submitted = st.form_submit_button("Estimer le niveau de risque")

    if submitted:
        values = {
            "Age": age,
            "Gender": gender,
            "Year_of_Study": year_of_study,
            "Sleep_Hours": sleep_hours,
            "Study_Hours_Per_Week": study_hours,
            "GPA": gpa,
            "Social_Media_Hours": social_media_hours,
            "Screen_Time_Hours": screen_time_hours,
            "Physical_Activity": physical_activity,
            "Financial_Stress": financial_stress,
            "Family_Support": family_support,
            "Attendance_Rate": attendance_rate,
            "Part_Time_Job": part_time_job,
            "Diet_Quality": diet_quality,
            "Previous_Mental_Health_History": previous_history,
        }
        expected_features = list(model.feature_names_in_)
        if set(values) != set(expected_features):
            st.error("Les champs du formulaire ne correspondent pas au modèle chargé.")
            st.stop()

        input_data = pd.DataFrame([values], columns=expected_features)
        predicted_index = int(model.predict(input_data)[0])
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        class_indices = np.asarray(model.classes_, dtype=int)
        probability_labels = label_encoder.inverse_transform(class_indices)
        probabilities = model.predict_proba(input_data)[0]
        probability_table = pd.DataFrame(
            {"Probabilité": probabilities},
            index=probability_labels,
        ).sort_values("Probabilité", ascending=False)

        st.success(f"Niveau estimé par le modèle : **{predicted_label}**")
        st.markdown("#### Probabilités estimées par classe")
        st.bar_chart(probability_table, horizontal=True, x_label="Probabilité")
        st.dataframe(
            probability_table.style.format({"Probabilité": "{:.1%}"}),
            width="stretch",
        )