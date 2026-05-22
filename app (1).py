import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── CONFIGURACIÓN DE PÁGINA ─────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Rendimiento Académico",
    page_icon="🎓",
    layout="wide"
)

# ── CARGA DE ARTEFACTOS ─────────────────────────────────────────
@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load("modelo_xgboost.pkl")
    artefactos = joblib.load("pipeline_preproc.pkl")
    return modelo, artefactos

modelo, art = cargar_artefactos()

# ── TÍTULO Y DESCRIPCIÓN ────────────────────────────────────────
st.title("🎓 Sistema de Alerta Temprana — Rendimiento Académico")
st.markdown('''Esta herramienta predice si un estudiante **aprobará o reprobará** a partir
de sus hábitos, antecedentes y desempeño parcial.

El modelo utilizado es **XGBoost**, entrenado siguiendo la metodología
**CRISP-DM** sobre el dataset *Student Exam Performance* (10,000 registros).
'''
)
st.divider()

# ── ENTRADAS DEL USUARIO ────────────────────────────────────────
st.sidebar.header("📋 Datos del estudiante")

with st.sidebar:
    st.subheader("Demográficos")
    gender = st.selectbox("Género", ["Female", "Male"])
    age = st.slider("Edad", 5, 25, 17)
    parental_education = st.selectbox(
        "Educación de los padres",
        ["High School", "Bachelor", "Master", "PhD"]
    )
    family_income = st.selectbox("Ingreso familiar", ["Low", "Medium", "High"])
    internet_access = st.selectbox("Acceso a internet", ["Yes", "No"])
    study_environment = st.selectbox(
        "Ambiente de estudio", ["Quiet", "Moderate", "Noisy"]
    )

    st.subheader("Hábitos")
    study_hours_per_day = st.slider("Horas de estudio/día", 0.0, 24.0, 3.0, 0.5)
    attendance_rate = st.slider("Tasa de asistencia (%)", 0, 100, 80)
    sleep_hours = st.slider("Horas de sueño", 0.0, 24.0, 7.0, 0.5)
    social_media_hours = st.slider("Horas en redes sociales", 0.0, 24.0, 3.0, 0.5)
    assignment_completion_rate = st.slider("Tareas completadas (%)", 0, 100, 75)
    participation_score = st.slider("Puntaje de participación", 0, 100, 70)
    online_courses_completed = st.slider("Cursos en línea completados", 0, 9, 2)
    tutoring = st.selectbox("Asiste a tutorías", ["Yes", "No"])

    st.subheader("Desempeño académico")
    math_score = st.slider("Matemáticas", 0, 100, 70)
    reading_score = st.slider("Lectura", 0, 100, 70)
    writing_score = st.slider("Escritura", 0, 100, 70)
    science_score = st.slider("Ciencias", 0, 100, 70)
    final_exam_score = st.slider("Examen final", 0, 100, 65)
    previous_gpa = st.slider("GPA previo", 0.0, 5.0, 3.0, 0.1)

# ── DATAFRAME DE ENTRADA ────────────────────────────────────────
entrada = pd.DataFrame([
    {   
    "gender": gender, "age": age,
    "parental_education": parental_education,
    "family_income": family_income,
    "internet_access": internet_access,
    "study_environment": study_environment,
    "study_hours_per_day": study_hours_per_day,
    "attendance_rate": attendance_rate,
    "sleep_hours": sleep_hours,
    "social_media_hours": social_media_hours,
    "assignment_completion_rate": assignment_completion_rate,
    "participation_score": participation_score,
    "online_courses_completed": online_courses_completed,
    "tutoring": tutoring,
    "math_score": math_score,
    "reading_score": reading_score,
    "writing_score": writing_score,
    "science_score": science_score,
    "final_exam_score": final_exam_score,
    "previous_gpa": previous_gpa,
}])

# ── PIPELINE DE TRANSFORMACIÓN ──────────────────────────────────
def preparar(df_in, art):
    df = df_in.copy()
    # binarias
    for col in art["binary_cols"]:
        le = art["binary_encoders"][col]
        df[col + "_enc"] = le.transform(df[col])
    # one-hot
    ohe = art["ohe"]
    ohe_arr = ohe.transform(df[art["ohe_cols"]])
    ohe_df = pd.DataFrame(
        ohe_arr,
        columns=ohe.get_feature_names_out(art["ohe_cols"]),
        index=df.index
    )
    df = pd.concat(
        [df.drop(columns=art["binary_cols"] + art["ohe_cols"]), ohe_df], axis=1
    )
    df = df.reindex(columns=art["feature_columns"], fill_value=0)
    df[art["num_cols_scale"]] = art["scaler"].transform(df[art["num_cols_scale"]])
    return df

# ── INTERFAZ PRINCIPAL ──────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Datos ingresados")
    st.dataframe(entrada.T.rename(columns={0: "Valor"}), use_container_width=True)

with col2:
    st.subheader("🔮 Predicción")
    if st.button("Analizar estudiante", type="primary", use_container_width=True):
        try:
            X_proc = preparar(entrada, art)
            pred = modelo.predict(X_proc)[0]
            prob = modelo.predict_proba(X_proc)[0]
            resultado = art["le_target"].inverse_transform([pred])[0]
            prob_pass = prob[1] * 100
            prob_fail = prob[0] * 100

            if resultado == "Pass":
                st.success(f"### ✅ APROBARÁ\n\nProbabilidad de aprobar: **{prob_pass:.1f}%**")
            else:
                st.error(f"### ⚠️ EN RIESGO DE REPROBAR\n\nProbabilidad de reprobar: **{prob_fail:.1f}%**")
                st.warning(
                    "**Recomendación:** Iniciar plan de acompañamiento académico "
                    "(tutorías, refuerzo, seguimiento personalizado)."
                )

            st.markdown("---")
            st.metric("Probabilidad de aprobar", f"{prob_pass:.1f}%"
            )
            st.metric("Probabilidad de reprobar", f"{prob_fail:.1f}%"
            )
            st.progress(float(prob_pass / 100)) # Cast to float
        except Exception as e:
            st.error(f"Error en la predicción: {e}")

# ── PIE DE PÁGINA ───────────────────────────────────────────────
st.divider()
st.caption(
    "Trabajo Final — Metodología CRISP-DM | "
    "Sara Gabriela Barrios Llanes · Gestión Financiera · UPB · 2026"
)
