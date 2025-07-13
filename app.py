import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from utils.preprocessing import encode_user_input
from utils.shap_helper import get_explainer, get_shap_values, plot_shap_waterfall


st.set_page_config(page_title="Depresyon Tahmini", layout="wide")

st.title("🧠 Depresyon Tahmini ve Açıklaması")
st.markdown("""
Bu uygulama, verdiğiniz bilgilere göre depresyon riskinizi tahmin eder  
ve tahminin hangi faktörlerden etkilendiğini **SHAP** yöntemiyle açıklar.
""")


model_names = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl",
    "SVM": "models/svm.pkl"
}

model_choice = st.sidebar.selectbox("📌 Kullanılacak Modeli Seçin", list(model_names.keys()))

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(model_names[model_choice])



st.subheader("📋 Bilgilerinizi Girin")

with st.form("user_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
        age = st.slider("Yaş", 18, 60, 25)
        cgpa = st.slider("Not Ortalaması (CGPA)", 0.0, 10.0, 7.5)
        academic_pressure = st.slider("Akademik Baskı (0-5)", 0, 5, 2)
        sleep_duration = st.selectbox("Uyku Süresi", ["<5 saat", "5-6 saat", "7-8 saat", ">8 saat"])
        diet = st.selectbox("Beslenme Alışkanlığı", ["Sağlıksız", "Orta", "Sağlıklı"])
        family_history = st.selectbox("Ailede ruhsal hastalık geçmişi var mı?", ["Evet", "Hayır"])


    with col2:
        job_satisfaction = st.slider("İş/Okul Memnuniyeti (0-5)", 0, 5, 3)
        work_pressure = st.slider("İş Baskısı (0-5)", 0, 5, 2)
        study_satisfaction = st.slider("Ders Memnuniyeti (0-5)", 0, 5, 3)
        work_hours = st.slider("Günlük Çalışma/Saat", 0, 12, 6)
        financial_stress = st.slider("Finansal Stres (0-5)", 0, 5, 2)
        suicidal = st.selectbox("Daha önce intihar düşünceniz oldu mu?", ["Evet", "Hayır"])

    submitted = st.form_submit_button("🎯 Tahmin Et")



if submitted:
    # Kullanıcı verilerini DataFrame'e dönüştür
    input_dict = {
        "Gender": 1 if gender == "Erkek" else 0,
        "Age": age,
        "Academic Pressure": academic_pressure,
        "Work Pressure": work_pressure,
        "CGPA": cgpa,
        "Study Satisfaction": study_satisfaction,
        "Job Satisfaction": job_satisfaction,
        "Sleep Duration": {"<5 saat": 0, "5-6 saat": 1, "7-8 saat": 2, ">8 saat": 3}[sleep_duration],
        "Dietary Habits": {"Sağlıksız": 0, "Orta": 1, "Sağlıklı": 2}[diet],
        "Work/Study Hours": work_hours,
        "Financial Stress": financial_stress,
        "Family History of Mental Illness": 1 if family_history == "Evet" else 0,
        "Suicidal Thoughts": 1 if suicidal == "Evet" else 0
    }

    user_input_df = pd.DataFrame([input_dict])
    processed_input = encode_user_input(user_input_df)

    # Tahmin
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]

    # Sonuç
    st.subheader("🎯 Tahmin Sonucu")
    if prediction == 1:
        st.error(f"⚠️ Depresyon riski var (%{probability*100:.1f})")
    else:
        st.success(f"✅ Depresyon riski yok (%{(1-probability)*100:.1f})")

    # SHAP
    st.subheader("🧠 Tahmin Açıklaması (SHAP)")
    explainer = get_explainer(model)
    shap_values = get_shap_values(explainer, processed_input)
    fig = plot_shap_waterfall(shap_values)
    st.pyplot(fig)
