import streamlit as st
import numpy as np
import pickle

# ==============================
# LOAD MODEL, SCALER, DAN AKURASI
# ==============================

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('akurasi_model.pkl', 'rb') as acc_file:
    train_accuracy = pickle.load(acc_file)

# ==============================
# UI APLIKASI
# ==============================

st.title("Aplikasi Prediksi Diabetes Menggunakan Model KNN")
st.markdown("---")
st.subheader("Formulir Data Pasien :")

# ==============================
# INPUT USER (2 KOLUM)
# ==============================

col1, col2 = st.columns(2)

with col1:
    gender_label = st.selectbox("Jenis Kelamin", options=["Laki-laki", "Perempuan"])
    gender = 1 if gender_label == "Laki-laki" else 0

    age = st.number_input("Usia (tahun)", min_value=0, step=1, format="%d")

    hypertension_label = st.radio("Apakah Mengidap Hipertensi?", options=["Ya", "Tidak"])
    hypertension = 1 if hypertension_label == "Ya" else 0

    heart_disease_label = st.radio("Apakah Mengidap Penyakit Jantung?", options=["Ya", "Tidak"])
    heart_disease = 1 if heart_disease_label == "Ya" else 0

with col2:
    # 3 kategori sederhana untuk riwayat merokok
    smoking_label = st.selectbox("Riwayat Merokok", options=[
        "Tidak Pernah Merokok", "Pernah Merokok", "Masih Merokok (Aktif)"
    ])
    smoking_dict = {
        "Tidak Pernah Merokok": 0,
        "Pernah Merokok": 1,
        "Masih Merokok (Aktif)": 2
    }
    smoking = smoking_dict[smoking_label]

    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, format="%.1f")
    hba1c = st.number_input("HbA1c Level", min_value=0.0, format="%.1f")
    blood_glucose = st.number_input("Kadar Glukosa Darah (mg/dL)", min_value=0.0, format="%.1f")

# ==============================
# TOMBOL PREDIKSI
# ==============================

if st.button("🔍 Prediksi Diabetes"):
    # Susun data input
    data = np.array([[gender, age, hypertension, heart_disease, smoking, bmi, hba1c, blood_glucose]])
    data_scaled = scaler.transform(data)

    # Prediksi & probabilitas
    prediction = model.predict(data_scaled)[0]
    proba = model.predict_proba(data_scaled)[0][1]
    percent = round(proba * 100, 2)
    accuracy_display = round(train_accuracy * 100, 2)

    # Hasil prediksi
    if prediction == 1:
        st.error(f"⚠️ Hasil Prediksi: **Pasien Terindikasi Diabetes** ({percent}% kemungkinan)")
    else:
        st.success(f"✅ Hasil Prediksi: **Pasien Tidak Terindikasi Diabetes** ({percent}% kemungkinan)")

    # Tampilkan akurasi model
    st.info(f"📊 Akurasi Model pada Data Latih: **{accuracy_display}%**")
