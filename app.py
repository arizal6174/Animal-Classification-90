import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import json
import time

# ---------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN (Wajib di baris pertama)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Animal AI Classifier",
    page_icon="🦁",
    layout="wide",  # Layout lebar agar terlihat pro
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------------------
# 2. LOAD MODEL & DATA (Cached agar cepat)
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    # Pastikan path model sesuai
    model = tf.keras.models.load_model('animal_model_90classes.keras')
    return model


@st.cache_data
def load_labels():
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}


# Load resource sekali saja
with st.spinner('Sedang memuat model AI...'):
    model = load_model()
    labels = load_labels()

# ---------------------------------------------------------------------
# 3. SIDEBAR (Informasi Project)
# ---------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=100)
    st.title("🦁 Animal Classification AI")
    st.markdown("---")
    st.markdown("""
    **Tentang Aplikasi:**
    Aplikasi ini menggunakan **Deep Learning (MobileNetV2)** untuk mengenali 90 jenis hewan dari gambar yang diupload.

    **Cara Pakai:**
    1. Upload gambar hewan.
    2. Tunggu AI menganalisa.
    3. Lihat hasil prediksi dan grafiknya.
    """)
    st.markdown("---")
    st.caption("Developed by Arizal")
    st.caption("Powered by TensorFlow & Streamlit")

# ---------------------------------------------------------------------
# 4. AREA UTAMA
# ---------------------------------------------------------------------
st.title("🔍 Deteksi Spesies Hewan")
st.markdown("Upload gambar hewan liar atau peliharaan, dan biarkan AI menebaknya!")

# File Uploader dengan styling
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Layout 2 Kolom: Kiri (Gambar), Kanan (Hasil)
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.info("Gambar yang diupload:")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True, caption="Input Image")

    with col2:
        st.info("Hasil Analisis AI:")

        # Animasi Loading
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

        # Preprocessing
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        predictions = model.predict(img_array)[0]

        # Ambil Top 1
        predicted_class_index = np.argmax(predictions)
        predicted_label = labels[predicted_class_index]
        confidence = predictions[predicted_class_index] * 100

        # Tampilkan Hasil Utama dengan Metrik Besar
        st.metric(label="Prediksi Utama", value=predicted_label, delta=f"{confidence:.2f}% Akurat")

        # Ambil Top 5 Prediksi untuk Chart
        # Mengurutkan index dari probabilitas tertinggi ke terendah
        top_5_indices = np.argsort(predictions)[-5:][::-1]
        top_5_values = predictions[top_5_indices] * 100
        top_5_labels = [labels[i] for i in top_5_indices]

        # Membuat DataFrame untuk Chart
        df_results = pd.DataFrame({
            'Hewan': top_5_labels,
            'Confidence (%)': top_5_values
        })

        # Menampilkan Bar Chart
        st.write("##### Top 5 Kemungkinan:")
        st.bar_chart(df_results.set_index('Hewan'))

        # Expander untuk detail teknis (opsional)
        with st.expander("Lihat Detail Probabilitas"):
            st.dataframe(df_results)

else:
    # Tampilan awal jika belum ada upload
    st.warning("👈 Silakan upload gambar melalui panel di atas untuk memulai.")