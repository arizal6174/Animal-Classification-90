import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Konfigurasi Halaman
st.set_page_config(page_title="Animal Classifier AI", page_icon="🦁")


# Load Model & Labels (Gunakan @st.cache_resource agar tidak reload terus menerus)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('animal_model_90classes.keras')
    return model


@st.cache_data
def load_labels():
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    # Membalik dictionary agar kuncinya adalah angka index, valuenya nama hewan
    return {v: k for k, v in class_indices.items()}


model = load_model()
labels = load_labels()

# UI Streamlit
st.title("🦁 Animal Classification AI")
st.write("Upload gambar hewan, dan AI akan menebak spesiesnya (dari 90 kategori).")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)

    # Preprocessing
    st.write("Menganalisa...")
    img = image.resize((224, 224))  # Resize sesuai input model
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Menambah batch dimension

    # Prediksi
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    predicted_label = labels[predicted_class]

    # Hasil
    st.success(f"Prediksi: **{predicted_label}**")
    st.info(f"Tingkat Keyakinan: {confidence:.2f}%")