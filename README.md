# 🦁 Animal Species Classification (90 Classes)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📋 Project Overview
Proyek ini adalah implementasi **Deep Learning** untuk mengklasifikasikan gambar hewan ke dalam **90 kategori spesies** yang berbeda. Mengingat banyaknya jumlah kelas dan variasi gambar, proyek ini menggunakan teknik **Transfer Learning** dengan arsitektur **MobileNetV2** untuk mencapai akurasi yang tinggi dengan waktu komputasi yang efisien.

Hasil akhir dari proyek ini dideploy menjadi aplikasi web interaktif menggunakan **Streamlit**, memungkinkan pengguna untuk mengunggah gambar hewan dan mendapatkan prediksi spesies beserta tingkat keyakinannya (*confidence score*).

## 🗂️ Dataset
Dataset yang digunakan berasal dari Kaggle: **[Animal Image Dataset (90 Different Animals)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)**.
* **Jumlah Kelas:** 90 Spesies (Antelope, Badger, Bat, Bear, dll.)
* **Format:** Image (JPG/PNG)
* **Preprocessing:** Rescaling (1./255), Resizing (224x224).

> *Catatan: Folder dataset tidak disertakan dalam repositori ini karena ukurannya yang besar. Silakan unduh dari link Kaggle di atas.*

## 🛠️ Tech Stack & Tools
* **Bahasa Pemrograman:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit
* **IDE:** PyCharm / Jupyter Notebook

## ⚙️ Methodology
1.  **Data Preparation:** Menggunakan `ImageDataGenerator` untuk augmentasi data ringan dan validasi split (80% Train, 20% Validation).
2.  **Model Architecture:**
    * **Base Model:** MobileNetV2 (Pre-trained on ImageNet) sebagai *feature extractor*. Layer ini dibekukan (*frozen*).
    * **Custom Head:** Menambahkan layer `GlobalAveragePooling2D`, `Dropout` (untuk mencegah overfitting), dan `Dense` (90 units, Softmax) sebagai classifier.
3.  **Training:** Menggunakan Optimizer `Adam` dan loss function `categorical_crossentropy`.
4.  **Evaluation:** Memantau grafik akurasi/loss dan menggunakan confusion matrix.

## 📊 Results
* **Training Accuracy:** ~96% 
* **Validation Accuracy:** ~86% 
* **Model Performance:** Model mampu membedakan hewan dengan ciri visual yang mirip berkat fitur yang dipelajari dari ImageNet.

## 🚀 How to Run Locally

Ikuti langkah-langkah berikut untuk menjalankan proyek ini di komputer lokal Anda:

### 1. Clone Repository
```bash
git clone [https://github.com/arizal6174/Animal-Classification-90.git](https://github.com/arizal6174/Animal-Classification-90.git)
cd Animal-Classification-90