import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Judul dan Deskripsi ---
st.title("Aplikasi Prediksi Penyakit Jantung 🫀")
st.write("""
Aplikasi ini menggunakan model **Logistic Regression** untuk memprediksi 
kemungkinan seseorang menderita penyakit jantung berdasarkan 13 fitur klinis.
""")

# --- Nama File Model dan Scaler ---
MODEL_FILE = 'heart_disease_model.pkl'
SCALER_FILE = 'scaler.pkl'

# --- Memuat Model dan Scaler ---
# @st.cache_resource digunakan agar Streamlit tidak me-load ulang file-file ini
# setiap kali ada interaksi dari user, sehingga aplikasi lebih cepat.
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    """Memuat model dan scaler yang sudah disimpan."""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        return None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file: {e}")
        return None, None

# Cek apakah file ada sebelum memuat
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    st.error(f"Error: File '{MODEL_FILE}' atau '{SCALER_FILE}' tidak ditemukan.")
    st.error("Pastikan kedua file tersebut berada di folder yang sama dengan `app.py` di repositori GitHub Anda.")
    st.stop() # Hentikan eksekusi aplikasi jika file tidak ada

model, scaler = load_model_and_scaler(MODEL_FILE, SCALER_FILE)

if model is None or scaler is None:
    st.stop() # Hentikan jika gagal load

# --- Input dari User (di Sidebar) ---
st.sidebar.header('Input Fitur Pasien:')

# Fungsi ini akan mengumpulkan input dari sidebar
def user_input_features():
    
    # --- Penjelasan Fitur ---
    help_text = {
        'cp': "Tipe Nyeri Dada (0: Angina Tipikal, 1: Angina Atipikal, 2: Nyeri Non-Angina, 3: Asimtomatik)",
        'fbs': "Gula Darah Puasa > 120 mg/dl (0: Tidak, 1: Ya)",
        'restecg': "Hasil EKG Istirahat (0, 1, 2)",
        'exang': "Angina Akibat Olahraga (0: Tidak, 1: Ya)",
        'slope': "Slope Puncak ST Olahraga (0, 1, 2)",
        'ca': "Jumlah Pembuluh Darah Utama yang Terlihat (0-4)",
        'thal': "Thalassemia (0: Null, 1: Normal, 2: Cacat Tetap, 3: Cacat Dapat Dibalik)"
    }
    
    # --- Input Slider dan Angka ---
    st.sidebar.markdown("### Fitur Kontinu")
    age = st.sidebar.slider('Usia (Tahun)', 29, 77, 50)
    trestbps = st.sidebar.slider('Tekanan Darah (trestbps) (mm Hg)', 94, 200, 120)
    chol = st.sidebar.slider('Kolesterol (chol) (mg/dl)', 126, 564, 200)
    thalach = st.sidebar.slider('Detak Jantung Maks (thalach)', 71, 202, 150)
    oldpeak = st.sidebar.slider('ST Depression (oldpeak)', 0.0, 6.2, 1.0, step=0.1)

    # --- Input Selectbox (Kategorikal) ---
    st.sidebar.markdown("### Fitur Kategorikal")
    sex = st.sidebar.selectbox('Jenis Kelamin (sex)', (0, 1), format_func=lambda x: 'Wanita' if x == 0 else 'Pria')
    cp = st.sidebar.selectbox('Tipe Nyeri Dada (cp)', (0, 1, 2, 3), help=help_text['cp'])
    fbs = st.sidebar.selectbox('Gula Darah Puasa > 120 mg/dl (fbs)', (0, 1), format_func=lambda x: 'Tidak' if x == 0 else 'Ya', help=help_text['fbs'])
    restecg = st.sidebar.selectbox('Hasil EKG Istirahat (restecg)', (0, 1, 2), help=help_text['restecg'])
    exang = st.sidebar.selectbox('Angina Akibat Olahraga (exang)', (0, 1), format_func=lambda x: 'Tidak' if x == 0 else 'Ya', help=help_text['exang'])
    slope = st.sidebar.selectbox('Slope Puncak ST Olahraga (slope)', (0, 1, 2), help=help_text['slope'])
    ca = st.sidebar.selectbox('Jumlah Pembuluh Darah (ca)', (0, 1, 2, 3, 4), help=help_text['ca'])
    thal = st.sidebar.selectbox('Thalassemia (thal)', (0, 1, 2, 3), help=help_text['thal'])

    # Kumpulkan data dalam dictionary
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    # Ubah ke DataFrame dengan urutan kolom yang BENAR (sesuai X_train di Colab)
    # Ini sangat penting agar scaler dan model bekerja dengan benar
    feature_cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    features = pd.DataFrame(data, index=[0])
    
    return features[feature_cols]

# Ambil input dari sidebar
input_df = user_input_features()

# --- Tampilkan Input User di Halaman Utama ---
st.subheader('Input Fitur yang Diberikan:')
st.dataframe(input_df)

# --- Tombol Prediksi ---
if st.sidebar.button('Lakukan Prediksi', type="primary"):
    
    # 1. Scaling input user menggunakan scaler yang sudah di-load
    # input_df diubah ke numpy array dulu (sesuai cara scaler dilatih)
    input_scaled = scaler.transform(input_df.to_numpy())
    
    # 2. Lakukan prediksi
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # --- Tampilkan Hasil Prediksi ---
    st.subheader('Hasil Prediksi:')
    
    # Bagi layout jadi 2 kolom
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Diagnosis Model")
        if prediction[0] == 0:
            st.success('**Risiko Rendah** (Tidak Terdeteksi Penyakit Jantung)')
        else:
            st.error('**Risiko Tinggi** (Terdeteksi Penyakit Jantung)')

    with col2:
        st.markdown("#### Keyakinan Prediksi (Probabilitas)")
        # Buat dataframe untuk probabilitas
        prob_df = pd.DataFrame({
            'Kelas': ['Risiko Rendah (0)', 'Risiko Tinggi (1)'],
            'Probabilitas': prediction_proba[0]
        })
        # Format probabilitas jadi persen
        prob_df['Probabilitas'] = prob_df['Probabilitas'].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(prob_df, hide_index=True)

else:
    st.info('Silakan isi fitur di sidebar dan klik **"Lakukan Prediksi"** untuk melihat hasilnya.')