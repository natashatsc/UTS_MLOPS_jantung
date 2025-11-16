import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Aplikasi Prediksi Risiko Terkena Penyakit Jantung")
st.write("""
Aplikasi ini menggunakan model **Logistic Regression** untuk memprediksi 
kemungkinan seseorang menderita penyakit jantung berdasarkan 13 fitur klinis.
""")

MODEL_FILE = 'heart_disease_model.pkl'
SCALER_FILE = 'scaler.pkl'

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

if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    st.error(f"Error: File '{MODEL_FILE}' atau '{SCALER_FILE}' tidak ditemukan.")
    st.error("Pastikan kedua file tersebut ada di folder yang sama dengan `app.py` di repositori GitHub")
    st.stop() 

model, scaler = load_model_and_scaler(MODEL_FILE, SCALER_FILE)

if model is None or scaler is None:
    st.stop() 

st.sidebar.header('Input Fitur Pasien:')

def user_input_features():
    
    help_text = {
        'cp': "Tipe Nyeri Dada (0: Angina Tipikal, 1: Angina Atipikal, 2: Nyeri Non-Angina, 3: Asimtomatik)",
        'fbs': "Gula Darah Puasa > 120 mg/dl (0: Tidak, 1: Ya)",
        'restecg': "Hasil EKG Istirahat (0, 1, 2)",
        'exang': "Angina Akibat Olahraga (0: Tidak, 1: Ya)",
        'slope': "Slope Puncak ST Olahraga (0, 1, 2)",
        'ca': "Jumlah Pembuluh Darah Utama yang Terlihat (0-4)",
        'thal': "Thalassemia (0: Null, 1: Normal, 2: Cacat Tetap, 3: Cacat Dapat Dibalik)"
    }
    
    st.sidebar.markdown("### Fitur Kontinu")
    age = st.sidebar.slider('Usia (Tahun)', 29, 77, 50)
    trestbps = st.sidebar.slider('Tekanan Darah (trestbps) (mm Hg)', 94, 200, 120)
    chol = st.sidebar.slider('Kolesterol (chol) (mg/dl)', 126, 564, 200)
    thalach = st.sidebar.slider('Detak Jantung Maks (thalach)', 71, 202, 150)
    oldpeak = st.sidebar.slider('ST Depression (oldpeak)', 0.0, 6.2, 1.0, step=0.1)

    st.sidebar.markdown("### Fitur Kategorikal")
    sex = st.sidebar.selectbox('Jenis Kelamin (sex)', (0, 1), format_func=lambda x: 'Wanita' if x == 0 else 'Pria')
    cp = st.sidebar.selectbox('Tipe Nyeri Dada (cp)', (0, 1, 2, 3), help=help_text['cp'])
    fbs = st.sidebar.selectbox('Gula Darah Puasa > 120 mg/dl (fbs)', (0, 1), format_func=lambda x: 'Tidak' if x == 0 else 'Ya', help=help_text['fbs'])
    restecg = st.sidebar.selectbox('Hasil EKG Istirahat (restecg)', (0, 1, 2), help=help_text['restecg'])
    exang = st.sidebar.selectbox('Angina Akibat Olahraga (exang)', (0, 1), format_func=lambda x: 'Tidak' if x == 0 else 'Ya', help=help_text['exang'])
    slope = st.sidebar.selectbox('Slope Puncak ST Olahraga (slope)', (0, 1, 2), help=help_text['slope'])
    ca = st.sidebar.selectbox('Jumlah Pembuluh Darah (ca)', (0, 1, 2, 3, 4), help=help_text['ca'])
    thal = st.sidebar.selectbox('Thalassemia (thal)', (0, 1, 2, 3), help=help_text['thal'])

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
    
    feature_cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    features = pd.DataFrame(data, index=[0])
    
    return features[feature_cols]

input_df = user_input_features()

st.subheader('Input Fitur yang Diberikan:')
st.dataframe(input_df)

if st.sidebar.button('Lakukan Prediksi', type="primary"):
    
    input_scaled = scaler.transform(input_df.to_numpy())
    
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader('Hasil Prediksi:')
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Diagnosis Model")
        if prediction[0] == 0:
            st.success('**Risiko Rendah** (Tidak Terdeteksi Penyakit Jantung)')
        else:
            st.error('**Risiko Tinggi** (Terdeteksi Penyakit Jantung)')

    with col2:
        st.markdown("#### Keyakinan Prediksi (Probabilitas)")
        prob_df = pd.DataFrame({
            'Kelas': ['Risiko Rendah (0)', 'Risiko Tinggi (1)'],
            'Probabilitas': prediction_proba[0]
        })
        prob_df['Probabilitas'] = prob_df['Probabilitas'].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(prob_df, hide_index=True)

else:
    st.info('Silakan isi fitur di sidebar dan klik **"Lakukan Prediksi"** untuk melihat hasilnya.')
