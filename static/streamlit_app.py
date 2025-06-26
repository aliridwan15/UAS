import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import os

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Prediksi Grade Glioma",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Mapping manual untuk Gender ---
# Sesuaikan ini jika 'Female' di dataset Anda direpresentasikan oleh 1 dan 'Male' oleh 0, dll.
# Berdasarkan dataset Anda (0=Female, 1=Male)
GENDER_MAP_INPUT_TO_NUMERIC = {"Female": 0, "Male": 1}

# Fungsi untuk memuat data, pra-pemrosesan, dan melatih model
@st.cache_resource
def initialize_and_train_models(dataset_path="glioma_dataset.csv"):
    st.info("Memuat dataset, pra-pemrosesan, dan melatih model... (Ini hanya terjadi sekali saat startup)")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        st.error(f"Error: File dataset '{dataset_path}' tidak ditemukan. Pastikan file ada di direktori yang sama dengan `streamlit_app.py`.")
        st.stop()

    # --- Pra-pemrosesan Data ---

    # 1. Penanganan kolom 'Gender'
    # Kolom 'Gender' di dataset Anda sudah numerik (0 atau 1).
    # Pastikan tipe datanya integer.
    df['Gender'] = df['Gender'].astype(int) 
    
    # 2. Label Encoding untuk kolom 'Race' (karena berisi string kategorikal)
    le_race = LabelEncoder()
    df['Race_encoded'] = le_race.fit_transform(df['Race'])
    # Buat mapping untuk Race agar bisa digunakan di bagian prediksi
    # Ini akan memetakan string Race ke angka yang digunakan oleh le_race
    RACE_MAP_INPUT_TO_NUMERIC = {label: index for index, label in enumerate(le_race.classes_)}


    # 3. Normalisasi (MinMaxScaler) untuk Age_at_diagnosis
    scaler = MinMaxScaler()
    df['Age_at_diagnosis'] = pd.to_numeric(df['Age_at_diagnosis'], errors='coerce')
    df['Age_at_diagnosis'].fillna(df['Age_at_diagnosis'].mean(), inplace=True) # Imputasi NaN jika ada
    df['Age_at_diagnosis_scaled'] = scaler.fit_transform(df[['Age_at_diagnosis']])

    # 4. PCA pada fitur genetik
    gene_columns = [
        'IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA', 'NF1',
        'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 'GRIN2A',
        'IDH2', 'FAT4', 'PDGFRA'
    ]
    for col in gene_columns:
        if col not in df.columns:
            df[col] = 0 # Tambahkan kolom gen yang hilang dengan nilai 0

    X_genes = df[gene_columns]
    pca = PCA(n_components=5)
    genes_pca = pca.fit_transform(X_genes)

    for i in range(5):
        df[f'PC{i+1}'] = genes_pca[:, i]

    # Pisahkan fitur (X) dan target (y)
    # Gunakan 'Gender' asli (numerik) dan 'Race_encoded'
    X = df[['Gender', 'Age_at_diagnosis_scaled', 'Race_encoded'] + [f'PC{i+1}' for i in range(5)]]
    y = df['Grade']

    # Simpan nama kolom yang digunakan untuk X (fitur)
    X_columns = X.columns.tolist()

    # Split data train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi model
    models = {
        "Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    # Latih setiap model
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    st.success("Model dan preprocessor berhasil dimuat dan dilatih!")
    
    # Kembalikan semua objek yang dibutuhkan untuk prediksi
    # Perhatikan bahwa le_gender tidak lagi dikembalikan
    return scaler, le_race, RACE_MAP_INPUT_TO_NUMERIC, pca, X_columns, models, gene_columns

# Panggil fungsi inisialisasi model
# Perhatikan perubahan dalam variabel yang diterima
scaler, le_race, RACE_MAP_INPUT_TO_NUMERIC, pca, X_columns, models, gene_columns_expected = initialize_and_train_models()


# --- Antarmuka Pengguna Streamlit ---
st.title("🔬 Prediksi Grade Glioma")
st.write("Aplikasi ini memprediksi grade glioma (LGG atau GBM) berdasarkan data klinis dan mutasi genetik.")
st.write("Masukkan informasi pasien di bawah ini:")

# --- Input Data Klinis ---
st.header("1. Data Klinis")

col1, col2 = st.columns(2)
with col1:
    # Opsi selectbox langsung dari keys GENDER_MAP_INPUT_TO_NUMERIC
    gender_input = st.selectbox("Jenis Kelamin:", list(GENDER_MAP_INPUT_TO_NUMERIC.keys()))
with col2:
    age_input = st.number_input("Usia saat Diagnosis:", min_value=0.0, max_value=100.0, value=40.0, step=0.1)

# Opsi selectbox langsung dari keys RACE_MAP_INPUT_TO_NUMERIC
race_input = st.selectbox("Ras:", list(RACE_MAP_INPUT_TO_NUMERIC.keys()))

# --- Input Mutasi Genetik ---
st.header("2. Mutasi Genetik (Centang jika gen bermutasi)")
genetic_inputs = {}
num_cols = 4 
cols = st.columns(num_cols)
for i, gene in enumerate(gene_columns_expected):
    with cols[i % num_cols]:
        genetic_inputs[gene] = st.checkbox(gene)

# --- Tombol Prediksi ---
st.markdown("---")
if st.button("Dapatkan Prediksi"):
    # Siapkan data input ke dalam format DataFrame
    input_data_dict = {
        'Gender': gender_input,
        'Age_at_diagnosis': age_input,
        'Race': race_input # Masih string di sini
    }
    for gene, is_mutated in genetic_inputs.items():
        input_data_dict[gene] = 1 if is_mutated else 0

    input_df = pd.DataFrame([input_data_dict])

    # Pra-pemrosesan data input menggunakan preprocessor yang sudah dilatih
    try:
        # Konversi string Gender dari selectbox ke numerik menggunakan mapping manual
        gender_encoded_for_prediction = GENDER_MAP_INPUT_TO_NUMERIC.get(input_df['Gender'].iloc[0])
        if gender_encoded_for_prediction is None:
            st.error(f"Nilai Jenis Kelamin '{input_df['Gender'].iloc[0]}' tidak dikenali. Pilih dari {list(GENDER_MAP_INPUT_TO_NUMERIC.keys())}.")
            st.stop()

        # Konversi string Race dari selectbox ke numerik menggunakan mapping dari le_race
        race_encoded_for_prediction = RACE_MAP_INPUT_TO_NUMERIC.get(input_df['Race'].iloc[0])
        if race_encoded_for_prediction is None:
            st.error(f"Nilai Ras '{input_df['Race'].iloc[0]}' tidak dikenali. Pilih dari {list(RACE_MAP_INPUT_TO_NUMERIC.keys())}.")
            st.stop()

        age_scaled = scaler.transform(input_df[['Age_at_diagnosis']])[0][0]

        # Siapkan DataFrame untuk PCA dengan kolom genetik yang diharapkan
        X_genes_input = input_df[gene_columns_expected]
        genes_pca_input = pca.transform(X_genes_input)

        # Buat DataFrame final untuk prediksi dengan semua fitur yang diproses
        final_features_data = {
            'Gender': [gender_encoded_for_prediction], # Menggunakan nilai numerik hasil mapping
            'Age_at_diagnosis_scaled': [age_scaled],
            'Race_encoded': [race_encoded_for_prediction] # Menggunakan nilai numerik hasil mapping
        }
        for i in range(5):
            final_features_data[f'PC{i+1}'] = [genes_pca_input[0, i]]
        
        # Buat DataFrame final
        X_final_features = pd.DataFrame(final_features_data)
        
        # Pastikan urutan kolom sesuai dengan X_columns yang digunakan saat pelatihan
        # Ini krusial agar model menerima fitur dalam urutan yang benar
        X_final_features = X_final_features[X_columns]

        # --- Membuat Prediksi ---
        st.subheader("Hasil Prediksi:")
        for name, model in models.items():
            pred_proba = model.predict_proba(X_final_features)[0].tolist()
            class_labels = ['LGG', 'GBM'] # Berdasarkan grade glioma umum (0 dan 1)
            prediction_dict = {label: round(prob * 100, 2) for label, prob in zip(class_labels, pred_proba)}
            
            st.write(f"**Model {name}:**")
            st.write(f"- Probabilitas LGG: **{prediction_dict['LGG']:.2f}%**")
            st.write(f"- Probabilitas GBM: **{prediction_dict['GBM']:.2f}%**")
            
            # Menampilkan kelas prediksi dengan probabilitas tertinggi
            predicted_class = max(prediction_dict, key=prediction_dict.get)
            st.success(f"Prediksi Grade: **{predicted_class}**")
            st.write("---")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data atau membuat prediksi: {e}")
        st.exception(e) # Menampilkan detail traceback untuk debugging

st.markdown("""
---
**Catatan Penting:**

* Aplikasi ini melatih ulang model dan preprocessor setiap kali di-deploy atau dimulai ulang, karena Anda memilih untuk tidak menggunakan `joblib` untuk penyimpanan model.
* Untuk aplikasi produksi, sangat disarankan untuk melatih model secara terpisah, menyimpan model dan preprocessor yang terlatih menggunakan `joblib`, dan kemudian memuatnya di aplikasi Streamlit. Ini akan sangat meningkatkan kinerja startup aplikasi.
""")
