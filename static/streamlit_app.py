import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import os

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Prediksi Grade Glioma",
    layout="centered",
    initial_sidebar_state="auto"
)

# Mapping manual untuk Gender
GENDER_MAP_INPUT_TO_NUMERIC = {"Female": 0, "Male": 1}

@st.cache_resource
def initialize_and_train_models(dataset_path="glioma_dataset.csv"):
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        st.error(f"Dataset '{dataset_path}' tidak ditemukan.")
        st.stop()

    df['Gender'] = df['Gender'].astype(int)

    le_race = LabelEncoder()
    df['Race_encoded'] = le_race.fit_transform(df['Race'])
    RACE_MAP_INPUT_TO_NUMERIC = {label: index for index, label in enumerate(le_race.classes_)}

    scaler = MinMaxScaler()
    df['Age_at_diagnosis'] = pd.to_numeric(df['Age_at_diagnosis'], errors='coerce')
    df['Age_at_diagnosis'].fillna(df['Age_at_diagnosis'].mean(), inplace=True)
    df['Age_at_diagnosis_scaled'] = scaler.fit_transform(df[['Age_at_diagnosis']])

    gene_columns = [
        'IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA', 'NF1',
        'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 'GRIN2A',
        'IDH2', 'FAT4', 'PDGFRA'
    ]
    for col in gene_columns:
        if col not in df.columns:
            df[col] = 0

    X_genes = df[gene_columns]
    pca = PCA(n_components=5)
    genes_pca = pca.fit_transform(X_genes)

    for i in range(5):
        df[f'PC{i+1}'] = genes_pca[:, i]

    X = df[['Gender', 'Age_at_diagnosis_scaled', 'Race_encoded'] + [f'PC{i+1}' for i in range(5)]]
    y = df['Grade']
    X_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    st.success("Model Naive Bayes berhasil dilatih.")
    return scaler, le_race, RACE_MAP_INPUT_TO_NUMERIC, pca, X_columns, model, gene_columns

# Panggil fungsi pelatihan
scaler, le_race, RACE_MAP_INPUT_TO_NUMERIC, pca, X_columns, model, gene_columns_expected = initialize_and_train_models()

# --- UI Streamlit ---
st.title("🔬 Prediksi Grade Glioma")
st.write("Masukkan data klinis dan mutasi genetik pasien untuk memprediksi grade glioma (LGG atau GBM).")

# Data Klinis
st.header("1. Data Klinis")
col1, col2 = st.columns(2)
with col1:
    gender_input = st.selectbox("Jenis Kelamin:", list(GENDER_MAP_INPUT_TO_NUMERIC.keys()))
with col2:
    age_input = st.number_input("Usia saat Diagnosis:", min_value=0.0, max_value=100.0, value=40.0, step=0.1)

race_input = st.selectbox("Ras:", list(RACE_MAP_INPUT_TO_NUMERIC.keys()))

# Genetik
st.header("2. Mutasi Genetik (Centang jika gen bermutasi)")
genetic_inputs = {}
cols = st.columns(4)
for i, gene in enumerate(gene_columns_expected):
    with cols[i % 4]:
        genetic_inputs[gene] = st.checkbox(gene)

st.markdown("---")

if st.button("Dapatkan Prediksi"):
    input_data = {
        'Gender': gender_input,
        'Age_at_diagnosis': age_input,
        'Race': race_input
    }
    for gene, mutated in genetic_inputs.items():
        input_data[gene] = 1 if mutated else 0

    input_df = pd.DataFrame([input_data])

    try:
        gender_encoded = GENDER_MAP_INPUT_TO_NUMERIC.get(input_df['Gender'].iloc[0])
        race_encoded = RACE_MAP_INPUT_TO_NUMERIC.get(input_df['Race'].iloc[0])
        age_scaled = scaler.transform(input_df[['Age_at_diagnosis']])[0][0]

        X_genes_input = input_df[gene_columns_expected]
        genes_pca_input = pca.transform(X_genes_input)

        final_features = {
            'Gender': [gender_encoded],
            'Age_at_diagnosis_scaled': [age_scaled],
            'Race_encoded': [race_encoded]
        }
        for i in range(5):
            final_features[f'PC{i+1}'] = [genes_pca_input[0, i]]

        X_final = pd.DataFrame(final_features)[X_columns]

        st.subheader("Hasil Prediksi:")
        pred_proba = model.predict_proba(X_final)[0].tolist()
        class_labels = ['LGG', 'GBM']
        prediction = {label: round(prob * 100, 2) for label, prob in zip(class_labels, pred_proba)}

        st.write(f"- Probabilitas LGG: **{prediction['LGG']:.2f}%**")
        st.write(f"- Probabilitas GBM: **{prediction['GBM']:.2f}%**")

        pred_result = max(prediction, key=prediction.get)
        st.success(f"Prediksi Grade: **{pred_result}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        st.exception(e)
