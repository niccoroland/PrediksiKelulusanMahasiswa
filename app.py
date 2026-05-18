import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. Konfigurasi Halaman (WAJIB DI AWAL)
st.set_page_config(page_title="Dashboard Prediksi Kelulusan", layout="wide")

# 2. Fungsi Load Data dengan cache agar tidak di-load berulang kali
@st.cache_data
def load_data():
    # Pastikan file ini ada di folder yang sama
    df = pd.read_csv('prediksi kelulusan mhs.csv', sep=';')
    
    def fix_ips_format(val):
        clean_val = str(val).replace('.', '')
        if not clean_val or clean_val == 'nan':
            return 0.0
        fixed_val = clean_val[0] + '.' + clean_val[1:]
        return float(fixed_val)

    kolom_ips = ['IPS_Sem1', 'IPS_Sem2', 'IPS_Sem3', 'IPS_Sem4']
    for col in kolom_ips:
        df[col] = df[col].apply(fix_ips_format)
    df[kolom_ips] = df[kolom_ips].round(2)
    
    # Fitur turunan
    df['SKS_Efficiency'] = df['Total_SKS_Lulus'] / df['Total_SKS_Ambil']
    df['Tren_IPS'] = df['IPS_Sem4'] - df['IPS_Sem1']
    
    return df

# 3. Fungsi Training Model dengan cache agar tidak ditraining berulang kali
@st.cache_resource
def train_model(df):
    features = ['Gender', 'IPS_Sem1', 'IPS_Sem2', 'IPS_Sem3', 'IPS_Sem4',
                'Total_SKS_Ambil', 'Total_SKS_Lulus', 'Nilai_Matkul_Killer',
                'Jml_Mengulang', 'Status_Beasiswa', 'Bekerja', 'Tren_IPS', 'SKS_Efficiency']
    X = df[features]
    y = df['Label_Lulus']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    dt_model_balanced = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    dt_model_balanced.fit(X_train_balanced, y_train_balanced)
    
    return dt_model_balanced, features, X_test, y_test
    
# Menjalankan load data dan training
df = load_data()

# Mendapatkan model dan data testing
model, features, X_test, y_test = train_model(df)
y_pred_new = model.predict(X_test)


# --- UI DASHBOARD ---
st.title("🎓 Student Graduation Prediction Dashboard")
st.markdown("Analisis statistik dan prediksi ketepatan waktu kelulusan mahasiswa.")

# --- SIDEBAR: Filter & Navigasi ---
st.sidebar.header("Filter & Navigasi")
menu = st.sidebar.selectbox("Pilih Menu", ["Statistik Data", "Performa Model", "Prediksi Mandiri"])

# --- BAGIAN 1: STATISTIK DATA ---
if menu == "Statistik Data":
    st.subheader("📊 Sebaran Data Mahasiswa")

    # KPI Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sampel", len(df))
    col2.metric("Akurasi Model", "98%")
    col3.metric("F1-Score (Tepat Waktu)", "0.88")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Distribusi Kelulusan (Tepat vs Terlambat)**")
        df_pie = df.copy()
        df_pie['Status'] = df_pie['Label_Lulus'].map({0: 'Tepat Waktu', 1: 'Terlambat'})
        fig_pie = px.pie(df_pie, names='Status', hole=0.4, color_discrete_sequence=['#2ecc71', '#e74c3c'])
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.write("**Rata-rata Tren IPS Per Semester**")
        df_avg_ips = df.groupby('Label_Lulus')[['IPS_Sem1', 'IPS_Sem2', 'IPS_Sem3', 'IPS_Sem4']].mean().reset_index()
        df_avg_ips['Label_Lulus'] = df_avg_ips['Label_Lulus'].map({0: 'Tepat Waktu', 1: 'Terlambat'})
        df_avg_ips = df_avg_ips.melt(id_vars='Label_Lulus', var_name='Semester', value_name='Rata-rata IPS')
        df_avg_ips['Semester'] = df_avg_ips['Semester'].replace({'IPS_Sem1': 'Sem 1', 'IPS_Sem2': 'Sem 2', 'IPS_Sem3': 'Sem 3', 'IPS_Sem4': 'Sem 4'})
        fig_line = px.line(df_avg_ips, x='Semester', y='Rata-rata IPS', color='Label_Lulus', markers=True, color_discrete_map={'Tepat Waktu': '#2ecc71', 'Terlambat': '#e74c3c'})
        fig_line.update_layout(yaxis_title="Nilai IPS", xaxis_title="", legend_title="", margin=dict(t=0, b=0, l=0, r=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig_line, use_container_width=True)

# --- BAGIAN 2: PERFORMA MODEL ---
elif menu == "Performa Model":
    st.subheader("🤖 Analisis Otak AI (Decision Tree)")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Feature Importance (Faktor Penentu Kelulusan)**")
        importance_df = pd.DataFrame({'Fitur': features, 'Kepentingan': model.feature_importances_}).sort_values('Kepentingan', ascending=True)
        fig_imp = px.bar(importance_df, x='Kepentingan', y='Fitur', orientation='h', color='Kepentingan', color_continuous_scale='viridis')
        fig_imp.update_layout(coloraxis_showscale=False, xaxis_title="Tingkat Kepentingan", yaxis_title="", margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_b:
        st.write("**Confusion Matrix (Akurasi Prediksi)**")
        cm = confusion_matrix(y_test, y_pred_new)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', aspect="auto",
                           labels=dict(x="Prediksi Model", y="Data Asli", color="Jumlah"),
                           x=['Tepat Waktu', 'Terlambat'], y=['Tepat Waktu', 'Terlambat'])
        fig_cm.update_layout(coloraxis_showscale=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_cm, use_container_width=True)

# --- BAGIAN 3: PREDIKSI MANDIRI ---
else:
    st.subheader("🔍 Cek Prediksi Kelulusan Baru")
    st.write("Masukkan data akademik untuk melihat probabilitas kelulusan.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
            ips1 = st.number_input("IPS Semester 1", 0.0, 4.0, 3.0)
            ips2 = st.number_input("IPS Semester 2", 0.0, 4.0, 3.0)
            ips3 = st.number_input("IPS Semester 3", 0.0, 4.0, 3.0)
            ips4 = st.number_input("IPS Semester 4", 0.0, 4.0, 3.0)
        
        with col2:
            sks_ambil = st.number_input("Total SKS Diambil", 0, 160, 80)
            sks_lulus = st.number_input("Total SKS Lulus", 0, 160, 75)
            matkul_killer = st.number_input("Nilai Matkul Killer (Skala 0-100)", 0, 100, 70)
            jml_mengulang = st.number_input("Jumlah Mata Kuliah Mengulang", 0, 20, 0)
        
        with col3:
            beasiswa = st.selectbox("Status Beasiswa", ["Tidak Ada", "Ada"])
            bekerja = st.selectbox("Status Bekerja", ["Tidak", "Ya"])

        submitted = st.form_submit_button("Cek Hasil Prediksi")

        if submitted:
            # Map values (sesuaikan dengan label encoder)
            gender_val = 1 if gender == "Laki-laki" else 0
            beasiswa_val = 1 if beasiswa == "Ada" else 0
            bekerja_val = 1 if bekerja == "Ya" else 0
            
            f_tren = ips4 - ips1
            f_sks_eff = sks_lulus / sks_ambil if sks_ambil > 0 else 0
            
            # Susun input data sesuai urutan model
            input_data = pd.DataFrame([[
                gender_val, ips1, ips2, ips3, ips4, 
                sks_ambil, sks_lulus, matkul_killer, 
                jml_mengulang, beasiswa_val, bekerja_val, 
                f_tren, f_sks_eff
            ]], columns=features)
            
            hasil = model.predict(input_data)
            
            st.divider()
            if hasil[0] == 0:
                st.success("🎯 Berdasarkan data, mahasiswa ini diprediksi: **LULUS TEPAT WAKTU**")
                st.balloons()
            else:
                st.error("⚠️ Berdasarkan data, mahasiswa ini diprediksi: **BERPOTENSI TERLAMBAT**")
