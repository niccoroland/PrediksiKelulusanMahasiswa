import streamlit as st
import pandas as pd
import joblib

# 1. Load Model & Data
@st.cache_resource
def load_assets():
    model = joblib.load('model_lulus.pkl')
    # Pastikan file CSV menggunakan separator yang benar
    df = pd.read_csv('prediksi kelulusan mhs.csv', sep=';')
    return model, df

model, df = load_assets()

st.set_page_config(page_title="Prediksi Kelulusan", layout="wide")
st.title("🎓 Dashboard Prediksi Kelulusan Mahasiswa")

menu = st.sidebar.selectbox("Pilih Menu", ["Statistik Data", "Cek Prediksi"])

if menu == "Cek Prediksi":
    st.subheader("🔍 Analisis Data Kelulusan Mahasiswa")

    # 1. RESTUKTURISASI TAMPILAN: Membagi halaman menjadi 2 tab
    tab1, tab2 = st.tabs(["📋 Prediksi Individu (Form)", "📁 Prediksi Massal (Upload CSV Angkatan 2023)"])

    # =========================================================================
    # TAB 1: PREDIKSI INDIVIDU (FORM)
    # =========================================================================
    with tab1:
        st.write("Silakan masukkan data mahasiswa satu per satu.")
        
        # --- TEMPEL KODE FORM LAMA DI SINI ---
        with st.form("form_prediksi"):
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

            btn = st.form_submit_button("Analisis Kelulusan")

            if btn:
                # --- PRE-PROCESSING INPUT ---
                gender_val = 1 if gender == "Laki-laki" else 0
                beasiswa_val = 1 if beasiswa == "Ada" else 0
                bekerja_val = 1 if bekerja == "Ya" else 0

                tren_ips = ips4 - ips1
                sks_efficiency = sks_lulus / sks_ambil if sks_ambil > 0 else 0

                input_data = [[
                    gender_val, ips1, ips2, ips3, ips4,
                    sks_ambil, sks_lulus, matkul_killer, jml_mengulang,
                    beasiswa_val, bekerja_val, tren_ips, sks_efficiency
                ]]

                hasil = model.predict(input_data)

                st.divider()
                if hasil[0] == 0:
                    st.success("🎯 Hasil Prediksi: **LULUS TEPAT WAKTU**")
                    st.balloons()
                else:
                    st.error("⚠️ Hasil Prediksi: **BERPOTENSI TERLAMBAT**")

    # =========================================================================
    # TAB 2: PREDIKSI MASSAL (UPLOAD CSV ANGKATAN 2023)
    # =========================================================================
    with tab2:
        st.write("Silakan unggah dataset berformat CSV untuk memprediksi banyak mahasiswa sekaligus.")
        
        # Komponen upload file CSV
        uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])

        if uploaded_file is not None:
            # Membaca data yang diunggah
            try:
                df_massal = pd.read_csv(uploaded_file, sep=None, engine='python')
            except Exception as e:
                st.error(f"Gagal membaca file CSV: {e}")
                st.stop()

            # Validasi Kolom Wajib (berdasarkan format data training asli)
            wajib_cols = [
                'Gender', 'IPS_Sem1', 'IPS_Sem2', 'IPS_Sem3', 'IPS_Sem4',
                'Total_SKS_Ambil', 'Total_SKS_Lulus', 'Nilai_Matkul_Killer',
                'Jml_Mengulang', 'Status_Beasiswa', 'Bekerja'
            ]
            
            missing_cols = [col for col in wajib_cols if col not in df_massal.columns]

            if missing_cols:
                st.error(f"❌ Gagal memproses! Kolom berikut tidak ditemukan di dalam file CSV: **{', '.join(missing_cols)}**")
            else:
                st.success("✅ File valid! Memproses prediksi massal...")

                # Filtering dataset sesuai kolom input mentah
                X_massal_raw = df_massal[wajib_cols].copy()
                
                # --- PRE-PROCESSING: BEBERSIH DATA FORMAT IPS TIPE STRING ---
                def fix_ips_format(val):
                    clean_val = str(val).replace('.', '')
                    if not clean_val or clean_val.lower() == 'nan':
                        return 0.0
                    if len(clean_val) > 1:
                        return float(clean_val[0] + '.' + clean_val[1:])
                    return float(clean_val)

                kolom_ips = ['IPS_Sem1', 'IPS_Sem2', 'IPS_Sem3', 'IPS_Sem4']
                for col_ips in kolom_ips:
                    X_massal_raw[col_ips] = X_massal_raw[col_ips].apply(fix_ips_format)

                # --- PRE-PROCESSING: MAPPING KATEGORI STRING KE NUMERIK ---
                mapping_gender = {'Laki-laki': 1, 'Perempuan': 0, 'Laki-Laki': 1}
                mapping_umum = {'Ada': 1, 'Tidak Ada': 0, 'Ya': 1, 'Tidak': 0}
                
                # Mapping nilai abjad menjadi angka (skala 0-100)
                mapping_nilai_huruf = {
                    'A': 85, 'A-': 80, 
                    'B+': 75, 'B': 70, 'B-': 65, 
                    'C+': 60, 'C': 55, 'C-': 50, 
                    'D': 40, 'E': 0
                }
                
                if X_massal_raw['Gender'].dtype == object:
                    X_massal_raw['Gender'] = X_massal_raw['Gender'].replace(mapping_gender)
                    
                if X_massal_raw['Status_Beasiswa'].dtype == object:
                    X_massal_raw['Status_Beasiswa'] = X_massal_raw['Status_Beasiswa'].replace(mapping_umum)
                    
                if X_massal_raw['Bekerja'].dtype == object:
                    X_massal_raw['Bekerja'] = X_massal_raw['Bekerja'].replace(mapping_umum)
                    
                if X_massal_raw['Nilai_Matkul_Killer'].dtype == object:
                    X_massal_raw['Nilai_Matkul_Killer'] = X_massal_raw['Nilai_Matkul_Killer'].str.strip()
                    X_massal_raw['Nilai_Matkul_Killer'] = X_massal_raw['Nilai_Matkul_Killer'].replace(mapping_nilai_huruf)
                    
                # Pastikan di-cast ke bentuk numerik setelah direplace
                X_massal_raw['Gender'] = pd.to_numeric(X_massal_raw['Gender'], errors='coerce')
                X_massal_raw['Status_Beasiswa'] = pd.to_numeric(X_massal_raw['Status_Beasiswa'], errors='coerce')
                X_massal_raw['Bekerja'] = pd.to_numeric(X_massal_raw['Bekerja'], errors='coerce')
                X_massal_raw['Nilai_Matkul_Killer'] = pd.to_numeric(X_massal_raw['Nilai_Matkul_Killer'], errors='coerce')

                # Menghitung Fitur Turunan (sesuai data training)
                X_massal_raw['SKS_Efficiency'] = X_massal_raw.apply(
                    lambda row: row['Total_SKS_Lulus'] / row['Total_SKS_Ambil'] if row['Total_SKS_Ambil'] > 0 else 0, axis=1
                )
                X_massal_raw['Tren_IPS'] = X_massal_raw['IPS_Sem4'] - X_massal_raw['IPS_Sem1']

                # Susun dataset sesuai urutan baku 13 fitur model
                fitur_model = [
                    'Gender', 'IPS_Sem1', 'IPS_Sem2', 'IPS_Sem3', 'IPS_Sem4',
                    'Total_SKS_Ambil', 'Total_SKS_Lulus', 'Nilai_Matkul_Killer',
                    'Jml_Mengulang', 'Status_Beasiswa', 'Bekerja', 'Tren_IPS', 'SKS_Efficiency'
                ]
                X_massal = X_massal_raw[fitur_model]
                
                # Imputasi sederhana jika ada data kosong (handling NaN)
                X_massal = X_massal.fillna(0)

                # Jalankan prediksi massal menggunakan model AI
                try:
                    prediksi_hasil = model.predict(X_massal)
                    
                    # Buat salinan DataFrame untuk hasil
                    df_hasil = df_massal.copy()

                    # Mapping prediksi 0 -> Tepat Waktu, 1 -> Terlambat (Sesuaikan konsistensi label)
                    mapping_label = {0: 'Tepat Waktu', 1: 'Terlambat'}
                    df_hasil['Status Prediksi'] = [mapping_label.get(p, 'Terlambat') for p in prediksi_hasil]

                    # Menampilkan Tabel Hasil
                    st.write("### 📄 Tabel Hasil Prediksi Data Mahasiswa")
                    st.dataframe(df_hasil)

                    # --- VISUALISASI DISTRIBUSI (DI BAWAH TABEL MASSAL) ---
                    st.write("### 📊 Statistik Prediksi Kelulusan Angkatan 2023")
                    
                    import matplotlib.pyplot as plt
                    
                    # Hitung rekap jumlah Tepat Waktu vs Terlambat
                    rekap_status = df_hasil['Status Prediksi'].value_counts().reset_index()
                    rekap_status.columns = ['Status', 'Jumlah']
                    
                    # Atur warna sesuai instruksi (Hijau untuk Tepat Waktu, Merah untuk Terlambat)
                    warna_custom = {'Tepat Waktu': '#2ecc71', 'Terlambat': '#e74c3c'}
                    warna_grafik = [warna_custom[stats] for stats in rekap_status['Status']]

                    col_chart1, col_chart2 = st.columns(2)

                    with col_chart1:
                        # Grafik 1: Pie Chart
                        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
                        ax_pie.pie(rekap_status['Jumlah'], labels=rekap_status['Status'], 
                                   autopct='%1.1f%%', startangle=140, colors=warna_grafik, explode=[0.05]*len(rekap_status))
                        ax_pie.set_title("Persentase Distribusi")
                        st.pyplot(fig_pie)

                    with col_chart2:
                        # Grafik 2: Bar Chart
                        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                        bar_plot = ax_bar.bar(rekap_status['Status'], rekap_status['Jumlah'], color=warna_grafik)
                        ax_bar.set_title("Jumlah Riil Mahasiswa")
                        ax_bar.set_ylabel("Total Mahasiswa")
                        # Menambahkan label angka di atas batang
                        for bar in bar_plot:
                            yval = bar.get_height()
                            ax_bar.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontweight='bold')
                        st.pyplot(fig_bar)

                    # --- FITUR DOWNLOAD ---
                    st.write("---")
                    csv_download = df_hasil.to_csv(index=False, sep=';').encode('utf-8')
                    st.download_button(
                        label="⬇️ Download File Hasil Prediksi (.CSV)",
                        data=csv_download,
                        file_name="Hasil_Prediksi_Massal_Angkatan_2023.csv",
                        mime="text/csv",
                    )

                except ValueError as e:
                    st.error(f"⚠️ Model AI mengalami error saat prediksi. Pastikan tipe data pada CSV sesuai. Error code: {e}")

elif menu == "Statistik Data":
    st.write("Silakan masukkan kode visualisasi data kamu di sini.")
