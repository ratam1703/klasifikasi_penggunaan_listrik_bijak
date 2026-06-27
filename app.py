import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import io

# ==============================================================================
# CONFIGURASI HALAMAN & TEMA VISUAL PREMIUM
# ==============================================================================
st.set_page_config(
    page_title="Bijak Watt - Klasifikasi Energi",
    page_icon="⚡",
    layout="centered"
)

# Injeksi CSS Modern dengan Google Fonts dan Efek Gradasi/Hover
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Header & Judul */
    .hero-container {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 40px 20px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 35px;
        box-shadow: 0px 10px 25px rgba(30, 58, 138, 0.15);
    }
    .main-title {
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 15px;
        color: #E0E7FF;
        opacity: 0.9;
    }
    
    /* Kartu Hasil (Cards) */
    .result-card {
        background: #FFFFFF;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
    }
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        display: block;
    }
    .metric-value-score {
        font-size: 32px;
        font-weight: 800;
        color: #1E3A8A;
    }
    .metric-value-status {
        font-size: 32px;
        font-weight: 800;
    }
    
    /* Form & Input Styling */
    .stNumberInput div div input {
        border-radius: 10px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F3F4F6;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Hero Section (Header Utama)
st.markdown("""
    <div class="hero-container">
        <div class="main-title">⚡ Bijak Watt</div>
        <div class="subtitle">Sistem Cerdas Analisis & Klasifikasi Efisiensi Energi Listrik Rumah Tangga</div>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# LATIH MODEL DI LATAR BELAKANG (AUTOMATIC TRAINING)
# ==============================================================================
@st.cache_resource
def latih_model_otomatis():
    try:
        df_base = pd.read_excel('household_electricity_usage.xlsx')
        df_base['usage_score'] = df_base['power_watts'] * df_base['duration_minutes'] * df_base['occupancy_count']
        
        q1 = df_base['usage_score'].quantile(0.33)
        q2 = df_base['usage_score'].quantile(0.66)
        
        def hitung_kategori(score):
            if score <= q1: return "Rendah"
            elif score <= q2: return "Normal"
            else: return "Boros"
            
        df_base['kategori'] = df_base['usage_score'].apply(hitung_kategori)
        
        X = df_base[['household_size', 'occupancy_count', 'power_watts', 'duration_minutes']]
        y = df_base['kategori']
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=42)
        model_lr = LogisticRegression(max_iter=1000)
        model_lr.fit(X_train, y_train)
        
        return model_lr, q1, q2
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file 'household_electricity_usage.xlsx' ada di GitHub. Error: {e}")
        return None, None, None

model, q1, q2 = latih_model_otomatis()

def dapatkan_kategori(score):
    if score <= q1: return "Rendah"
    elif score <= q2: return "Normal"
    else: return "Boros"

# ==============================================================================
# DESAIN TABS INTERAKTIF
# ==============================================================================
tab1, tab2 = st.tabs(["📝 Prediksi Tunggal", "📂 Proses File Excel"])

# ------------------------------------------------------------------------------
# TAB 1: INPUT TUNGGAL (SINGLE DATA)
# ------------------------------------------------------------------------------
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("#### Masukkan Parameter Penggunaan Listrik")
    
    col1, col2 = st.columns(2)
    with col1:
        power = st.number_input("Daya Alat Listrik (Watt)", min_value=0.0, step=10.0, value=450.0)
        duration = st.number_input("Durasi Penggunaan (Menit)", min_value=0, step=5, value=120)
    with col2:
        occupancy = st.number_input("Jumlah Penghuni Aktif (Orang)", min_value=0, step=1, value=2)
        h_size = st.number_input("Total Anggota Keluarga di Rumah", min_value=1, step=1, value=4)
        
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Mulai Analisis", type="primary", use_container_width=True):
        if model is not None:
            score_tunggal = power * duration * occupancy
            kategori_tunggal = dapatkan_kategori(score_tunggal)
            
            st.markdown("---")
            st.write("### 📊 Hasil Prediksi Real-Time")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="result-card">
                    <span class="metric-label">Skor Penggunaan (Usage Score)</span>
                    <span class="metric-value-score">{score_tunggal:,.0f}</span>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                color_map = {"Rendah": "#10B981", "Normal": "#3B82F6", "Boros": "#EF4444"}
                warna = color_map.get(kategori_tunggal, "#1E3A8A")
                
                st.markdown(f"""
                <div class="result-card" style="border-top: 5px solid {warna};">
                    <span class="metric-label">Status Efisiensi</span>
                    <span class="metric-value-status" style="color: {warna};">{kategori_tunggal}</span>
                </div>
                """, unsafe_allow_html=True)
                
            # Alert Khusus Edukatif
            if kategori_tunggal == "Boros":
                st.error("💡 **Tips Hemat:** Kurangi durasi pemakaian alat berdaya tinggi atau pastikan alat dimatikan saat ruangan kosong!")
            elif kategori_tunggal == "Normal":
                st.info("💡 **Tips:** Penggunaan energi Anda sudah stabil. Pertahankan pola pemakaian ini.")
            else:
                st.success("💡 **Bagus!** Penggunaan energi sangat efisien.")
        else:
            st.warning("Sistem belum siap. Periksa file dataset Anda.")

# ------------------------------------------------------------------------------
# TAB 2: PROSES BANYAK DATA (BULK EXCEL)
# ------------------------------------------------------------------------------
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("#### Pemrosesan Data Massal Lewat File")
    st.info("Format kolom file harus mengandung: `household_size`, `occupancy_count`, `power_watts`, dan `duration_minutes`.")
    
    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx) di sini", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            df_user = pd.read_excel(uploaded_file)
            required_cols = ['household_size', 'occupancy_count', 'power_watts', 'duration_minutes']
            
            if all(col in df_user.columns for col in required_cols):
                df_user['usage_score'] = df_user['power_watts'] * df_user['duration_minutes'] * df_user['occupancy_count']
                df_user['kategori'] = df_user['usage_score'].apply(dapatkan_kategori)
                
                st.success("🎉 File berhasil diproses secara instan!")
                st.write("##### 👁️ Pratinjau Hasil Klasifikasi Data Anda:")
                st.dataframe(df_user.head(10), use_container_width=True)
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_user.to_excel(writer, index=False, sheet_name='Hasil Bijak Watt')
                buffer.seek(0)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 Unduh File Excel Baru Berkolom Klasifikasi",
                    data=buffer,
                    file_name="hasil_analisis_bijak_watt.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )
            else:
                st.error(f"Format gagal. Pastikan file Anda memiliki kolom-kolom berikut: {required_cols}")
        except Exception as e:
            st.error(f"Terjadi kesalahan pemrosesan file: {e}")
