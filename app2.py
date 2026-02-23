import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Sistem Rekomendasi Wisata Karanganyar",
    layout="wide"
)

# --- FUNGSI LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Membaca dataset
        df = pd.read_csv("WK_combination_final.csv")
        
        # Membersihkan data null pada kolom penting
        df['deskripsi_kombinasi_clean'] = df['deskripsi_kombinasi_clean'].fillna('')
        df['nama_wisata'] = df['nama_wisata'].fillna('Tanpa Nama')
        return df
    except FileNotFoundError:
        st.error("File 'WK_combination_final.csv' tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py.")
        return None

# --- FUNGSI MODEL REKOMENDASI ---
@st.cache_resource
def build_model(df):
    # Membuat model TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['deskripsi_kombinasi_clean'])
    
    # Menghitung Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim

def get_recommendations_by_query(query, df, tfidf, tfidf_matrix):
    try:
        # Transformasi input query pengguna menjadi vektor TF-IDF
        query_vec = tfidf.transform([query])
        
        # Menghitung cosine similarity antara query dan semua data wisata
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Mengurutkan dari kemiripan tertinggi (mengambil 5 teratas)
        top_indices = sim_scores.argsort()[-5:][::-1]
        
        # Opsional: Hanya mengambil yang memiliki kemiripan > 0 agar relevan
        valid_indices = [i for i in top_indices if sim_scores[i] > 0]
        
        if not valid_indices:
            return None
            
        return df.iloc[valid_indices]
    except Exception:
        return None

def get_recommendations(nama_wisata, df, cosine_sim):
    try:
        # Mencari index wisata yang dipilih
        idx = df[df['nama_wisata'] == nama_wisata].index[0]
        
        # Mengambil skor kemiripan
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Mengambil 5 teratas (index 0 adalah wisata itu sendiri, jadi mulai dari index 1)
        sim_scores = sim_scores[1:6]
        wisata_indices = [i[0] for i in sim_scores]
        
        return df.iloc[wisata_indices]
    except IndexError:
        return None

# --- KOMPONEN KARTU WISATA (KLIK GAMBAR & TEKS) ---
def render_wisata_card(row):
    # Mengamankan URL gambar (jika kosong, gunakan placeholder)
    img_url = row.get('url_gambar', '')
    if pd.isna(img_url) or not str(img_url).startswith('http'):
        img_url = "https://via.placeholder.com/300x200?text=Gambar+Tidak+Tersedia"
    
    # Meng-encode nama wisata agar aman dimasukkan ke dalam URL Parameter
    encoded_name = urllib.parse.quote(row['nama_wisata'])
    
    # Desain HTML kustom: Gambar dan Teks yang bisa diklik
    # Saat diklik, akan mengarahkan ke URL dengan parameter ?wisata=NamaWisata
    html_code = f"""
    <div style="text-align: center; margin-bottom: 30px; transition: transform 0.2s ease-in-out;" 
         onmouseover="this.style.transform='scale(1.05)'" 
         onmouseout="this.style.transform='scale(1)'">
        <a href="?wisata={encoded_name}" target="_self" style="text-decoration: none; color: inherit; display: block; cursor: pointer;">
            <img src="{img_url}" alt="{row['nama_wisata']}" style="width: 100%; height: 180px; object-fit: cover; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
            <h4 style="margin-top: 12px; font-size: 1.1rem; font-weight: 600;">{row['nama_wisata']}</h4>
        </a>
    </div>
    """
    
    # Menampilkan komponen HTML di Streamlit
    st.markdown(html_code, unsafe_allow_html=True)


# --- HALAMAN UTAMA (HOME) ---
def show_home_page(df, tfidf, tfidf_matrix):
    st.title("Sistem Rekomendasi Wisata")
    st.markdown("Temukan destinasi wisata populer dan klik untuk melihat detail serta rekomendasi tempat serupa.")
    
    # --- MENU PENCARIAN ---
    st.subheader("Cari Wisata Sesuai Keinginanmu")
    search_query = st.text_input(
        ""
    )
    
    # Jika pengguna memasukkan teks pencarian
    if search_query.strip():
        st.markdown(f"**Rekomendasi untuk:** *{search_query}*")
        search_results = get_recommendations_by_query(search_query, df, tfidf, tfidf_matrix)
        
        if search_results is not None and not search_results.empty:
            rec_cols = st.columns(5)
            # Menampilkan hasil (maksimal 5)
            for index, row in enumerate(search_results.to_dict('records')):
                with rec_cols[index]:
                    render_wisata_card(row)
        else:
            st.warning("Maaf, tidak ditemukan wisata yang cocok dengan pencarian Anda. Coba gunakan kata kunci lain.")
            
    st.divider()

    st.subheader("Wisata Populer")
    
    # Daftar wisata populer yang akan tampil di halaman depan
    popular_names = [
        "Air Terjun Grojogan Sewu", 
        "Air Terjun Jumog", 
        "Candi Cetho", 
        "Kebun Teh Kemuning",
        "Telaga Madirda",
        "Candi Sukuh",
        "The Lawu Park",
        "Bukit Sekipan",
        "The Lawu Fresh"
    ]
    
    # Menyaring data berdasarkan wisata populer
    popular_df = df[df['nama_wisata'].isin(popular_names)]
    
    # Jika data wisata populer tidak ditemukan di CSV, ambil 8 data pertama sebagai cadangan
    if popular_df.empty:
        popular_df = df.head(8)
    else:
        popular_df = popular_df.head(8)

    # Membuat grid 4 kolom untuk menampilkan daftar wisata
    cols = st.columns(4)
    for index, row in enumerate(popular_df.to_dict('records')):
        col_idx = index % 4
        with cols[col_idx]:
            render_wisata_card(row)


# --- HALAMAN DETAIL WISATA & REKOMENDASI ---
def show_detail_page(df, cosine_sim, selected_name):
    # Tombol untuk kembali ke beranda utama
    if st.button("⬅️ Kembali ke Beranda"):
        st.query_params.clear() # Menghapus parameter dari URL
        st.rerun() # Memuat ulang halaman
    
    st.divider()

    # Validasi apakah wisata ada di dalam dataset
    if selected_name not in df['nama_wisata'].values:
        st.error(f"Wisata '{selected_name}' tidak ditemukan di database.")
        return

    # Mengambil data lengkap untuk wisata yang dipilih
    wisata_data = df[df['nama_wisata'] == selected_name].iloc[0]

    # --- BAGIAN ATAS: INFO DETAIL WISATA ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        img_url = wisata_data.get('url_gambar', '')
        if pd.isna(img_url) or not str(img_url).startswith('http'):
            img_url = "https://via.placeholder.com/600x400?text=Gambar+Tidak+Tersedia"
            
        # Menampilkan gambar wisata dengan ukuran menyesuaikan kolom
        st.markdown(f'<img src="{img_url}" style="width:100%; border-radius:12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">', unsafe_allow_html=True)

    with col2:
        st.title(wisata_data['nama_wisata'])
        st.markdown(f"**Kategori:** {wisata_data.get('kategori', '-')}")
        
        # Mengambil informasi operasional dan harga
        jam_buka = wisata_data.get('jam_operasional', 'Informasi tidak tersedia')
        harga = wisata_data.get('harga_tiket', 'Cek langsung ke lokasi') # Akan menggunakan default ini jika kolom tidak ada
        
        st.markdown(f"🕒 **Jam Operasional:** {jam_buka}")
        
        # Link Google Maps
        if pd.notna(wisata_data.get('url_gmaps')):
            st.markdown(f"📍 **Lokasi:** [Buka di Google Maps]({wisata_data['url_gmaps']})")

    # Deskripsi & Fasilitas
    st.markdown("### 📖 Deskripsi")
    deskripsi = wisata_data.get('deskripsi', 'Deskripsi tidak tersedia.')
    st.write(deskripsi)
    
    if pd.notna(wisata_data.get('fasilitas')):
        st.markdown("### 🚿 Fasilitas")
        st.write(wisata_data['fasilitas'])

    st.divider()

    # --- BAGIAN BAWAH: 5 REKOMENDASI WISATA ---
    st.subheader(f"Rekomendasi wisata yang mirip dengan {selected_name}:")
    
    # Memanggil fungsi rekomendasi
    recommendations = get_recommendations(selected_name, df, cosine_sim)
    
    if recommendations is not None and not recommendations.empty:
        # Menampilkan 5 rekomendasi dalam bentuk grid sejajar
        rec_cols = st.columns(5)
        for index, row in enumerate(recommendations.to_dict('records')):
            with rec_cols[index]:
                render_wisata_card(row)
    else:
        st.info("Belum ada rekomendasi yang cocok untuk wisata ini.")


# --- PROGRAM UTAMA ---
def main():
    # 1. Muat Data
    df = load_data()
    
    if df is not None:
        # 2. Bangun Model Cosine Similarity
        tfidf, tfidf_matrix, cosine_sim = build_model(df)
        
        # 3. Cek parameter URL untuk mengetahui posisi halaman
        # Streamlit versi baru menggunakan st.query_params
        selected_wisata = st.query_params.get("wisata")
        
        # 4. Navigasi Halaman
        if not selected_wisata:
            # Tampilkan Beranda jika tidak ada yang diklik
            show_home_page(df, tfidf, tfidf_matrix)
        else:
            # Tampilkan Detail & Rekomendasi jika ada parameter "wisata" di URL
            show_detail_page(df, cosine_sim, selected_wisata)

if __name__ == '__main__':
    main()