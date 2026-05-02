# Bike Sharing Dashboard

Dashboard analisis interaktif berbasis **Streamlit** untuk dataset **Bike Sharing Dataset** (UCI Machine Learning Repository).

---

##  Deskripsi

Dashboard ini menjawab empat pertanyaan bisnis utama:

1. **Tren Harian** — Bagaimana tren penyewaan sepeda selama 2011–2012?
2. **Pola Musiman** — Pada bulan apa puncak penyewaan terjadi?
3. **Pola Per Jam** — Bagaimana pola penyewaan dalam sehari?
4. **Cuaca & Lingkungan** — Bagaimana pengaruh cuaca, suhu, kelembapan, dan angin?

Serta **analisis lanjutan** (manual grouping & binning) mencakup segmentasi waktu, user segment, temperatur, kelembapan, dan kecepatan angin.

---

## Cara Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/adindanrsn/ProyekAnalisisData.git
cd ProyekAnalisisData
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan dashboard

```bash
streamlit run dashboard/dashboard.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`.

---

## Dependencies

| Library     | Versi Minimum |
|-------------|---------------|
| streamlit   | 1.32.0        |
| pandas      | 2.0.0         |
| matplotlib  | 3.7.0         |
| seaborn     | 0.13.0        |
| numpy       | 1.24.0        |
| requests    | 2.28.0        |

---

## Deploy ke Streamlit Community Cloud

1. Push repository ini ke GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io).
3. Klik **New app** → pilih repo & file `dashboard/dashboard.py`.
4. Klik **Deploy** — selesai! 

---

## Struktur File

```
.
├───dashboard
│   ├───main_data.csv
│   └───dashboard.py
├───data
│   ├───day.csv
│   └───hour.csv
├───notebook.ipynb
├───README.md
├───requirements.txt
└───url.txt
```

---

## Sumber Data

- **Nama:** Bike Sharing Dataset  
- **Sumber:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)  
- **Format:** ZIP → `day.csv` & `hour.csv`  
- **Periode:** Januari 2011 – Desember 2012
