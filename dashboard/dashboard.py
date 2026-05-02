import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
import zipfile
import requests
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Sharing Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: #2C3E50;
        }
        .subtitle {
            font-size: 1rem;
            color: #636e72;
            margin-top: -10px;
        }
        .metric-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px 20px;
            border-left: 5px solid #00B894;
        }
        section[data-testid="stSidebar"] {
            background-color: #2C3E50;
            color: white;
        }
        section[data-testid="stSidebar"] * {
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
    r = requests.get(url, timeout=30)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    day  = pd.read_csv(z.open("day.csv"))
    hour = pd.read_csv(z.open("hour.csv"))

    # Parse dates
    day["dteday"]  = pd.to_datetime(day["dteday"],  errors="coerce")
    hour["dteday"] = pd.to_datetime(hour["dteday"], errors="coerce")
    hour["datetime"] = hour["dteday"] + pd.to_timedelta(hour["hr"], unit="h")

    # Weather labels
    weather_map = {1: "Clear", 2: "Mist", 3: "Light Rain/Snow", 4: "Heavy Rain/Snow"}
    day["weather_label"] = day["weathersit"].map(weather_map)

    # Manual grouping — time of day
    def time_category(h):
        if   0 <= h <  6: return "Dini Hari"
        elif 6 <= h < 12: return "Pagi"
        elif 12 <= h < 18: return "Siang"
        else:              return "Malam"

    hour["time_category"] = hour["hr"].apply(time_category)

    # Casual ratio segments
    day["casual_ratio"]  = day["casual"] / day["cnt"]
    day["user_segment"]  = pd.cut(
        day["casual_ratio"],
        bins=[0, 0.3, 0.7, 1],
        labels=["Mostly Registered", "Balanced", "Mostly Casual"],
    )

    # Binning
    day["temp_cat"] = pd.cut(day["temp"],      bins=[0, 0.337, 0.655, 1],  labels=["Cold","Mild","Warm"])
    day["hum_cat"]  = pd.cut(day["hum"],       bins=[0, 0.5,   0.7,   1],  labels=["Low","Medium","High"])
    day["wind_cat"] = pd.cut(day["windspeed"], bins=[0, 0.15,  0.25,  0.6], labels=["Low","Medium","High"])

    return day, hour

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🚲 Bike Sharing Dataset — Dashboard Analisis</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Eksplorasi pola penyewaan sepeda tahun 2011–2012 berdasarkan waktu, cuaca, dan faktor lingkungan.</p>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("⏳ Mengunduh & memproses data ..."):
    day, hour = load_data()

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🗂️ Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "📊 Overview",
        "📅 Pertanyaan 1 — Tren Harian",
        "🗓️ Pertanyaan 2 — Pola Bulanan",
        "⏰ Pertanyaan 3 — Pola Per Jam",
        "🌤️ Pertanyaan 4 — Cuaca & Lingkungan",
        "🔬 Analisis Lanjutan (Clustering)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔎 Filter Global")
year_filter = st.sidebar.multiselect(
    "Tahun",
    options=[2011, 2012],
    default=[2011, 2012],
    format_func=lambda x: str(x),
)
if not year_filter:
    year_filter = [2011, 2012]

day_f  = day[day["dteday"].dt.year.isin(year_filter)].copy()
hour_f = hour[hour["dteday"].dt.year.isin(year_filter)].copy()

st.sidebar.markdown("---")
st.sidebar.info("📌 Sumber data: [UCI ML Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def section_header(title: str, insight: str = ""):
    st.subheader(title)
    if insight:
        with st.expander("💡 Insight", expanded=False):
            st.markdown(insight)

# ═════════════════════════════════════════════
# PAGE: OVERVIEW
# ═════════════════════════════════════════════
if page == "📊 Overview":
    st.header("📊 Overview Dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Penyewaan",  f"{day_f['cnt'].sum():,.0f}")
    col2.metric("Rata-rata Harian", f"{day_f['cnt'].mean():,.0f}")
    col3.metric("Hari Teramai",     f"{day_f['cnt'].max():,.0f}")
    col4.metric("Hari Tersenyap",   f"{day_f['cnt'].min():,.0f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Ringkasan Statistik — Data Harian")
        st.dataframe(
            day_f[["cnt","casual","registered","temp","hum","windspeed"]].describe().round(2),
            use_container_width=True,
        )

    with col_b:
        st.markdown("#### Distribusi Total Penyewaan per Tahun")
        yearly = day_f.groupby(day_f["dteday"].dt.year)["cnt"].sum().reset_index()
        yearly.columns = ["Tahun", "Total"]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors_yr = ["#6C5CE7", "#00B894"]
        bars = ax.bar(yearly["Tahun"].astype(str), yearly["Total"], color=colors_yr, edgecolor="black", linewidth=0.6)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{b.get_height():,.0f}',
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title("Total Penyewaan per Tahun", fontweight="bold")
        ax.set_ylabel("Total Penyewaan")
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═════════════════════════════════════════════
# PAGE: PERTANYAAN 1 — TREN HARIAN
# ═════════════════════════════════════════════
elif page == "📅 Pertanyaan 1 — Tren Harian":
    st.header("📅 Pertanyaan 1: Tren Penyewaan Harian (2011–2012)")

    section_header(
        "Tren Penyewaan Sepeda Harian",
        """
**Temuan utama:**
- Jumlah penyewaan menunjukkan **tren meningkat** dari 2011 ke 2012.
- Terdapat **pola musiman** (*seasonality*) yang berulang setiap tahun — naik di pertengahan tahun, turun di awal/akhir tahun.
- Titik terendah terjadi pada **akhir Oktober 2012**, kemungkinan akibat gangguan eksternal.
        """,
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(day_f["dteday"], day_f["cnt"], alpha=0.7, color="#0984E3")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.tick_params(axis="x", which="major", length=7)
    ax.tick_params(axis="x", which="minor", length=4)
    plt.xticks(rotation=45)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Jumlah Penyewaan Sepeda")
    ax.set_title("Tren Penyewaan Sepeda Harian (2011–2012)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("#### Statistik Ringkas per Tahun")
    summary = day_f.groupby(day_f["dteday"].dt.year)["cnt"].agg(["mean","median","min","max","sum"]).rename(
        columns={"mean":"Rata-rata","median":"Median","min":"Min","max":"Max","sum":"Total"}).round(0)
    summary.index.name = "Tahun"
    st.dataframe(summary.astype(int), use_container_width=True)

# ═════════════════════════════════════════════
# PAGE: PERTANYAAN 2 — POLA BULANAN
# ═════════════════════════════════════════════
elif page == "🗓️ Pertanyaan 2 — Pola Bulanan":
    st.header("🗓️ Pertanyaan 2: Puncak Penyewaan & Pola Musiman Bulanan")

    section_header(
        "Perbandingan Tren Bulanan 2011 vs 2012",
        """
**Temuan utama:**
- Puncak penyewaan terjadi pada **Juni–September** di kedua tahun.
- Penyewaan terendah pada **Januari–Februari**.
- Rata-rata penyewaan 2012 **lebih tinggi** di hampir semua bulan → pertumbuhan YoY.
- Puncak 2012 bergeser sedikit lebih lambat (Agustus–September) dibanding 2011 (Juni–Juli).
        """,
    )

    monthly = day_f.groupby([day_f["dteday"].dt.year, day_f["dteday"].dt.month])["cnt"].mean().unstack(level=0)
    monthly.columns = [str(c) for c in monthly.columns]

    bulan_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_line = {"2011": "#6C5CE7", "2012": "#00B894"}
    for yr in monthly.columns:
        ax.plot(monthly.index, monthly[yr], marker="o", label=yr, color=colors_line.get(yr, "#333"))
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(bulan_labels)
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Rata-rata Penyewaan")
    ax.set_title("Perbandingan Tren Penyewaan Sepeda per Bulan (2011 vs 2012)", fontweight="bold", fontsize=13)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("#### Tabel Rata-rata Penyewaan per Bulan")
    monthly.index = bulan_labels[:len(monthly)]
    monthly.index.name = "Bulan"
    st.dataframe(monthly.round(0).astype(int), use_container_width=True)

# ═════════════════════════════════════════════
# PAGE: PERTANYAAN 3 — POLA PER JAM
# ═════════════════════════════════════════════
elif page == "⏰ Pertanyaan 3 — Pola Per Jam":
    st.header("⏰ Pertanyaan 3: Pola Penyewaan Per Jam dalam Sehari")

    section_header(
        "Rata-rata Penyewaan Sepeda per Jam",
        """
**Temuan utama:**
- Dua puncak utama: **pukul 08.00** (berangkat kerja) dan **pukul 17.00–18.00** (pulang kerja).
- Titik minimum terjadi pada **pukul 04.00** (dini hari).
- Pola ini mencerminkan dominasi penggunaan sepeda sebagai **sarana komuter**.
        """,
    )

    hourly_avg = hour_f.groupby("hr")["cnt"].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hourly_avg.index, hourly_avg.values, marker="o", color="#E17055", linewidth=2)
    ax.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.15, color="#E17055")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlabel("Jam")
    ax.set_ylabel("Rata-rata Jumlah Penyewaan")
    ax.set_title("Rata-rata Penyewaan Sepeda per Jam", fontweight="bold", fontsize=13)

    # Annotate peaks
    peak1_hr = hourly_avg.iloc[6:12].idxmax()
    peak2_hr = hourly_avg.iloc[12:].idxmax()
    for ph in [peak1_hr, peak2_hr]:
        ax.annotate(
            f"Puncak\n{hourly_avg[ph]:.0f}",
            xy=(ph, hourly_avg[ph]),
            xytext=(ph + 1, hourly_avg[ph] + 30),
            arrowprops=dict(arrowstyle="->", color="#2D3436"),
            fontsize=9, fontweight="bold", color="#2D3436",
        )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top 5 Jam Tersibuk")
        top5 = hourly_avg.nlargest(5).reset_index()
        top5.columns = ["Jam", "Rata-rata Penyewaan"]
        top5["Jam"] = top5["Jam"].apply(lambda h: f"{h:02d}:00")
        st.dataframe(top5.round(1), use_container_width=True)
    with col2:
        st.markdown("#### Top 5 Jam Tersenyap")
        bot5 = hourly_avg.nsmallest(5).reset_index()
        bot5.columns = ["Jam", "Rata-rata Penyewaan"]
        bot5["Jam"] = bot5["Jam"].apply(lambda h: f"{h:02d}:00")
        st.dataframe(bot5.round(1), use_container_width=True)

# ═════════════════════════════════════════════
# PAGE: PERTANYAAN 4 — CUACA & LINGKUNGAN
# ═════════════════════════════════════════════
elif page == "🌤️ Pertanyaan 4 — Cuaca & Lingkungan":
    st.header("🌤️ Pertanyaan 4: Pengaruh Cuaca & Faktor Lingkungan")

    # -- Boxplot weather
    section_header(
        "Distribusi Penyewaan Berdasarkan Kondisi Cuaca",
        """
- **Clear** → median dan rentang penyewaan tertinggi.
- **Light Rain/Snow** → penyewaan paling rendah dan terkompres.
- **Mist** → berada di tengah, masih cukup aktif.
        """,
    )

    palette_map = {
        "Clear": "#FDCB6E", "Mist": "#B2BEC3",
        "Light Rain/Snow": "#74B9FF", "Heavy Rain/Snow": "#2D3436",
    }
    preferred_order = ["Clear", "Mist", "Light Rain/Snow", "Heavy Rain/Snow"]
    existing = day_f["weather_label"].value_counts().index.tolist()
    order_w  = [x for x in preferred_order if x in existing]
    palette_w = [palette_map[x] for x in order_w]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(x="weather_label", y="cnt", data=day_f, order=order_w,
                palette=palette_w, width=0.6, showfliers=False, ax=ax)
    ax.set_title("Distribusi Penyewaan berdasarkan Kondisi Cuaca", fontsize=14, fontweight="bold")
    ax.set_xlabel("Kondisi Cuaca")
    ax.set_ylabel("Jumlah Penyewaan")
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    col1, col2 = st.columns(2)

    # -- Heatmap korelasi
    with col1:
        section_header(
            "Korelasi Variabel Numerik",
            """
- **Suhu (temp)** memiliki korelasi positif paling kuat (0.63) terhadap penyewaan.
- **Kelembapan (hum)** & **kecepatan angin (windspeed)** berkorelasi negatif lemah.
            """,
        )
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            day_f[["cnt","temp","hum","windspeed"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", ax=ax2,
            linewidths=0.5, vmin=-1, vmax=1,
        )
        ax2.set_title("Heatmap Korelasi", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # -- Scatter suhu vs penyewaan
    with col2:
        section_header(
            "Hubungan Suhu & Jumlah Penyewaan",
            """
- Hubungan **positif tidak linier** — penyewaan naik seiring suhu, namun melambat di suhu tinggi.
- Penyebaran luas pada suhu menengah → faktor lain turut berperan.
            """,
        )
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.scatter(day_f["temp"], day_f["cnt"], alpha=0.4, color="#6C5CE7", edgecolors="none")
        ax3.set_xlabel("Temperature (normalized)")
        ax3.set_ylabel("Bike Rentals")
        ax3.set_title("Suhu vs Jumlah Penyewaan", fontweight="bold")
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

# ═════════════════════════════════════════════
# PAGE: ANALISIS LANJUTAN (CLUSTERING)
# ═════════════════════════════════════════════
elif page == "🔬 Analisis Lanjutan (Clustering)":
    st.header("🔬 Analisis Lanjutan — Manual Grouping & Binning")

    tab1, tab2, tab3, tab4 = st.tabs(["⏰ Waktu Hari", "👥 User Segment", "🌡️ Temperatur", "💧 Kelembapan & Angin"])

    # ── TAB 1: Waktu Hari
    with tab1:
        section_header(
            "Rata-rata Penyewaan per Kategori Waktu",
            "Aktivitas tertinggi terjadi pada **Siang** hari, terendah pada **Dini Hari**. Pagi dan Malam berada di kisaran menengah sesuai pola aktivitas komuter.",
        )
        order_t = ["Dini Hari", "Pagi", "Siang", "Malam"]
        colors_t = ["#2C3E50", "#FDCB6E", "#00B894", "#6C5CE7"]
        time_group = hour_f.groupby("time_category")["cnt"].mean().reindex(order_t)

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(time_group.index, time_group.values, color=colors_t, edgecolor="black", linewidth=0.6)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f'{b.get_height():.0f}', ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_title("Rata-rata Penyewaan Sepeda per Waktu Hari", fontsize=15, fontweight="bold")
        ax.set_xlabel("Kategori Waktu", fontsize=12)
        ax.set_ylabel("Rata-rata Jumlah Penyewaan", fontsize=12)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── TAB 2: User Segment
    with tab2:
        section_header(
            "Distribusi Total Penyewaan per User Segment",
            "Sekitar **79%** observasi masuk kategori *Mostly Registered*, menunjukkan dominasi pengguna terdaftar. Kategori *Mostly Casual* hampir tidak muncul.",
        )
        user_group = day_f.groupby("user_segment", observed=True)["cnt"].sum()
        user_group_f = user_group[user_group > 0]
        colors_u = ["#00B894", "#FDCB6E", "#6C5CE7"][: len(user_group_f)]
        explode = [0.05 if v == user_group_f.max() else 0 for v in user_group_f.values]

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax2.pie(
            user_group_f, labels=user_group_f.index,
            autopct="%1.1f%%", startangle=90, colors=colors_u,
            explode=explode, pctdistance=0.75, textprops={"fontsize": 11},
        )
        centre_circle = plt.Circle((0, 0), 0.55, fc="white")
        ax2.add_artist(centre_circle)
        for at in autotexts:
            at.set_fontsize(11); at.set_fontweight("bold")
        ax2.set_title("Distribusi Total Penyewaan per User Segment", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── TAB 3: Temperatur
    with tab3:
        section_header(
            "Distribusi Penyewaan berdasarkan Kategori Temperatur",
            "Median penyewaan **tertinggi** pada kondisi *Warm*, **terendah** pada *Cold*. Kategori *Mild* menunjukkan sebaran (IQR) paling lebar.",
        )
        palette_t = {"Cold": "#74B9FF", "Mild": "#A0AEC0", "Warm": "#FDCB6E"}

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="temp_cat", y="cnt", data=day_f,
                    order=["Cold", "Mild", "Warm"],
                    palette=palette_t, width=0.6, showfliers=False, ax=ax3)
        ax3.set_title("Distribusi Penyewaan berdasarkan Temperatur", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Kategori Temperatur")
        ax3.set_ylabel("Jumlah Penyewaan")
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    # ── TAB 4: Kelembapan & Angin
    with tab4:
        col_a, col_b = st.columns(2)

        with col_a:
            section_header(
                "Distribusi berdasarkan Kelembapan",
                "Median tertinggi pada *Medium*, terendah pada *High*. Perbedaan antarkategori relatif kecil.",
            )
            palette_h = {"Low": "#81ECEC", "Medium": "#FDCB6E", "High": "#E17055"}
            fig4, ax4 = plt.subplots(figsize=(6, 5))
            sns.boxplot(x="hum_cat", y="cnt", data=day_f,
                        order=["Low","Medium","High"],
                        palette=palette_h, width=0.6, showfliers=False, ax=ax4)
            ax4.set_title("Penyewaan vs Kelembapan", fontweight="bold")
            ax4.set_xlabel("Kategori Kelembapan")
            ax4.set_ylabel("Jumlah Penyewaan")
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

        with col_b:
            section_header(
                "Distribusi berdasarkan Kecepatan Angin",
                "Median tertinggi pada *Low wind*, terendah pada *High wind*. Perbedaan pengaruh relatif kecil dibanding suhu & cuaca.",
            )
            palette_wi = {"Low": "#81ECEC", "Medium": "#0984E3", "High": "#6C5CE7"}
            fig5, ax5 = plt.subplots(figsize=(6, 5))
            sns.boxplot(x="wind_cat", y="cnt", data=day_f,
                        order=["Low","Medium","High"],
                        palette=palette_wi, width=0.6, showfliers=False, ax=ax5)
            ax5.set_title("Penyewaan vs Kecepatan Angin", fontweight="bold")
            ax5.set_xlabel("Kategori Kecepatan Angin")
            ax5.set_ylabel("Jumlah Penyewaan")
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Dashboard dibuat dengan menggunakan Streamlit &nbsp;|&nbsp; Data: Bike Sharing Dataset (UCI ML Repository)</small></center>",
    unsafe_allow_html=True,
)
