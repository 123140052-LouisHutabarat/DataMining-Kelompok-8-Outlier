# Analisis Komparatif Metode Deteksi dan Strategi Penanganan Outlier terhadap Performa Model Klasifikasi

> **Mata Kuliah:** IF25-3024 - Penambangan Data  
> **Institusi:** Program Studi Teknik Informatika, Institut Teknologi Sumatera (ITERA)  
> **Tahun:** 2026  
> 📁 **Google Drive Proyek:** [Akses Laporan, Logbook](https://drive.google.com/drive/folders/1HUnU5-H2YZy9KZof3AXA5gQiXmUW-E9t?usp=drive_link)

---

## 👥 Anggota Tim (Kelompok 8)

| Nama | NIM | Kontribusi |
|------|-----|-----------|
| Muhammad Nurikhsan | 123140057 | Koordinasi tim, kerangka laporan (BAB I & II), pipeline eksperimen Python |
| Aryasatya Widyatna Akbar | 123140164 | Implementasi deteksi outlier (IQR, Z-Score, Isolation Forest), analisis perbandingan |
| Giovan Lado | 123140068 | Implementasi treatment outlier (Removal, Capping, Log Transform), visualisasi |
| Louis Hutabarat | 123140052 | BAB III Metodologi, alur penelitian, validasi train-test split & data leakage |
| Muhammad Fajri Firdaus | 123140050 | BAB IV Hasil & Pembahasan, confusion matrix, analisis metrik |
| Martino Kelvin | 123140165 | Visualisasi metrik & ROC Curve, daftar pustaka, formatting laporan |

---

## 📋 Deskripsi Proyek

Penelitian ini melakukan **analisis komparatif** antara berbagai kombinasi metode deteksi dan strategi penanganan outlier pada **Online Shoppers Purchasing Intention Dataset** dari UCI Machine Learning Repository. Setiap skenario preprocessing dievaluasi menggunakan model **Random Forest Classifier** untuk menentukan kombinasi yang paling optimal terhadap performa klasifikasi.

---

## 📂 Struktur File

```
├── Source Code - Kelompok 8.ipynb   # Notebook utama eksperimen (Jupyter)
├── source_code.py                    # Script Python mandiri untuk replikasi
└── README.md                         # File ini
```

---

## 📊 Dataset

**Online Shoppers Purchasing Intention Dataset** — UCI Machine Learning Repository

| Atribut | Nilai |
|---------|-------|
| Jumlah sampel | 12.330 |
| Jumlah fitur | 18 (17 prediktor + 1 target) |
| Fitur numerik | 10 |
| Fitur kategorikal/biner | 8 |
| Variabel target | `Revenue` (True/False) |
| Missing values | 0% |
| Duplikat | 125 baris (1,01%) |
| Class imbalance ratio | 5,40 (84,4% False : 15,6% True) |

---

## 🔬 Metodologi

### Alur Penelitian
```
Pengumpulan Data → EDA → Train-Test Split → Preprocessing (Train Only) → Training Model → Evaluasi
```

### Pembagian Data
- **Training set:** 80% (9.764 sampel) — menggunakan stratified sampling (setelah 125 duplikat dihapus)
- **Test set:** 20% (2.441 sampel) — tidak disentuh saat preprocessing untuk validasi objektif

> ⚠️ Pembagian data dilakukan **sebelum** proses deteksi dan penanganan outlier untuk mencegah **data leakage**.

### Metode Deteksi Outlier (Pada Training Set)

| Metode | Tipe | Threshold | Outlier Terdeteksi | Persentase |
|--------|------|-----------|--------------------|------------|
| IQR (1,5×IQR) | Statistik univariat | Q1 − 1,5×IQR s/d Q3 + 1,5×IQR | 5.548 | 56,82% |
| Z-Score (\|z\|>3) | Statistik univariat | \|z\| > 3 | 1.738 | 17,80% |
| Isolation Forest | ML multivariat | contamination = 0,05 | 489 | 5,01% |

### Strategi Penanganan Outlier

| Strategi | Mekanisme | Dampak pada Ukuran Data |
|----------|-----------|------------------------|
| Removal | Hapus baris yang mengandung outlier | Berkurang |
| Capping (Winsorization) | Batasi nilai ke batas atas/bawah IQR atau Z-score | Tetap (9.764 baris) |
| Log Transformation | Terapkan `log(1+x)` pada fitur skewed > 0,5 | Tetap (9.764 baris) |

### Algoritma Klasifikasi

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
```

---

## 🧪 Skenario Eksperimen (10 Skenario)

Hasil diurutkan berdasarkan performa **F1-Score (Macro)** pada test set:

| Skenario | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|----------|----------|-----------|--------|----------|---------|
| 🥇 **Z-Score + Capping** | **0,9009** | **0,8345** | **0,7589** | **0,7891** | **0,9262** |
| Z-Score + Log Transform | 0,8984 | 0,8260 | 0,7585 | 0,7860 | 0,9240 |
| IQR + Log Transform | 0,8984 | 0,8260 | 0,7585 | 0,7860 | 0,9240 |
| IsolationForest + Log Transform | 0,8984 | 0,8260 | 0,7585 | 0,7860 | 0,9240 |
| **Baseline (No Treatment)** | 0,8980 | 0,8247 | 0,7583 | 0,7854 | 0,9241 |
| IsolationForest + Removal | 0,8996 | 0,8332 | 0,7539 | 0,7851 | **0,9297** |
| Z-Score + Removal | 0,8943 | 0,8287 | 0,7305 | 0,7662 | 0,9230 |
| IQR + Capping | 0,8435 | 0,6776 | 0,5426 | 0,5434 | 0,7663 |
| IsolationForest + Capping | 0,8435 | 0,6776 | 0,5426 | 0,5434 | 0,7663 |
| ❌ **IQR + Removal** | 0,8439 | 0,9219 | 0,5013 | **0,4603** | **0,6180** |

---

## 📈 Temuan Utama

### ✅ Skenario Terbaik: Z-Score + Capping
- F1-Score: **0,7891** | AUC-ROC: **0,9262**
- **Mengapa terbaik?** Deteksi Z-Score secara moderat menyaring outlier ekstrem (17,80%), dan teknik *capping* (Winsorization) berbasis standar deviasi `mean ± 3*std` mampu meredam distorsi nilai tanpa membuang baris data latihan, sehingga variabilitas esensial data tetap terjaga utuh bagi Random Forest.

### ✅ Alternatif Terbaik: Log Transformation (semua detektor)
- F1-Score: **0,7860** | AUC-ROC: **0,9240**
- Transformasi logaritma ($\ln(x+1)$) sangat aman dan stabil untuk menstabilkan sebaran fitur numerik yang *highly-skewed* tanpa memengaruhi status baris data (karena tidak ada baris yang dihapus).

### ⚠️ Perhatian: Capping Berbasis IQR (IQR / IF + Capping)
- F1-Score turun drastis ke **0,5434**
- **Mengapa drop?** IQR mendeteksi 56,82% data training sebagai outlier. Capping pada data sebanyak ini memaksa lebih dari separuh data bernilai konstan pada pagar atas ($Q3 + 1.5 \times IQR$). Akibatnya, model kehilangan daya pembeda fitur karena hilangnya variabilitas secara ekstrem.

### ❌ Skenario Terburuk: IQR + Removal
- F1-Score: **0,4603** | AUC-ROC: **0,6180**
- **Mengapa gagal?** Penghapusan 56,82% data training berbasis IQR memicu kelangkaan data latihan (*data scarcity*). Random Forest kehilangan sebagian besar sampel perilaku belanja pelanggan yang bernilai besar (yang terdeteksi sebagai outlier), sehingga model mengalami *underfitting* parah.

---

## 💡 Insight Kunci

> **Outlier pada dataset belanja ini adalah "Legitimate Extreme" (Pencilan Alami), bukan noise.**  
> Nilai durasi kunjungan yang sangat lama (`ProductRelated_Duration`) dan nilai halaman yang sangat tinggi (`PageValues`) adalah indikator/sinyal terkuat yang membedakan pembeli (`Revenue=True`) dari non-pembeli. Menghapus pencilan alami ini (seperti pada IQR + Removal) akan melumpuhkan performa klasifikasi model.

---

## ⚙️ Cara Menjalankan

### Prasyarat
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter scipy
```

### Menjalankan Notebook
```bash
jupyter notebook "Source Code - Kelompok 8.ipynb"
```

### Menjalankan Script Replikasi
```bash
python source_code.py
```

### Dataset
Dataset `online_shoppers_intention.csv` dimuat secara otomatis secara lokal. Jika tidak ditemukan, script akan mengunduhnya secara otomatis dari UCI Machine Learning Repository.

---

## 📏 Metrik Evaluasi

Metrik utama yang digunakan (mengatasi class imbalance):
- **F1-Score (Macro)** — rata-rata harmonik Precision & Recall yang objektif untuk kelas minoritas
- **AUC-ROC** — kemampuan model memisahkan kelas positif dan negatif pada berbagai threshold

---

## 📚 Referensi

1. Alenezi, M., et al. (2023). *Enhancing classification performance in imbalanced datasets*. AIMS Mathematics, 8(12).
2. Mah, P. M., Skalna, I., & Pelech-Pilichowski, T. (2025). *AI-Driven Anomaly Detection in E-Commerce Services*. JTAER, 20(3).
3. Koukaras, P., & Tjortjis, C. (2025). *Data Preprocessing and Feature Engineering for Data Mining*. AI, 6(10).
4. Samariya, D., & Thakkar, A. (2023). *A Comprehensive Survey of Anomaly Detection Algorithms*. Annals of Data Science, 10.
5. Ahsan, M. M., et al. (2021). *Effect of data scaling methods on machine learning algorithms*. Technologies, 9(3).
6. Dube, L., & Verster, T. (2024). *Interpretability of the random forest model under class imbalance*. Data Science in Finance and Economics, 4(3).
