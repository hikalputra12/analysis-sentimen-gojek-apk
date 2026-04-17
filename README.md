# Analisis Sentimen Ulasan Aplikasi Gojek

Proyek ini bertujuan untuk melakukan analisis sentimen terhadap ulasan pengguna aplikasi Gojek di Google Play Store. Proyek mencakup tahap scraping data, prapemrosesan teks, ekstraksi fitur, hingga pelatihan dan evaluasi model machine learning menggunakan beberapa algoritma dan teknik ekstraksi fitur yang berbeda.

## Struktur Proyek

* `scrapping.ipynb`: Notebook untuk mengambil ulasan dari Play Store.
* `training_model.ipynb`: Notebook untuk prapemrosesan data, pelatihan model, dan evaluasi.
* `ulasan_aplikasi.csv`: Dataset hasil scraping.
* `requirements.txt`: Daftar pustaka Python yang dibutuhkan.

## Teknologi yang Digunakan

* **Bahasa Pemrograman**: Python
* **Library Utama**:
    * `pandas` & `numpy` (Manipulasi Data)
    * `scikit-learn` (Machine Learning & TF-IDF)
    * `Gensim` (Word2Vec)
    * `NLTK` / `Sastrawi` (Pembersihan Teks Bahasa Indonesia)
    * `google-play-scraper` (Scraping Data)

## Alur Kerja

1. **Scraping Data**: Mengambil ulasan pengguna secara otomatis dari Play Store.
2. **Preprocessing**: 
   - Case Folding
   - Cleaning (Menghapus simbol/angka)
   - Tokenizing
   - Filtering (Stopword Removal)
   - Stemming (Mengembalikan kata ke bentuk dasar)
3. **Feature Extraction**: Mengubah teks menjadi angka menggunakan TF-IDF atau Word2Vec.
4. **Model Training**: Melatih model menggunakan Support Vector Machine (SVM) dan Random Forest (RF).
5. **Evaluation**: Menguji akurasi model pada data testing.

## Perbandingan Skenario Model

Berikut adalah hasil eksperimen menggunakan berbagai kombinasi algoritma, ekstraksi fitur, dan pembagian data:

### Skenario 1: SVM + TF-IDF (80:20)
* **Pelatihan**: SVM
* **Ekstraksi Fitur**: TF-IDF
* **Pembagian Data**: 80% Latih, 20% Uji
* **Akurasi**:
    * Data Latih: `0.9227`
    * Data Uji: `0.8433`

**Hasil Inferensi Skenario 1:**
| Teks | Hasil |
| :--- | :--- |
| "Aplikasi Gojek sangat membantu saya memesan makanan dengan cepat!" | **NEGATIF** |
| "Biasa saja, tidak ada yang istimewa dari update kali ini." | **NETRAL** |
| "Kecewa banget, aplikasinya sering error dan driver tidak datang." | **NEGATIF** |
| "Aplikasi ini sangat jelek sekali" | **NEGATIF** |
| "Ada beberapa promo yang muncul di halaman utama tadi pagi." | **NETRAL** |

---

### Skenario 2: Random Forest + Word2Vec (80:20)
* **Pelatihan**: Random Forest (RF)
* **Ekstraksi Fitur**: Word2Vec
* **Pembagian Data**: 80% Latih, 20% Uji
* **Akurasi**:
    * Data Latih: `0.8257`
    * Data Uji: `0.6305`

**Hasil Inferensi Skenario 2:**
| Teks | Hasil |
| :--- | :--- |
| "Aplikasi Gojek sangat membantu saya memesan makanan dengan cepat!" | **POSITIF** |
| "Biasa saja, tidak ada yang istimewa dari update kali ini." | **NEGATIF** |
| "Kecewa banget, aplikasinya sering error dan driver tidak datang." | **NEGATIF** |
| "Aplikasi ini sangat jelek sekali" | **NEGATIF** |
| "Ada beberapa promo yang muncul di halaman utama tadi pagi." | **POSITIF** |

---

### Skenario 3: Random Forest + TF-IDF (70:30)
* **Pelatihan**: Random Forest (RF)
* **Ekstraksi Fitur**: TF-IDF
* **Pembagian Data**: 70% Latih, 30% Uji
* **Akurasi**:
    * Data Latih: `0.8026`
    * Data Uji: `0.6973`

**Hasil Inferensi Skenario 3:**
| Teks | Hasil |
| :--- | :--- |
| "Aplikasi Gojek sangat membantu saya memesan makanan dengan cepat!" | **POSITIF** |
| "Biasa saja, tidak ada yang istimewa dari update kali ini." | **NETRAL** |
| "Kecewa banget, aplikasinya sering error dan driver tidak datang." | **NEGATIF** |
| "Aplikasi ini sangat jelek sekali" | **NEGATIF** |
| "Ada beberapa promo yang muncul di halaman utama tadi pagi." | **NETRAL** |

## Kesimpulan
Berdasarkan hasil uji coba, **Skenario 1 (SVM + TF-IDF)** memiliki akurasi tertinggi pada data uji (84.3%), namun hasil inferensinya pada teks positif ("membantu") masih salah diklasifikasikan sebagai negatif. **Skenario 3** memberikan keseimbangan inferensi yang lebih masuk akal untuk kategori positif dan netral meskipun dengan akurasi yang lebih rendah (69.7%).