# Face Recognition Kasir Pingkal

Repository ini berisi sistem **Face Recognition** berbasis Python, OpenCV, dan DeepFace (**FaceNet512**) untuk kebutuhan kasir/member Warung Pingkal. 

Sistem ini mendukung pengenalan wajah real-time, pendaftaran member baru, manajemen data biometrik lokal, integrasi API backend (Laravel), serta suite evaluasi performa komprehensif (DeepFace vs ORB hybrid).

---

## 🚀 Fitur Utama

- **Real-time Face Recognition**: Deteksi dan identifikasi wajah dari kamera dengan model `FaceNet512` & Cosine Distance.
- **Visual GUI / OpenCV UI**: Antarmuka visual intuitif untuk kasir (Recognize, Register Wajah, List Data Member, Hapus Data).
- **ORB Hybrid Recognition**: Kombinasi feature matching ORB dengan DeepFace untuk pencarian cepat & akurat.
- **Manajemen Embedding Otomatis**: Penyimpanan embedding lokal (`known_faces_embeddings.json`) yang diperbarui secara otomatis saat ada penambahan/penghapusan member.
- **Integrasi API Backend**: Pengiriman event deteksi member & pendaftaran ke API server Laravel secara langsung.
- **Comprehensive Evaluation Tools**: Evaluasi metrik (Accuracy, Precision, Recall, F1-Score), Leave-One-Out (LOO) Cross Validation, & Pairwise Similarity evaluation.

---

## 📁 Struktur Repository

```text
facerecog/
├── api/                             # Handler & Endpoint API Backend
├── api_client.py                    # Client HTTP untuk integrasi API Laravel
├── config.py                        # Konfigurasi sistem & ambang batas (threshold)
├── download_lfw_unknown.py          # Script download dataset LFW (wajah non-member)
├── eval_deepface_loo.py             # Evaluasi Leave-One-Out (LOO) Cross-Validation
├── evaluate_full_convergentai.py    # Suite evaluasi lengkap (Accuracy, Precision, Recall, F1)
├── evaluate_orb_vs_deepface.py      # Script pembanding DeepFace vs ORB baseline
├── evaluate_pairwise_convergentai.py # Evaluasi Pairwise similarity
├── gui.py                           # Antarmuka Visual GUI & komponen OpenCV menu
├── known_faces/                     # Folder sampel foto registrasi member (hanya .gitkeep di git)
├── main.py                          # Entry point utama aplikasi (CLI & GUI launcher)
├── orb_hybrid_realtime.py           # Engine pengenalan wajah hybrid (ORB + DeepFace)
├── recognition.py                   # Core engine deteksi & pengenalan wajah
├── storage.py                       # Manajemen dataset lokal & JSON embedding generator
├── README.md                        # Dokumentasi repository
└── LICENSE                          # Lisensi MIT
```

> **Catatan Keamanan Biometrik**: Folder `known_faces/` dan file `known_faces_embeddings.json` diabaikan oleh Git (`.gitignore`) untuk melindungi data biometrik sensitif customer.

---

## 🛠️ Setup & Instalasi

### 1. Prasyarat
- Python 3.9+
- Webcam / Kamera
- Koneksi Internet (pada pengujian pertama untuk mengunduh bobot pretrained DeepFace)

### 2. Clone & Environment
```bash
git clone https://github.com/bayuaaar12/facerecog.git
cd facerecog

# Buat virtual environment
python3 -m venv venv

# Aktivasi virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install Dependency
```bash
pip install deepface tensorflow opencv-python numpy matplotlib requests
```

---

## 💻 Cara Menjalankan Aplikasi

### 1. Visual GUI (Default Kasir Menu)
Menjalankan menu antarmuka visual berbasis OpenCV untuk kasir:
```bash
python main.py
```

### 2. Mode Recognition Direct
Langsung mengaktifkan kamera untuk pengenalan wajah:
```bash
python main.py --mode recognize
```
- Menampilkan bounding box wajah, label nama member, dan skor kemiripan.
- Nilai ambang batas (threshold) default cosine distance adalah `<= 0.40`. Di atas threshold ini akan terdeteksi sebagai `Unknown`.

### 3. Mode Register Member via CLI
Mendaftarkan member baru dengan parameter:
```bash
python main.py --mode register --name "Budi Santoso" --phone "081234567890" --discount 10
```

### 4. Mode Manajemen Data & Rebuild Embedding
- Buka antarmuka manajemen member:
  ```bash
  python main.py --mode manage
  ```
- Membangun ulang file embedding (`known_faces_embeddings.json`):
  ```bash
  python main.py --mode rebuild-embeddings
  ```

### 5. Memilih Kamera
Jika menggunakan kamera eksternal/USB:
```bash
python main.py --mode recognize --camera-index 1
```

---

## 📊 Evaluasi Performa Model

Repository ini menyertakan berbagai script pengujian metrik untuk pengujian ilmiah/skripsi/riset:

### 1. Evaluasi ORB vs DeepFace
Membandingkan keakuratan baseline ORB dengan DeepFace FaceNet512:
```bash
python evaluate_orb_vs_deepface.py
```

### 2. Full Benchmark (Accuracy, Precision, Recall, F1)
```bash
python evaluate_full_convergentai.py
```

### 3. Leave-One-Out (LOO) Cross Validation
```bash
python eval_deepface_loo.py
```

### 4. Download Dataset Benchmark Non-Member (LFW)
```bash
python download_lfw_unknown.py
```

---

## 🔗 Integrasi Backend API

Secara default, aplikasi akan mengirimkan payload HTTP POST ke server backend Laravel:
- **Register Face**: `http://127.0.0.1:8000/api/customers/register-face`
- **Detect Member**: `http://127.0.0.1:8000/api/customers/detect-member`

Custom URL API dapat disesuaikan via argumen:
```bash
python main.py --mode register --name "Nama Member" --api-url "http://192.168.1.100:8000/api/customers/register-face"
```

---

## 📜 Lisensi

Proyek ini dirilis di bawah lisensi [MIT](LICENSE).
