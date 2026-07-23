# Face Recognition Kasir Pingkal

Repository ini berisi aplikasi face recognition sederhana berbasis Python dan OpenCV untuk kebutuhan kasir/member. Aplikasi dapat:

- mendeteksi wajah dari kamera,
- mengenali wajah dengan embedding DeepFace **FaceNet512** dan cosine distance,
- mendaftarkan wajah customer baru,
- mengirim data register/deteksi ke API Laravel lokal,
- melihat dan menghapus data wajah yang tersimpan lokal.

Data wajah tidak disertakan di repository. Folder `known_faces` sengaja hanya menyimpan `.gitkeep`, dan isi folder tersebut diabaikan oleh Git agar sampel wajah tidak ikut ter-upload.

## Requirements

- Python 3.9 atau lebih baru
- Kamera/webcam
- Paket Python:
  - `opencv-python`
  - `numpy`
  - `deepface`
  - `tensorflow`

## Setup

Clone repository, lalu masuk ke folder project:

```bash
git clone https://github.com/bayuaaar12/facerecog.git
cd facerecog
```

Buat dan aktifkan virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependency:

```bash
pip install deepface tensorflow opencv-python numpy
```

## Cara Menjalankan

Jalankan menu visual:

```bash
python main.py
```

Jalankan mode pengenalan wajah langsung:

```bash
python main.py --mode recognize
```

Recognition memakai Haar Cascade untuk mencari/crop wajah, lalu `DeepFace.represent()` dengan model `Facenet512`. Setiap embedding kamera dibandingkan dengan seluruh sampel customer memakai cosine distance. Customer dengan jarak terkecil akan dipilih bila nilainya `<= 0.40`; selain itu hasilnya `Unknown`.

Register customer lewat argumen CLI:

```bash
python main.py --mode register --name "Nama Customer" --phone "08123456789" --discount 10
```

Buka menu manajemen data wajah:

```bash
python main.py --mode manage
```

## Embedding FaceNet512

Saat register, aplikasi menyimpan beberapa crop wajah berwarna di `known_faces/` dan otomatis membangun ulang `known_faces_embeddings.json`. File tersebut menyimpan embedding FaceNet512 serta label customer untuk setiap sampel. Bila data wajah dihapus, index embedding juga otomatis diperbarui.

Jika file embedding belum ada, rusak, atau ada foto yang baru ditambahkan secara manual, aplikasi akan membangunnya kembali saat recognition dimulai. Anda juga dapat menjalankannya sendiri:

```bash
python main.py --mode rebuild-embeddings
```

Pastikan koneksi internet tersedia pada penggunaan pertama bila DeepFace perlu mengunduh bobot FaceNet512. Jika muncul pesan DeepFace belum terpasang, jalankan kembali perintah instalasi di atas.

Jika kamera default tidak sesuai, gunakan opsi `--camera-index`:

```bash
python main.py --mode recognize --camera-index 1
```

## Evaluasi ORB vs DeepFace

Gunakan `evaluate_orb_vs_deepface.py` untuk membandingkan baseline ORB dengan
DeepFace FaceNet512. Kedua metode menggunakan **foto registrasi yang sama** dari
`known_faces/` (folder yang juga dipakai `main.py`), sehingga tidak perlu membuat
salinan data member untuk masing-masing metode.

Siapkan foto uji yang berbeda dari foto registrasi:

```text
dataset/
├── test/
│   ├── normal/
│   ├── low_light/
│   └── rotated/
└── unknown/       # opsional: wajah non-member
```

Nama foto uji harus sesuai label registrasi, misalnya `bayu_anugrah_1.jpg`
untuk member `bayu_anugrah`. Lalu jalankan:

```bash
pip install matplotlib
python evaluate_orb_vs_deepface.py
```

Hasil metrik Accuracy, Precision, Recall, dan F1 akan disimpan pada
`hasil_evaluasi.txt`, serta grafik pada `bar_chart_fig5.png`. Untuk lokasi data
yang berbeda, gunakan `--known-faces-dir`, `--test-dir`, atau `--unknown-dir`.

## Integrasi API

Secara default aplikasi memakai endpoint lokal:

- register face: `http://127.0.0.1:8000/api/customers/register-face`
- detect member: `http://127.0.0.1:8000/api/customers/detect-member`

Endpoint register dapat diganti dengan opsi:

```bash
python main.py --mode register --name "Nama Customer" --api-url "http://127.0.0.1:8000/api/customers/register-face"
```

## Catatan Data Wajah

Sampel wajah yang dibuat aplikasi akan tersimpan di `known_faces/`, sedangkan embedding lokal tersimpan di `known_faces_embeddings.json`. Keduanya adalah data sensitif/biometrik dan sebaiknya tidak diunggah.

Untuk publikasi ke Zenodo atau GitHub, pastikan hanya kode, konfigurasi, dan file non-sensitif yang diunggah.

## License

Kode ini dirilis dengan lisensi MIT. Lihat file `LICENSE`.
