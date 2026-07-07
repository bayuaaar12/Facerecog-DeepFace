# Face Recognition Kasir Pingkal

Repository ini berisi aplikasi face recognition sederhana berbasis Python dan OpenCV untuk kebutuhan kasir/member. Aplikasi dapat:

- mendeteksi wajah dari kamera,
- mengenali wajah dari sampel lokal di folder `known_faces`,
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
pip install opencv-python numpy
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

Register customer lewat argumen CLI:

```bash
python main.py --mode register --name "Nama Customer" --phone "08123456789" --discount 10
```

Buka menu manajemen data wajah:

```bash
python main.py --mode manage
```

Jika kamera default tidak sesuai, gunakan opsi `--camera-index`:

```bash
python main.py --mode recognize --camera-index 1
```

## Integrasi API

Secara default aplikasi memakai endpoint lokal:

- register face: `http://127.0.0.1:8000/api/customers/register-face`
- detect member: `http://127.0.0.1:8000/api/customers/detect-member`

Endpoint register dapat diganti dengan opsi:

```bash
python main.py --mode register --name "Nama Customer" --api-url "http://127.0.0.1:8000/api/customers/register-face"
```

## Catatan Data Wajah

Sampel wajah yang dibuat aplikasi akan tersimpan di `known_faces/`. File gambar di folder tersebut tidak dilacak Git karena termasuk data sensitif/biometrik.

Untuk publikasi ke Zenodo atau GitHub, pastikan hanya kode, konfigurasi, dan file non-sensitif yang diunggah.

## License

Kode ini dirilis dengan lisensi MIT. Lihat file `LICENSE`.
