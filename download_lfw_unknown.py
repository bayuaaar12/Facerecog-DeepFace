"""
=============================================================
STEP 0 (sebelum evaluasi): Download foto orang ASING dari LFW
=============================================================
Tujuan: menyiapkan foto "unknown" (orang yang BUKAN member
Warung Pingkal) untuk menguji OPEN-SET FALSE ACCEPTANCE --
apakah sistem salah mengenali orang asing sebagai salah satu
member terdaftar.

LFW (Labeled Faces in the Wild) adalah dataset wajah publik yang
umum dipakai untuk riset akademik (boleh dikutip di paper).

CARA PAKAI:
-----------
1. pip install scikit-learn pillow numpy
2. Jalankan (butuh internet, download ~200MB sekali saja, lalu
   di-cache otomatis oleh scikit-learn untuk run berikutnya):

   python download_lfw_unknown.py --num-people 15

3. Foto akan tersimpan di: dataset/unknown/lfw_<nama>_<idx>.jpg
   Jalankan script ini SEJAJAR dengan folder known_faces/ kamu
   (di folder proyek facedetec yang sama).

4. PENTING: cek manual sekilas ke folder dataset/unknown/ untuk
   pastikan tidak ada nama yang kebetulan sama dengan member kamu
   (alif, alvin, dst.) -- kalau ada, hapus saja fotonya.
=============================================================
"""
import argparse
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
UNKNOWN_DIR = BASE_DIR / "dataset" / "unknown"


def main(num_people: int, seed: int):
    from sklearn.datasets import fetch_lfw_people
    from PIL import Image

    print("Mengunduh/memuat dataset LFW (bisa makan waktu di percobaan pertama)...")
    lfw = fetch_lfw_people(color=True, min_faces_per_person=1, resize=1.0)
    images = lfw.images  # (n_samples, h, w, 3) float 0..255
    names = lfw.target_names[lfw.target]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), size=min(num_people, len(images)), replace=False)

    UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i in idx:
        img_arr = images[i].astype(np.uint8)
        img = Image.fromarray(img_arr)
        label = str(names[i]).replace(" ", "_").lower()
        out_path = UNKNOWN_DIR / f"lfw_{label}_{i}.jpg"
        img.save(out_path)
        saved += 1

    print(f"Selesai. {saved} foto orang asing (LFW) disimpan ke: {UNKNOWN_DIR}")
    print("Cek sekilas foldernya, lalu lanjut jalankan evaluate_full_convergentai.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download foto acak LFW untuk uji open-set false acceptance.")
    parser.add_argument("--num-people", type=int, default=15, help="Jumlah foto unknown yang diambil (default 15)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.num_people, args.seed)
