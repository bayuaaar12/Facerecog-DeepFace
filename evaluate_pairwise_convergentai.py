"""
=============================================================
SCRIPT EVALUASI PAIRWISE: ORB vs DeepFace (Facenet512)
Untuk paper: ConvergentAI 2026 submission
=============================================================
CARA PAKAI:
-----------
1. Taruh SEMUA foto yang mau dievaluasi di folder known_faces/
   dengan format nama file:  <label>_<nomor>.jpg
   Contoh: alif_1.jpg, alif_2.jpg, alvin_1.jpg, alvin_2.jpg, ...
   (label = nama orang, tanpa perlu jumlah foto yang sama per orang)

2. Install dependency (di komputer kamu sendiri):
   pip install opencv-python deepface numpy matplotlib

3. Taruh script ini SEJAJAR dengan folder known_faces/ lalu jalankan:
   python evaluate_pairwise_convergentai.py

4. Hasil akan muncul di terminal + disimpan ke:
   - hasil_evaluasi_pairwise.txt   (angka untuk paper, TP/FP/FN/TN + metrik)
   - bar_chart_pairwise.png        (Figure untuk paper)

=============================================================
DESAIN METODE: EXHAUSTIVE PAIRWISE COMPARISON
-------------------------------------------------------------
Sama seperti metodologi paper CODHES sebelumnya: SETIAP DUA FOTO
di known_faces/ dibandingkan sebagai template tunggal (1-vs-1),
BUKAN multi-kelas identifikasi terhadap seluruh database.

Untuk n foto total, jumlah pasangan = C(n, 2) = n*(n-1)/2.
Ground truth: pasangan dianggap "same person" jika label (nama
sebelum angka di akhir nama file) sama.

Catatan penting:
- Desain ini TIDAK menguji false-acceptance terhadap orang yang
  sama sekali di luar known_faces/ ("benar-benar asing"). Kalau
  kamu masih mau menguji itu, tambahkan folder terpisah berisi
  foto orang lain dan minta script tambahan.
- Karena hanya pakai known_faces/, template & query berasal dari
  kumpulan foto yang sama (persis seperti desain paper CODHES
  sebelumnya) -- konsisten secara metodologis dengan paper lama.
"""

import re
import itertools
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = BASE_DIR / "known_faces"

# ── Konstanta ORB (identik dengan evaluate_orb_vs_deepface.py) ──────────────
FACE_IMAGE_SIZE = (200, 200)
LOW_LIGHT_MEAN_LIMIT = 95
LOW_LIGHT_TARGET_MEAN = 125
LOW_LIGHT_BRIGHTNESS_BONUS = 18
ORB_DISTANCE_LIMIT = 60
ORB_RATIO_TEST = 0.75
MIN_GOOD_MATCHES = 18
MIN_MATCH_MARGIN = 8
MIN_MATCH_RATIO = 0.18
LOW_FEATURE_DESCRIPTOR_LIMIT = 90
LOW_FEATURE_MIN_GOOD_MATCHES = 12
LOW_FEATURE_MIN_MATCH_MARGIN = 5
LOW_FEATURE_MIN_MATCH_RATIO = 0.12
MIN_TEMPLATE_SIMILARITY = 62
MIN_TEMPLATE_MARGIN = 3
MIN_TEMPLATE_ORB_SUPPORT = 4

# ── Konstanta DeepFace (identik dengan main.py produksi) ────────────────────
FACENET_MODEL_NAME = "Facenet512"
COSINE_DISTANCE_THRESHOLD = 0.40

orb = cv2.ORB_create(nfeatures=700)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


def face_label_from_path(p):
    return re.sub(r"_\d+$", "", Path(p).stem)


def normalize_lighting(gray):
    mean_val = float(np.mean(gray))
    if mean_val < LOW_LIGHT_MEAN_LIMIT:
        alpha = min(2.0, LOW_LIGHT_TARGET_MEAN / max(mean_val, 1.0))
        gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=LOW_LIGHT_BRIGHTNESS_BONUS)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_face(face_roi):
    resized = cv2.resize(face_roi, FACE_IMAGE_SIZE)
    return normalize_lighting(resized)


def list_known_face_files():
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in sorted(KNOWN_FACES_DIR.iterdir()) if p.suffix.lower() in exts]


# ── ORB: bangun template dari satu foto ──────────────────────────────────
def orb_template_from_path(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = preprocess_face(img)
    kps, desc = orb.detectAndCompute(img, None)
    return {
        "name": face_label_from_path(path),
        "image": img,
        "descriptors": desc,
        "keypoint_count": len(kps),
    }


def count_good_matches(src_desc, known_desc):
    if src_desc is None or known_desc is None or len(known_desc) < 2:
        return 0
    knn = matcher.knnMatch(src_desc, known_desc, k=2)
    good = [
        m for m, n in knn
        if m.distance < ORB_DISTANCE_LIMIT and m.distance < ORB_RATIO_TEST * n.distance
    ]
    return len(good)


def template_similarity(src, known):
    corr = cv2.matchTemplate(src, known, cv2.TM_CCOEFF_NORMED)[0][0]
    abs_sim = 1.0 - (np.mean(cv2.absdiff(src, known)) / 255.0)
    return max(0.0, ((corr + 1.0) / 2.0) * 100.0, abs_sim * 100.0)


def orb_pair_match(query_path, template):
    """Bandingkan satu foto query terhadap SATU template (1-vs-1)."""
    img = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    proc = preprocess_face(img)
    kps, desc = orb.detectAndCompute(proc, None)
    desc_count = max(len(kps), 1)

    orb_s = count_good_matches(desc, template["descriptors"])
    tmpl_s = template_similarity(proc, template["image"])
    ratio = orb_s / desc_count

    req_good, req_ratio = MIN_GOOD_MATCHES, MIN_MATCH_RATIO
    if desc_count < LOW_FEATURE_DESCRIPTOR_LIMIT:
        req_good, req_ratio = LOW_FEATURE_MIN_GOOD_MATCHES, LOW_FEATURE_MIN_MATCH_RATIO

    # Catatan: MIN_MATCH_MARGIN & MIN_TEMPLATE_MARGIN dari main.py dipakai untuk
    # membedakan kandidat terbaik #1 vs #2 saat identifikasi multi-kelas.
    # Dalam pairwise 1-vs-1 tidak ada kandidat kedua, jadi margin diabaikan.
    orb_conf = orb_s >= req_good and ratio >= req_ratio
    tmpl_conf = tmpl_s >= MIN_TEMPLATE_SIMILARITY and orb_s >= MIN_TEMPLATE_ORB_SUPPORT

    return bool(orb_conf or tmpl_conf)


# ── DeepFace: embedding + cosine distance ────────────────────────────────
def create_deepface_embedding(image):
    from deepface import DeepFace

    result = DeepFace.represent(
        img_path=image,
        model_name=FACENET_MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False,
        align=False,
    )
    return np.asarray(result[0]["embedding"], dtype=np.float32)


def cosine_distance(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / norm)


def deepface_pair_match(embedding_a, embedding_b):
    return cosine_distance(embedding_a, embedding_b) < COSINE_DISTANCE_THRESHOLD


def compute_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def run_all():
    files = list_known_face_files()
    n = len(files)
    if n < 2:
        print("Butuh minimal 2 foto di known_faces/ untuk membuat pasangan.")
        return

    pairs = list(itertools.combinations(files, 2))
    total_pairs = len(pairs)

    labels = sorted({face_label_from_path(p) for p in files})
    print("=" * 60)
    print("  EVALUASI PAIRWISE — ORB vs DeepFace (ConvergentAI 2026)")
    print("=" * 60)
    print(f"Jumlah individu terdeteksi : {len(labels)}  -> {', '.join(labels)}")
    print(f"Total foto                : {n}")
    print(f"Total pasangan (C({n},2))  : {total_pairs}")
    if total_pairs > 20000:
        print("\n[PERINGATAN] Jumlah pasangan sangat besar, proses bisa lama.")

    print("\n[1/3] Menyiapkan template ORB untuk semua foto...")
    orb_templates = {p: orb_template_from_path(p) for p in files}

    print("[2/3] Menghitung embedding DeepFace (Facenet512) untuk semua foto...")
    print("      (Butuh waktu di awal untuk download model jika belum ada)")
    deep_embeddings = {}
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [SKIP] {p.name}: gambar tidak dapat dibaca")
            continue
        try:
            deep_embeddings[p] = create_deepface_embedding(img)
        except Exception as e:
            print(f"  [SKIP] {p.name}: {e}")

    orb_tp = orb_fp = orb_fn = orb_tn = 0
    deep_tp = deep_fp = deep_fn = deep_tn = 0

    print(f"\n[3/3] Menjalankan {total_pairs} perbandingan pasangan...")
    for idx, (path_a, path_b) in enumerate(pairs, start=1):
        label_a = face_label_from_path(path_a)
        label_b = face_label_from_path(path_b)
        same_person = label_a == label_b

        template_a = orb_templates.get(path_a)
        if template_a is not None and template_a["descriptors"] is not None:
            predicted_match = orb_pair_match(path_b, template_a)
            if same_person and predicted_match:
                orb_tp += 1
            elif not same_person and predicted_match:
                orb_fp += 1
            elif same_person and not predicted_match:
                orb_fn += 1
            else:
                orb_tn += 1

        emb_a = deep_embeddings.get(path_a)
        emb_b = deep_embeddings.get(path_b)
        if emb_a is not None and emb_b is not None:
            predicted_match = deepface_pair_match(emb_a, emb_b)
            if same_person and predicted_match:
                deep_tp += 1
            elif not same_person and predicted_match:
                deep_fp += 1
            elif same_person and not predicted_match:
                deep_fn += 1
            else:
                deep_tn += 1

        if idx % 200 == 0 or idx == total_pairs:
            print(f"  ...{idx}/{total_pairs} pasangan selesai")

    orb_metrics = compute_metrics(orb_tp, orb_fp, orb_fn, orb_tn)
    deep_metrics = compute_metrics(deep_tp, deep_fp, deep_fn, deep_tn)

    print("\n" + "=" * 60)
    print("  HASIL EVALUASI PAIRWISE")
    print("=" * 60)
    print(f"{'Metric':<12}{'ORB':>12}{'DeepFace':>14}")
    for key in ["TP", "FP", "FN", "TN"]:
        print(f"{key:<12}{orb_metrics[key]:>12}{deep_metrics[key]:>14}")
    for key in ["accuracy", "precision", "recall", "f1"]:
        print(f"{key:<12}{orb_metrics[key]*100:>11.1f}%{deep_metrics[key]*100:>13.1f}%")

    with open(BASE_DIR / "hasil_evaluasi_pairwise.txt", "w") as f:
        f.write("HASIL EVALUASI PAIRWISE -- ORB vs DeepFace (Facenet512)\n")
        f.write("ConvergentAI 2026 submission\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Jumlah individu: {len(labels)} ({', '.join(labels)})\n")
        f.write(f"Total foto: {n}, total pasangan: {total_pairs}\n\n")
        f.write(f"{'Metric':<12}{'ORB':>12}{'DeepFace':>14}\n")
        for key in ["TP", "FP", "FN", "TN"]:
            f.write(f"{key:<12}{orb_metrics[key]:>12}{deep_metrics[key]:>14}\n")
        for key in ["accuracy", "precision", "recall", "f1"]:
            f.write(f"{key:<12}{orb_metrics[key]*100:>11.1f}%{deep_metrics[key]*100:>13.1f}%\n")
    print("\nDisimpan ke hasil_evaluasi_pairwise.txt")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = ["accuracy", "precision", "recall", "f1"]
        orb_vals = [orb_metrics[m] * 100 for m in metrics]
        deep_vals = [deep_metrics[m] * 100 for m in metrics]
        x = np.arange(len(metrics))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        bars1 = ax.bar(x - width / 2, orb_vals, width, label="ORB (hybrid)", color="#4472C4", edgecolor="white")
        bars2 = ax.bar(x + width / 2, deep_vals, width, label="DeepFace (Facenet512)", color="#ED7D31", edgecolor="white")
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=11)
        ax.set_ylabel("Score (%)", fontsize=11)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=10)
        ax.set_title("ORB vs DeepFace (Facenet512) — Pairwise Evaluation", fontsize=12)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(BASE_DIR / "bar_chart_pairwise.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Bar chart disimpan ke bar_chart_pairwise.png")
    except ImportError:
        print("matplotlib tidak terinstall, skip chart. Jalankan: pip install matplotlib")

    print("\n✅ Selesai! Salin angka dari hasil_evaluasi_pairwise.txt ke tabel paper kamu.")


if __name__ == "__main__":
    if not KNOWN_FACES_DIR.exists():
        print(f"Folder tidak ditemukan: {KNOWN_FACES_DIR}")
        print("Taruh script ini sejajar dengan folder known_faces/ lalu jalankan ulang.")
    else:
        run_all()
