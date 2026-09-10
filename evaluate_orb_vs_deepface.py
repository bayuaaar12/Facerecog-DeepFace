"""
=============================================================
 SCRIPT EVALUASI: ORB vs DeepFace (Facenet512)
 Untuk paper: AI-Powered Customer Recognition in Culinary MSMEs
 CoDHES 2026
=============================================================

CARA PAKAI:
-----------
1. Siapkan folder dataset seperti ini:

   known_faces/              ← foto registrasi dari main.py (dipakai DUA metode)
   ├── bayu_anugrah_1.jpg
   ├── bayu_anugrah_2.jpg
   └── ...

   dataset/
   ├── test/                 ← foto uji TERPISAH, jangan menyalin foto registrasi
   │   ├── normal/           ← foto test kondisi normal
   │   │   ├── bayu_anugrah.jpg   (nama file = face_label)
   │   │   ├── nama_member2.jpg
   │   │   └── ...
   │   ├── low_light/        ← foto test cahaya redup
   │   │   ├── bayu_anugrah.jpg
   │   │   └── ...
   │   └── rotated/          ← foto test sudut 30 derajat
   │       ├── bayu_anugrah.jpg
   │       └── ...
   └── unknown/              ← foto orang BUKAN member (untuk test false acceptance)
       ├── orang1.jpg
       └── ...

2. Install dependency:
   pip install opencv-python deepface numpy matplotlib

3. Jalankan:
   python evaluate_orb_vs_deepface.py

4. Hasil akan muncul di terminal + disimpan ke:
   - hasil_evaluasi.txt   (angka untuk paper)
   - bar_chart.png        (Fig. 5 untuk paper)

=============================================================
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import argparse
import re
import cv2
import numpy as np
from pathlib import Path

# ── Konfigurasi path ─────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
# Gunakan folder yang sama dengan aplikasi utama (main.py). Dengan begitu
# template/registrasi yang diuji oleh ORB dan DeepFace selalu identik.
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
TEST_DIR        = BASE_DIR / "dataset" / "test"
UNKNOWN_DIR     = BASE_DIR / "dataset" / "unknown"
CASCADE_PATH    = BASE_DIR / "face_ref.xml"   # pakai punya kamu

# ── Konstanta ORB untuk baseline pembanding ──────────────────────────────────
FACE_IMAGE_SIZE             = (200, 200)
LOW_LIGHT_MEAN_LIMIT        = 95
LOW_LIGHT_TARGET_MEAN       = 125
LOW_LIGHT_BRIGHTNESS_BONUS  = 18
ORB_DISTANCE_LIMIT          = 60
ORB_RATIO_TEST              = 0.75
MIN_GOOD_MATCHES            = 18
MIN_MATCH_MARGIN            = 8
MIN_MATCH_RATIO             = 0.18
LOW_FEATURE_DESCRIPTOR_LIMIT = 90
LOW_FEATURE_MIN_GOOD_MATCHES = 12
LOW_FEATURE_MIN_MATCH_MARGIN = 5
LOW_FEATURE_MIN_MATCH_RATIO  = 0.12
MIN_TEMPLATE_SIMILARITY     = 62
MIN_TEMPLATE_MARGIN         = 3
MIN_TEMPLATE_ORB_SUPPORT    = 4

CONDITIONS = ["normal", "low_light", "rotated"]

# =============================================================================
# PREPROCESSING ORB
# =============================================================================

def normalize_lighting(gray):
    mean_val = float(np.mean(gray))
    if mean_val < LOW_LIGHT_MEAN_LIMIT:
        alpha = min(2.0, LOW_LIGHT_TARGET_MEAN / max(mean_val, 1.0))
        gray  = cv2.convertScaleAbs(gray, alpha=alpha, beta=LOW_LIGHT_BRIGHTNESS_BONUS)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def preprocess_face(face_roi):
    resized = cv2.resize(face_roi, FACE_IMAGE_SIZE)
    return normalize_lighting(resized)

def load_image(path):
    """Baca citra BGR sekali; kedua metode menerima citra yang sama."""
    return cv2.imread(str(path), cv2.IMREAD_COLOR)

# =============================================================================
# ORB ENGINE (baseline pembanding)
# =============================================================================

orb     = cv2.ORB_create(nfeatures=700)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

def face_label_from_path(p):
    return re.sub(r"_\d+$", "", Path(p).stem)

def load_known_faces_orb():
    faces = []
    for p in sorted(KNOWN_FACES_DIR.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img  = preprocess_face(img)
        kps, desc = orb.detectAndCompute(img, None)
        faces.append({
            "name": face_label_from_path(p),
            "image": img,
            "descriptors": desc,
            "keypoint_count": len(kps),
        })
    return faces

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
    corr      = cv2.matchTemplate(src, known, cv2.TM_CCOEFF_NORMED)[0][0]
    abs_sim   = 1.0 - (np.mean(cv2.absdiff(src, known)) / 255.0)
    return max(0.0, ((corr + 1.0) / 2.0) * 100.0, abs_sim * 100.0)

def recognize_orb(face_roi, known_faces):
    proc   = preprocess_face(face_roi)
    kps, desc = orb.detectAndCompute(proc, None)
    if not known_faces:
        return "Unknown", 0

    desc_count   = max(len(kps), 1)
    scores       = {}
    for kf in known_faces:
        orb_s  = count_good_matches(desc, kf["descriptors"])
        tmpl_s = template_similarity(proc, kf["image"])
        comb   = orb_s + int(max(0, tmpl_s - 50) * 0.35)
        name   = kf["name"]
        if comb > scores.get(name, {}).get("combined", 0):
            scores[name] = {"combined": comb, "orb": orb_s, "template": tmpl_s}

    best = max(scores.items(), key=lambda x: x[1]["combined"], default=(None, {}))
    rest = [v["combined"] for k, v in scores.items() if k != best[0]]
    second_comb   = max(rest, default=0)
    second_tmpl   = max(
        (v["template"] for k, v in scores.items() if k != best[0]), default=0
    )

    best_name   = best[0] or "Unknown"
    best_val    = best[1]
    orb_score   = best_val.get("orb", 0)
    tmpl_score  = best_val.get("template", 0)
    comb_score  = best_val.get("combined", 0)
    ratio       = orb_score / desc_count

    req_good    = MIN_GOOD_MATCHES
    req_margin  = MIN_MATCH_MARGIN
    req_ratio   = MIN_MATCH_RATIO
    if desc_count < LOW_FEATURE_DESCRIPTOR_LIMIT:
        req_good   = LOW_FEATURE_MIN_GOOD_MATCHES
        req_margin = LOW_FEATURE_MIN_MATCH_MARGIN
        req_ratio  = LOW_FEATURE_MIN_MATCH_RATIO

    orb_conf  = (orb_score >= req_good
                and comb_score - second_comb >= req_margin
                and ratio >= req_ratio)
    tmpl_conf = (tmpl_score >= MIN_TEMPLATE_SIMILARITY
                and tmpl_score - second_tmpl >= MIN_TEMPLATE_MARGIN
                and orb_score >= MIN_TEMPLATE_ORB_SUPPORT)

    if not orb_conf and not tmpl_conf:
        return "Unknown", comb_score
    return best_name, comb_score

# =============================================================================
# DEEPFACE ENGINE
# =============================================================================

def create_deepface_embedding(image):
    """
    # [AI/ML INFERENCE]
    Konfigurasi ini sama dengan create_embedding() di main.py (Model Facenet512 512D).
    """
    from deepface import DeepFace
    result = DeepFace.represent(
        img_path=image,
        model_name="Facenet512",
        detector_backend="skip",
        enforce_detection=False,
        align=False,
    )
    return np.asarray(result[0]["embedding"], dtype=np.float32)


def load_known_faces_deepface():
    """Load dan embed semua known faces pakai Facenet512."""
    faces = []
    for p in sorted(KNOWN_FACES_DIR.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            image = load_image(p)
            if image is None:
                raise ValueError("gambar tidak dapat dibaca")
            emb = create_deepface_embedding(image)
            faces.append({"name": face_label_from_path(p), "embedding": emb})
        except Exception as e:
            print(f"  [SKIP] {p.name}: {e}")
    return faces

def cosine_distance(a, b):
    """
    # [AI/ML INFERENCE]
    Kalkulasi Cosine Distance antar vektor embedding.
    """
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / norm)

def recognize_deepface(face_roi, known_faces, threshold=0.40):
    """
    # [AI/ML INFERENCE]
    Pengenalan DeepFace Facenet512.
    Threshold default 0.40 mengacu pada rekomendasi DeepFace dan terbukti optimal pada hasil_evaluasi.txt.
    """
    try:
        emb = create_deepface_embedding(face_roi)
    except Exception:
        return "Unknown", 1.0

    best_name = "Unknown"
    best_dist = threshold
    for kf in known_faces:
        d = cosine_distance(emb, kf["embedding"])
        if d < best_dist:
            best_dist = d
            best_name = kf["name"]
    return best_name, best_dist

# =============================================================================
# EVALUASI UTAMA
# =============================================================================

def evaluate(recognize_fn, known_faces, condition):
    """
    Jalankan recognition ke semua foto di dataset/test/<condition>/
    Nama file (tanpa ekstensi, tanpa angka di belakang) = ground truth label.
    Contoh: bayu_anugrah.jpg  → label = bayu_anugrah
            bayu_anugrah_2.jpg → label = bayu_anugrah
    """
    folder = TEST_DIR / condition
    if not folder.exists():
        print(f"  [SKIP] Folder tidak ada: {folder}")
        return None

    TP = FP = FN = TN = 0
    results = []

    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        gt_label = face_label_from_path(p)
        image = load_image(p)
        if image is None:
            continue

        # ORB membutuhkan grayscale, sedangkan DeepFace memakai BGR. Keduanya
        # berasal dari file/sampel uji yang sama; tidak ada data khusus metode.
        method_input = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if recognize_fn is recognize_orb else image
        pred_label, score = recognize_fn(method_input, known_faces)
        correct = (pred_label == gt_label)

        if correct and pred_label != "Unknown":
            TP += 1
        elif not correct and pred_label != "Unknown":
            FP += 1
        elif not correct and pred_label == "Unknown":
            FN += 1
        else:
            TN += 1

        results.append({
            "file": p.name, "gt": gt_label, "pred": pred_label,
            "score": score, "correct": correct
        })

    # Unknown test
    if UNKNOWN_DIR.exists():
        for p in sorted(UNKNOWN_DIR.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = load_image(p)
            if image is None:
                continue
            method_input = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if recognize_fn is recognize_orb else image
            pred_label, score = recognize_fn(method_input, known_faces)
            if pred_label == "Unknown":
                TN += 1
            else:
                FP += 1

    total   = TP + FP + FN + TN
    acc     = (TP + TN) / total  if total > 0 else 0
    prec    = TP / (TP + FP)     if (TP + FP) > 0 else 0
    rec     = TP / (TP + FN)     if (TP + FN) > 0 else 0
    f1      = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "condition": condition,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "details": results
    }

def run_all():
    print("=" * 60)
    print("  EVALUASI ORB vs DeepFace — Warung Pingkal")
    print("=" * 60)

    # ── Load known faces ──────────────────────────────────────────────────
    print("\n[1/4] Loading known faces (ORB)...")
    known_orb = load_known_faces_orb()
    print(f"      {len(known_orb)} sampel dari {len(set(f['name'] for f in known_orb))} member")

    print("[2/4] Loading known faces (DeepFace / Facenet512)...")
    print("      Ini butuh beberapa menit pertama kali karena download model...")
    known_deepface = load_known_faces_deepface()
    print(f"      {len(known_deepface)} embedding berhasil dibuat")

    # ── Evaluasi per kondisi ──────────────────────────────────────────────
    print("\n[3/4] Menjalankan evaluasi...")
    orb_results  = {}
    deep_results = {}

    for cond in CONDITIONS:
        print(f"\n  Kondisi: {cond}")

        print("    ORB  ...", end=" ", flush=True)
        r = evaluate(recognize_orb, known_orb, cond)
        orb_results[cond] = r
        if r:
            print(f"Acc={r['accuracy']:.1%}  P={r['precision']:.1%}  R={r['recall']:.1%}  F1={r['f1']:.1%}")

        print("    Deep ...", end=" ", flush=True)
        r = evaluate(recognize_deepface, known_deepface, cond)
        deep_results[cond] = r
        if r:
            print(f"Acc={r['accuracy']:.1%}  P={r['precision']:.1%}  R={r['recall']:.1%}  F1={r['f1']:.1%}")

    # ── Hitung overall ────────────────────────────────────────────────────
    def overall(res_dict):
        vals = [v for v in res_dict.values() if v]
        if not vals:
            return {}
        return {
            "accuracy":  np.mean([v["accuracy"]  for v in vals]),
            "precision": np.mean([v["precision"] for v in vals]),
            "recall":    np.mean([v["recall"]    for v in vals]),
            "f1":        np.mean([v["f1"]        for v in vals]),
        }

    orb_overall  = overall(orb_results)
    deep_overall = overall(deep_results)

    # ── Print hasil ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HASIL KESELURUHAN (rata-rata 3 kondisi)")
    print("=" * 60)
    print(f"{'Metric':<18} {'ORB':>10} {'DeepFace':>12} {'Delta':>10}")
    print("-" * 52)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        o = orb_overall.get(metric, 0)
        d = deep_overall.get(metric, 0)
        print(f"  {metric.capitalize():<16} {o:>9.1%} {d:>11.1%} {(d-o):>+10.1%}")

    print("\n  PER KONDISI — Accuracy:")
    print(f"  {'Kondisi':<20} {'ORB':>10} {'DeepFace':>12}")
    print("  " + "-" * 44)
    for cond in CONDITIONS:
        o = orb_results.get(cond, {})
        d = deep_results.get(cond, {})
        oa = o.get("accuracy", 0) if o else 0
        da = d.get("accuracy", 0) if d else 0
        print(f"  {cond:<20} {oa:>9.1%} {da:>11.1%}")

    # ── Simpan hasil ke file ──────────────────────────────────────────────
    print("\n[4/4] Menyimpan hasil...")
    output_path = BASE_DIR / "hasil_evaluasi.txt"
    with open(output_path, "w") as f:
        f.write("HASIL EVALUASI ORB vs DeepFace (Facenet512)\n")
        f.write("Warung Pingkal MSME Cashier System\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Metric':<18} {'ORB':>10} {'DeepFace':>12} {'Delta':>10}\n")
        f.write("-" * 52 + "\n")
        for metric in ["accuracy", "precision", "recall", "f1"]:
            o = orb_overall.get(metric, 0)
            d = deep_overall.get(metric, 0)
            f.write(f"  {metric.capitalize():<16} {o:>9.1%} {d:>11.1%} {(d-o):>+10.1%}\n")
        f.write("\nPER KONDISI — Accuracy:\n")
        for cond in CONDITIONS:
            o = orb_results.get(cond, {})
            d = deep_results.get(cond, {})
            oa = o.get("accuracy", 0) if o else 0
            da = d.get("accuracy", 0) if d else 0
            f.write(f"  {cond:<20} {oa:>9.1%} {da:>11.1%}\n")
    print(f"  Tersimpan: {output_path}")

    # ── Buat bar chart (Fig. 5) ───────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams["font.family"] = "DejaVu Sans"

        metrics    = ["Accuracy", "Precision", "Recall", "F1-Score"]
        orb_vals   = [orb_overall.get(m.lower().replace("-", ""), 0) * 100
                      for m in ["Accuracy", "Precision", "Recall", "f1"]]
        deep_vals  = [deep_overall.get(m.lower().replace("-", ""), 0) * 100
                      for m in ["Accuracy", "Precision", "Recall", "f1"]]

        x      = np.arange(len(metrics))
        width  = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))

        bars1 = ax.bar(x - width/2, orb_vals,  width, label="ORB (hybrid)",
                       color="#4472C4", edgecolor="white")
        bars2 = ax.bar(x + width/2, deep_vals, width, label="DeepFace (Facenet512)",
                       color="#ED7D31", edgecolor="white")

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("Score (%)", fontsize=11)
        ax.set_title("Recognition Performance: ORB vs. DeepFace (Facenet512)", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        plt.tight_layout()

        chart_path = BASE_DIR / "bar_chart_fig5.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Bar chart: {chart_path}")
        print("  → Ini adalah Fig. 5 untuk paper kamu!")
    except ImportError:
        print("  matplotlib tidak terinstall, skip chart. Jalankan: pip install matplotlib")

    print("\n✅ Selesai! Salin angka dari hasil_evaluasi.txt ke tabel paper kamu.")
    print("   bar_chart_fig5.png = Fig. 5 siap paste ke Word.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bandingkan ORB dan DeepFace menggunakan data registrasi yang sama."
    )
    parser.add_argument(
        "--known-faces-dir",
        type=Path,
        default=KNOWN_FACES_DIR,
        help="Folder data registrasi bersama (default: ./known_faces).",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=TEST_DIR,
        help="Folder data uji berisi normal/, low_light/, dan rotated/.",
    )
    parser.add_argument(
        "--unknown-dir",
        type=Path,
        default=UNKNOWN_DIR,
        help="Folder wajah non-member untuk mengukur false acceptance.",
    )
    args = parser.parse_args()
    KNOWN_FACES_DIR = args.known_faces_dir.resolve()
    TEST_DIR = args.test_dir.resolve()
    UNKNOWN_DIR = args.unknown_dir.resolve()

    # Cek ketersediaan folder
    missing = []
    if not KNOWN_FACES_DIR.exists():
        missing.append("known_faces/  ← daftar wajah hasil registrasi dari main.py")
    if not TEST_DIR.exists():
        missing.append(f"dataset/test/normal/, dataset/test/low_light/, dataset/test/rotated/  ← foto test kamu")
    if not CASCADE_PATH.exists():
        missing.append(f"face_ref.xml  ← copy dari folder proyek kamu")

    if missing:
        print("\n⚠️  Folder/file yang belum ada:")
        for m in missing:
            print(f"   - {m}")
        print("\nSiapkan dulu sesuai petunjuk di atas, lalu jalankan ulang.")
    else:
        run_all()
