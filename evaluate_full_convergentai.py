"""
=============================================================
SCRIPT EVALUASI LENGKAP: ORB vs DeepFace (Facenet512)
Untuk paper: ConvergentAI 2026 submission
=============================================================
Menggabungkan DUA jenis pengujian:

1) CLOSED-SET (exhaustive pairwise, sama seperti paper CODHES):
   Setiap 2 foto DI ANTARA member terdaftar (known_faces/)
   dibandingkan 1-vs-1. Ini juga otomatis menangkap kasus
   "member A ketuker jadi member B".

2) OPEN-SET (false acceptance terhadap orang asing):
   Foto-foto di dataset/unknown/ (bukan member manapun, misalnya
   dari LFW -- lihat download_lfw_unknown.py) diuji terhadap
   SELURUH database known_faces/ sekaligus (multi-class, seperti
   sistem aslinya di main.py). Ground truth-nya selalu "Unknown".

CARA PAKAI:
-----------
1. (Opsional tapi direkomendasikan) Jalankan dulu:
     python download_lfw_unknown.py --num-people 15
   supaya folder dataset/unknown/ terisi foto orang asing.

2. Install dependency:
     pip install opencv-python deepface numpy matplotlib

3. Taruh script ini sejajar dengan known_faces/ dan dataset/, lalu:
     python evaluate_full_convergentai.py

4. Hasil disimpan ke:
   - hasil_evaluasi_full.txt   (breakdown closed-set, open-set, gabungan)
   - bar_chart_full.png        (figure untuk paper)

Catatan: kalau folder dataset/unknown/ kosong/tidak ada, script tetap
jalan -- hanya bagian open-set yang dilewati (dan itu perlu disebut
sebagai keterbatasan/limitation di paper, bukan diklaim sebagai
"false acceptance test" penuh).
=============================================================
"""

import sys
import io
import re
import itertools
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
UNKNOWN_DIR = BASE_DIR / "dataset" / "unknown"

# ── Konstanta ORB (identik dengan evaluate_orb_vs_deepface.py / main.py) ───
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

# ── Konstanta DeepFace (identik dengan main.py produksi) ────────────
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


def load_image(path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def list_image_files(folder):
    exts = {".jpg", ".jpeg", ".png"}
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]


# =============================================================================
# ORB ENGINE
# =============================================================================
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


def load_known_faces_orb():
    """Bangun daftar template ORB untuk SEMUA foto di known_faces/."""
    faces = []
    for p in list_image_files(KNOWN_FACES_DIR):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = preprocess_face(img)
        kps, desc = orb.detectAndCompute(img, None)
        faces.append({
            "path": p,
            "name": face_label_from_path(p),
            "image": img,
            "descriptors": desc,
            "keypoint_count": len(kps),
        })
    return faces


def orb_pair_match(query_path, template):
    """1-vs-1: query dibandingkan terhadap SATU template saja (untuk closed-set pairwise)."""
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

    orb_conf = orb_s >= req_good and ratio >= req_ratio
    tmpl_conf = tmpl_s >= MIN_TEMPLATE_SIMILARITY and orb_s >= MIN_TEMPLATE_ORB_SUPPORT
    return bool(orb_conf or tmpl_conf)


def recognize_orb(face_roi, known_faces):
    """Multi-class: query dibandingkan terhadap SELURUH database (untuk open-set).
    Identik dengan logika recognize_orb() di evaluate_orb_vs_deepface.py / main.py."""
    proc = preprocess_face(face_roi)
    kps, desc = orb.detectAndCompute(proc, None)
    if not known_faces:
        return "Unknown", 0
    desc_count = max(len(kps), 1)
    scores = {}
    for kf in known_faces:
        orb_s = count_good_matches(desc, kf["descriptors"])
        tmpl_s = template_similarity(proc, kf["image"])
        comb = orb_s + int(max(0, tmpl_s - 50) * 0.35)
        name = kf["name"]
        if comb > scores.get(name, {}).get("combined", 0):
            scores[name] = {"combined": comb, "orb": orb_s, "template": tmpl_s}
    best = max(scores.items(), key=lambda x: x[1]["combined"], default=(None, {}))
    rest = [v["combined"] for k, v in scores.items() if k != best[0]]
    second_comb = max(rest, default=0)
    second_tmpl = max((v["template"] for k, v in scores.items() if k != best[0]), default=0)
    best_name = best[0] or "Unknown"
    best_val = best[1]
    orb_score = best_val.get("orb", 0)
    tmpl_score = best_val.get("template", 0)
    comb_score = best_val.get("combined", 0)
    ratio = orb_score / desc_count
    req_good, req_margin, req_ratio = MIN_GOOD_MATCHES, MIN_MATCH_MARGIN, MIN_MATCH_RATIO
    if desc_count < LOW_FEATURE_DESCRIPTOR_LIMIT:
        req_good, req_margin, req_ratio = (
            LOW_FEATURE_MIN_GOOD_MATCHES,
            LOW_FEATURE_MIN_MATCH_MARGIN,
            LOW_FEATURE_MIN_MATCH_RATIO,
        )
    orb_conf = orb_score >= req_good and comb_score - second_comb >= req_margin and ratio >= req_ratio
    tmpl_conf = (
        tmpl_score >= MIN_TEMPLATE_SIMILARITY
        and tmpl_score - second_tmpl >= MIN_TEMPLATE_MARGIN
        and orb_score >= MIN_TEMPLATE_ORB_SUPPORT
    )
    if not orb_conf and not tmpl_conf:
        return "Unknown", comb_score
    return best_name, comb_score


# =============================================================================
# DEEPFACE ENGINE
# =============================================================================
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


def load_known_faces_deepface():
    faces = []
    for p in list_image_files(KNOWN_FACES_DIR):
        try:
            image = load_image(p)
            if image is None:
                raise ValueError("gambar tidak dapat dibaca")
            emb = create_deepface_embedding(image)
            faces.append({"path": p, "name": face_label_from_path(p), "embedding": emb})
        except Exception as e:
            print(f"  [SKIP] {p.name}: {e}")
    return faces


def deepface_pair_match(embedding_a, embedding_b):
    return cosine_distance(embedding_a, embedding_b) < COSINE_DISTANCE_THRESHOLD


def recognize_deepface(face_roi, known_faces, threshold=COSINE_DISTANCE_THRESHOLD):
    """Multi-class: query dibandingkan terhadap SELURUH database (untuk open-set)."""
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
# METRIK
# =============================================================================
def compute_metrics(TP=0, FP=0, FN=0, TN=0, tp=None, fp=None, fn=None, tn=None):
    if tp is not None: TP = tp
    if fp is not None: FP = fp
    if fn is not None: FN = fn
    if tn is not None: TN = tn
    total = TP + FP + FN + TN
    acc = (TP + TN) / total if total else 0
    prec = TP / (TP + FP) if (TP + FP) else 0
    rec = TP / (TP + FN) if (TP + FN) else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def add_confusion(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in ("TP", "FP", "FN", "TN")}


# =============================================================================
# CLOSED-SET: exhaustive pairwise di dalam known_faces/
# =============================================================================
def run_closed_set(known_orb, known_deepface):
    files = [kf["path"] for kf in known_orb]
    n = len(files)
    pairs = list(itertools.combinations(range(n), 2))
    total_pairs = len(pairs)
    print(f"  Total foto: {n}  ->  total pasangan (C({n},2)): {total_pairs}")

    orb_by_path = {kf["path"]: kf for kf in known_orb}
    deep_by_path = {kf["path"]: kf for kf in known_deepface}

    orb_conf = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    deep_conf = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    for idx, (i, j) in enumerate(pairs, start=1):
        path_a, path_b = files[i], files[j]
        same_person = face_label_from_path(path_a) == face_label_from_path(path_b)

        template_a = orb_by_path.get(path_a)
        if template_a is not None and template_a["descriptors"] is not None:
            pred_match = orb_pair_match(path_b, template_a)
            key = ("TP" if pred_match else "FN") if same_person else ("FP" if pred_match else "TN")
            orb_conf[key] += 1

        emb_a = deep_by_path.get(path_a, {}).get("embedding")
        emb_b = deep_by_path.get(path_b, {}).get("embedding")
        if emb_a is not None and emb_b is not None:
            pred_match = deepface_pair_match(emb_a, emb_b)
            key = ("TP" if pred_match else "FN") if same_person else ("FP" if pred_match else "TN")
            deep_conf[key] += 1

        if idx % 500 == 0 or idx == total_pairs:
            print(f"    ...{idx}/{total_pairs} pasangan selesai")

    return orb_conf, deep_conf


# =============================================================================
# OPEN-SET: dataset/unknown/ diuji terhadap SELURUH database
# =============================================================================
def run_open_set(known_orb, known_deepface):
    unknown_files = list_image_files(UNKNOWN_DIR)
    if not unknown_files:
        print("  [DILEWATI] dataset/unknown/ kosong atau tidak ada.")
        return None, None, 0

    orb_conf = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    deep_conf = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    for idx, p in enumerate(unknown_files, start=1):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        bgr = load_image(p)
        if gray is not None:
            pred_name, _ = recognize_orb(gray, known_orb)
            orb_conf["TN" if pred_name == "Unknown" else "FP"] += 1
        if bgr is not None:
            pred_name, _ = recognize_deepface(bgr, known_deepface)
            deep_conf["TN" if pred_name == "Unknown" else "FP"] += 1

        if idx % 20 == 0 or idx == len(unknown_files):
            print(f"    ...{idx}/{len(unknown_files)} foto unknown selesai")

    return orb_conf, deep_conf, len(unknown_files)


def print_and_collect(title, orb_conf, deep_conf, lines):
    orb_m = compute_metrics(**orb_conf)
    deep_m = compute_metrics(**deep_conf)
    block = [f"\n{title}", f"{'Metric':<12}{'ORB':>12}{'DeepFace':>14}"]
    for key in ["TP", "FP", "FN", "TN"]:
        block.append(f"{key:<12}{orb_m[key]:>12}{deep_m[key]:>14}")
    for key in ["accuracy", "precision", "recall", "f1"]:
        block.append(f"{key:<12}{orb_m[key]*100:>11.1f}%{deep_m[key]*100:>13.1f}%")
    for line in block:
        print(line)
    lines.extend(block)
    return orb_m, deep_m


def run_all():
    print("=" * 60)
    print("  EVALUASI LENGKAP — ORB vs DeepFace (ConvergentAI 2026)")
    print("=" * 60)

    print("\n[1/4] Menyiapkan known_faces/ (ORB + DeepFace)...")
    known_orb = load_known_faces_orb()
    print(f"      ORB template siap: {len(known_orb)} foto")
    print("      Menghitung embedding DeepFace (Facenet512)... (bisa lama di awal)")
    known_deepface = load_known_faces_deepface()
    print(f"      DeepFace embedding siap: {len(known_deepface)} foto")

    labels = sorted({kf["name"] for kf in known_orb})
    print(f"      Jumlah individu: {len(labels)} -> {', '.join(labels)}")

    print("\n[2/4] CLOSED-SET (exhaustive pairwise antar known_faces)...")
    closed_orb, closed_deep = run_closed_set(known_orb, known_deepface)

    print("\n[3/4] OPEN-SET (dataset/unknown/ vs seluruh database)...")
    open_orb, open_deep, unknown_count = run_open_set(known_orb, known_deepface)

    lines = []
    lines.append("HASIL EVALUASI LENGKAP -- ORB vs DeepFace (Facenet512)")
    lines.append("ConvergentAI 2026 submission")
    lines.append("=" * 60)
    lines.append(f"Jumlah individu terdaftar: {len(labels)} ({', '.join(labels)})")
    lines.append(f"Jumlah foto unknown (open-set): {unknown_count}")

    print("\n[4/4] Menyusun hasil akhir...")
    print_and_collect("CLOSED-SET (pairwise antar member terdaftar):", closed_orb, closed_deep, lines)

    if open_orb is not None:
        print_and_collect("OPEN-SET (orang asing vs database, false acceptance):", open_orb, open_deep, lines)
        combined_orb = add_confusion(closed_orb, open_orb)
        combined_deep = add_confusion(closed_deep, open_deep)
        print_and_collect("GABUNGAN (closed-set + open-set):", combined_orb, combined_deep, lines)
    else:
        lines.append("\n[CATATAN] Open-set false acceptance TIDAK diuji (dataset/unknown/ kosong).")
        lines.append("Sebutkan ini sebagai keterbatasan (limitation) di paper, jangan klaim false")
        lines.append("acceptance testing terhadap orang asing kalau bagian ini dilewati.")

    with open(BASE_DIR / "hasil_evaluasi_full.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nDisimpan ke hasil_evaluasi_full.txt")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        final_orb = combined_orb if open_orb is not None else closed_orb
        final_deep = combined_deep if open_orb is not None else closed_deep
        orb_m = compute_metrics(**final_orb)
        deep_m = compute_metrics(**final_deep)

        metrics = ["accuracy", "precision", "recall", "f1"]
        orb_vals = [orb_m[m] * 100 for m in metrics]
        deep_vals = [deep_m[m] * 100 for m in metrics]
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
        title_suffix = "Closed-set + Open-set" if open_orb is not None else "Closed-set only"
        ax.set_title(f"ORB vs DeepFace (Facenet512) — {title_suffix}", fontsize=12)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(BASE_DIR / "bar_chart_full.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Bar chart disimpan ke bar_chart_full.png")
    except ImportError:
        print("matplotlib tidak terinstall, skip chart. Jalankan: pip install matplotlib")

    print("\n✅ Selesai! Salin angka dari hasil_evaluasi_full.txt ke tabel paper kamu.")


if __name__ == "__main__":
    if not KNOWN_FACES_DIR.exists():
        print(f"Folder tidak ditemukan: {KNOWN_FACES_DIR}")
        print("Taruh script ini sejajar dengan folder known_faces/ lalu jalankan ulang.")
    else:
        run_all()
