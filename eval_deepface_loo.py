import re
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MODEL_NAME = "Facenet512"
THRESHOLD = 0.40
SELECTED_FILES = [
    "bim_1.jpg",
    "bim_2.jpg",
    "bim_3.jpg",
    "bim_4.jpg",
    "bim_5.jpg",
    "bayu_1.jpg",
    "bayu_2.jpg",
    "bayu_3.jpg",
    "bayu_4.jpg",
    "bayu_5.jpg",
    "alvin_1.jpg",
    "alvin_2.jpg",
    "alvin_3.jpg",
    "alvin_4.jpg",
    "alvin_5.jpg",
    "nopal_1.jpg",
    "nopal_2.jpg",
    "nopal_3.jpg",
    "nopal_4.jpg",
    "nopal_5.jpg",
]

FACE_IMAGE_SIZE = (200, 200)
LOW_LIGHT_MEAN_LIMIT = 95
LOW_LIGHT_TARGET_MEAN = 125
LOW_LIGHT_BRIGHTNESS_BONUS = 18
ORB_DISTANCE_LIMIT = 60
ORB_RATIO_TEST = 0.75
MIN_GOOD_MATCHES = 18
MIN_MATCH_RATIO = 0.18


def label_from_path(path):
    return re.sub(r"_\d+$", "", path.stem)


def cosine_distance(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / norm)


def percent(value):
    return value * 100.0


def normalize_lighting(gray):
    mean_val = float(np.mean(gray))
    if mean_val < LOW_LIGHT_MEAN_LIMIT:
        alpha = min(2.0, LOW_LIGHT_TARGET_MEAN / max(mean_val, 1.0))
        gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=LOW_LIGHT_BRIGHTNESS_BONUS)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_face(gray):
    resized = cv2.resize(gray, FACE_IMAGE_SIZE)
    return normalize_lighting(resized)


def template_similarity(left, right):
    corr = cv2.matchTemplate(left, right, cv2.TM_CCOEFF_NORMED)[0][0]
    abs_sim = 1.0 - (np.mean(cv2.absdiff(left, right)) / 255.0)
    return max(0.0, ((corr + 1.0) / 2.0) * 100.0, abs_sim * 100.0)


def count_good_matches(matcher, left_desc, right_desc):
    if left_desc is None or right_desc is None or len(left_desc) < 2 or len(right_desc) < 2:
        return 0
    knn = matcher.knnMatch(left_desc, right_desc, k=2)
    good = [
        m for m, n in knn
        if m.distance < ORB_DISTANCE_LIMIT and m.distance < ORB_RATIO_TEST * n.distance
    ]
    return len(good)


def metrics_from_counts(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def evaluate_pairwise(samples, predict_same):
    tp = fp = fn = tn = 0
    total_pairs = 0

    for left, right in combinations(samples, 2):
        same_person = left["label"] == right["label"]
        predicted_same = predict_same(left, right)
        total_pairs += 1

        if same_person and predicted_same:
            tp += 1
        elif same_person and not predicted_same:
            fn += 1
        elif not same_person and predicted_same:
            fp += 1
        else:
            tn += 1

    metrics = metrics_from_counts(tp, fp, fn, tn)
    metrics["pairs"] = total_pairs
    return metrics


def print_result(title, result):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)
    print(f"Pairs:     {result['pairs']}")
    print(f"Accuracy:  {percent(result['accuracy']):.1f}%")
    print(f"Precision: {percent(result['precision']):.1f}%")
    print(f"Recall:    {percent(result['recall']):.1f}%")
    print(f"F1-Score:  {percent(result['f1']):.1f}%")
    print(f"TP={result['TP']} FP={result['FP']} FN={result['FN']} TN={result['TN']}")


def main():
    from deepface import DeepFace

    if SELECTED_FILES:
        image_paths = [KNOWN_FACES_DIR / name for name in SELECTED_FILES]
        missing = [p.name for p in image_paths if not p.exists()]
        if missing:
            raise SystemExit(
                "File pilihan ini belum ada di known_faces:\n- "
                + "\n- ".join(missing)
            )
    else:
        image_paths = sorted(
            p for p in KNOWN_FACES_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

    if len(image_paths) < 2:
        raise SystemExit("Butuh minimal 2 gambar di folder known_faces.")

    print(f"Loading {len(image_paths)} gambar dari {KNOWN_FACES_DIR}")
    print(f"Model: {MODEL_NAME}")

    orb = cv2.ORB_create(nfeatures=700)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    deepface_samples = []
    orb_samples = []

    for path in image_paths:
        label = label_from_path(path)
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            proc = preprocess_face(gray)
            keypoints, descriptors = orb.detectAndCompute(proc, None)
            orb_samples.append({
                "path": path,
                "label": label,
                "image": proc,
                "descriptors": descriptors,
                "keypoint_count": len(keypoints),
            })

        try:
            embedding = DeepFace.represent(
                img_path=str(path),
                model_name=MODEL_NAME,
                enforce_detection=False,
            )[0]["embedding"]
            deepface_samples.append({
                "path": path,
                "label": label,
                "embedding": np.array(embedding),
            })
            print(f"  OK   {path.name}")
        except Exception as exc:
            print(f"  SKIP {path.name}: {exc}")

    if len(deepface_samples) < 2 or len(orb_samples) < 2:
        raise SystemExit("Embedding berhasil kurang dari 2 gambar, evaluasi tidak bisa dihitung.")

    labels = sorted({s["label"] for s in deepface_samples})
    print(f"\nTotal embedding: {len(deepface_samples)}")
    print(f"Total label: {len(labels)} ({', '.join(labels)})")
    print(f"Threshold cosine distance: {THRESHOLD:.2f}")

    def predict_deepface_same(left, right):
        distance = cosine_distance(left["embedding"], right["embedding"])
        return distance <= THRESHOLD

    def predict_orb_same(left, right):
        good = count_good_matches(matcher, left["descriptors"], right["descriptors"])
        desc_count = max(min(left["keypoint_count"], right["keypoint_count"]), 1)
        ratio = good / desc_count
        tmpl = template_similarity(left["image"], right["image"])
        return (good >= MIN_GOOD_MATCHES and ratio >= MIN_MATCH_RATIO) or (
            tmpl >= 62 and good >= 4
        )

    orb_result = evaluate_pairwise(orb_samples, predict_orb_same)
    deepface_result = evaluate_pairwise(deepface_samples, predict_deepface_same)

    print_result("ORB LOO / known_faces only", orb_result)
    print_result("DeepFace LOO / known_faces only", deepface_result)

    print("\n" + "=" * 54)
    print("FINAL TABLE")
    print("=" * 54)
    print(f"{'Metric':<12} {'ORB':>10} {'DeepFace':>12} {'Delta':>10}")
    print("-" * 54)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        orb_value = orb_result[metric]
        deep_value = deepface_result[metric]
        delta = deep_value - orb_value
        name = "F1-Score" if metric == "f1" else metric.capitalize()
        print(
            f"{name:<12} {percent(orb_value):>9.1f}% "
            f"{percent(deep_value):>11.1f}% {percent(delta):>+9.1f}%"
        )


if __name__ == "__main__":
    main()
