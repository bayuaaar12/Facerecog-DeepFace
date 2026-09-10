"""=============================================================
 RECOGNITION REALTIME: ORB Hybrid (Pembanding/Template)
-------------------------------------------------------------
 Tujuan:
 - File terpisah untuk recognition realtime berbasis ORB hybrid
 - Agar bisa dibandingkan dengan DeepFace di file lain

 Cara pakai:
 - Pastikan ada:
   dataset/known_faces/  (isi gambar member)
   face_ref.xml            (untuk deteksi wajah pakai Haar Cascade)
 - Jalankan:
   python orb_hybrid_realtime.py

 Catatan:
 - Ini hanya recognition realtime ORB hybrid (bukan DeepFace).
 - Deteksi wajah: Haar cascade multi-rotation seperti main.py.
=============================================================
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
CASCADE_PATH = BASE_DIR / "face_ref.xml"
KNOWN_FACES_DIR = BASE_DIR / "dataset" / "known_faces"

# ---------------- ORB Hybrid constants (konsisten dengan main.py) ----------------
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

FACE_DETECTION_ANGLES = (0, -20, 20, -35, 35)
FACE_DETECTION_IOU_LIMIT = 0.35

# ORB setup
orb = cv2.ORB_create(nfeatures=700)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

face_ref = cv2.CascadeClassifier(str(CASCADE_PATH))


def slugify_label(image_stem: str) -> str:
    return re.sub(r"_\d+$", "", image_stem)


def normalize_lighting(gray_image: np.ndarray) -> np.ndarray:
    mean_light = float(np.mean(gray_image))
    if mean_light < LOW_LIGHT_MEAN_LIMIT:
        alpha = min(2.0, LOW_LIGHT_TARGET_MEAN / max(mean_light, 1.0))
        gray_image = cv2.convertScaleAbs(
            gray_image,
            alpha=alpha,
            beta=LOW_LIGHT_BRIGHTNESS_BONUS,
        )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)


def preprocess_face(face_roi: np.ndarray) -> np.ndarray:
    resized_face = cv2.resize(face_roi, FACE_IMAGE_SIZE)
    return normalize_lighting(resized_face)


def template_similarity(source_face: np.ndarray, known_face: np.ndarray) -> float:
    correlation = cv2.matchTemplate(source_face, known_face, cv2.TM_CCOEFF_NORMED)[0][0]
    abs_similarity = 1.0 - (np.mean(cv2.absdiff(source_face, known_face)) / 255.0)
    # hasil 0..100, konsisten dengan main.py
    return float(max(0.0, ((correlation + 1.0) / 2.0) * 100.0, abs_similarity * 100.0))


def count_good_matches(source_descriptors: np.ndarray, known_descriptors: np.ndarray) -> int:
    if source_descriptors is None:
        return 0
    if known_descriptors is None or len(known_descriptors) < 2:
        return 0

    knn_matches = matcher.knnMatch(source_descriptors, known_descriptors, k=2)
    good_matches = [
        first
        for first, second in knn_matches
        if first.distance < ORB_DISTANCE_LIMIT
        and first.distance < ORB_RATIO_TEST * second.distance
    ]
    return len(good_matches)


def load_known_faces_orb() -> list[dict]:
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    faces = []

    if not KNOWN_FACES_DIR.exists():
        return faces

    for image_path in sorted(KNOWN_FACES_DIR.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        image = preprocess_face(image)
        keypoints, descriptors = orb.detectAndCompute(image, None)

        faces.append(
            {
                "name": slugify_label(image_path.stem),
                "image": image,
                "descriptors": descriptors,
                "keypoint_count": len(keypoints),
            }
        )

    return faces


KNOWN_FACES = load_known_faces_orb()


def box_iou(first_box, second_box) -> float:
    first_x, first_y, first_w, first_h = first_box
    second_x, second_y, second_w, second_h = second_box

    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_w, second_x + second_w)
    bottom = min(first_y + first_h, second_y + second_h)

    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = first_w * first_h
    second_area = second_w * second_h
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def merge_overlapping_boxes(boxes: list[tuple] | np.ndarray) -> np.ndarray:
    merged_boxes = []
    boxes_iter = boxes if isinstance(boxes, list) else boxes.tolist()
    for box in sorted(boxes_iter, key=lambda item: item[2] * item[3], reverse=True):
        if all(box_iou(box, existing_box) < FACE_DETECTION_IOU_LIMIT for existing_box in merged_boxes):
            merged_boxes.append(box)
    return np.array(merged_boxes, dtype=np.int32)


def unrotate_box(box, rotation_matrix, width, height):
    x, y, w, h = box
    inverse_matrix = cv2.invertAffineTransform(rotation_matrix)

    corners = np.array(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ],
        dtype=np.float32,
    )

    transformed = cv2.transform(np.array([corners]), inverse_matrix)[0]
    x1, y1 = np.floor(transformed.min(axis=0)).astype(int)
    x2, y2 = np.ceil(transformed.max(axis=0)).astype(int)

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def detect_faces_at_angles(gray_frame: np.ndarray) -> np.ndarray:
    detections = []
    height, width = gray_frame.shape[:2]
    center = (width / 2, height / 2)

    for angle in FACE_DETECTION_ANGLES:
        if angle == 0:
            rotated_frame = gray_frame
            rotation_matrix = None
        else:
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_frame = cv2.warpAffine(
                gray_frame,
                rotation_matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

        faces = face_ref.detectMultiScale(
            rotated_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        for x, y, w, h in faces:
            if rotation_matrix is None:
                box = (x, y, w, h)
            else:
                box = unrotate_box((x, y, w, h), rotation_matrix, width, height)
            detections.append(box)

    return merge_overlapping_boxes(detections)


def face_detection(frame_bgr: np.ndarray):
    gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_frame = normalize_lighting(gray_frame)
    faces = detect_faces_at_angles(gray_frame)
    return gray_frame, faces


def recognize_orb_hybrid(face_roi_gray: np.ndarray, known_faces: list[dict]):
    normalized_face = preprocess_face(face_roi_gray)
    keypoints, descriptors = orb.detectAndCompute(normalized_face, None)

    if not known_faces:
        return "Unknown", 0.0

    desc_count = max(len(keypoints), 1)

    scores_by_name = {}
    for kf in known_faces:
        orb_score = count_good_matches(descriptors, kf["descriptors"])
        tmpl_score = template_similarity(normalized_face, kf["image"])
        combined_score = orb_score + int(max(0, tmpl_score - 50) * 0.35)

        name = kf["name"]
        cur = scores_by_name.get(name, {"combined": 0, "orb": 0, "template": 0})
        if combined_score > cur["combined"]:
            scores_by_name[name] = {
                "combined": combined_score,
                "orb": orb_score,
                "template": tmpl_score,
            }

    best = max(scores_by_name.items(), key=lambda x: x[1]["combined"], default=(None, {}))

    if best[0] is None:
        return "Unknown", 0.0

    best_name = best[0]
    best_val = best[1]

    # second best gabungan
    rest_combined = [v["combined"] for k, v in scores_by_name.items() if k != best_name]
    second_combined = max(rest_combined, default=0)

    rest_templates = [v["template"] for k, v in scores_by_name.items() if k != best_name]
    second_template = max(rest_templates, default=0)

    orb_score = float(best_val.get("orb", 0))
    tmpl_score = float(best_val.get("template", 0))
    comb_score = float(best_val.get("combined", 0))

    ratio = orb_score / desc_count

    req_good = MIN_GOOD_MATCHES
    req_margin = MIN_MATCH_MARGIN
    req_ratio = MIN_MATCH_RATIO
    if desc_count < LOW_FEATURE_DESCRIPTOR_LIMIT:
        req_good = LOW_FEATURE_MIN_GOOD_MATCHES
        req_margin = LOW_FEATURE_MIN_MATCH_MARGIN
        req_ratio = LOW_FEATURE_MIN_MATCH_RATIO

    orb_conf = (
        orb_score >= req_good
        and comb_score - second_combined >= req_margin
        and ratio >= req_ratio
    )

    tmpl_conf = (
        tmpl_score >= MIN_TEMPLATE_SIMILARITY
        and tmpl_score - second_template >= MIN_TEMPLATE_MARGIN
        and orb_score >= MIN_TEMPLATE_ORB_SUPPORT
    )

    if not orb_conf and not tmpl_conf:
        return "Unknown", comb_score

    return best_name, comb_score


def run(camera_index: int = 0):
    if not KNOWN_FACES:
        print("[WARNING] dataset/known_faces kosong. Isi dulu gambar member.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera index {camera_index} tidak bisa dibuka.")

    cv2.namedWindow("ORB Hybrid Realtime", cv2.WINDOW_AUTOSIZE)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        gray_frame, faces = face_detection(frame)

        for x, y, w, h in faces:
            face_roi = gray_frame[y : y + h, x : x + w]
            label, score = recognize_orb_hybrid(face_roi, KNOWN_FACES)

            color = (238, 112, 35) if label != "Unknown" else (90, 90, 96)
            text = f"{label} ({int(score)})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("ORB Hybrid Realtime", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # gunakan kamera index 0 by default
    run(camera_index=0)

