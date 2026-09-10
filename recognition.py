import platform
import cv2
import numpy as np
from config import (
    CASCADE_PATH,
    COSINE_DISTANCE_THRESHOLD,
    FACE_DETECTION_ANGLES,
    FACE_DETECTION_IOU_LIMIT,
    FACENET_MODEL_NAME,
    LOW_LIGHT_BRIGHTNESS_BONUS,
    LOW_LIGHT_MEAN_LIMIT,
    LOW_LIGHT_TARGET_MEAN,
)

face_ref = cv2.CascadeClassifier(str(CASCADE_PATH))


def default_camera_index():
    if platform.system() == "Darwin":
        return 0
    return 0


def camera_backend_candidates(camera_index):
    system_name = platform.system()

    if system_name == "Darwin":
        return [
            (camera_index, cv2.CAP_AVFOUNDATION),
            (camera_index, cv2.CAP_ANY),
        ]

    if system_name == "Windows":
        return [
            (camera_index, cv2.CAP_DSHOW),
            (camera_index, cv2.CAP_MSMF),
            (camera_index, cv2.CAP_ANY),
        ]

    return [(camera_index, cv2.CAP_ANY)]


def open_camera(camera_index=0):
    for index, backend in camera_backend_candidates(camera_index):
        camera = cv2.VideoCapture(index, backend)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            return camera
        camera.release()

    extra_hint = ""
    if platform.system() == "Darwin":
        extra_hint = (
            " Di macOS cek System Settings > Privacy & Security > Camera, "
            "pastikan Terminal/Python diberi izin kamera. Jika kamera eksternal, "
            "coba Camera Index 1 atau 2."
        )

    raise RuntimeError(f"Camera index {camera_index} tidak bisa dibuka.{extra_hint}")


def get_deepface():
    """Import DeepFace only when recognition/embedding is actually needed."""
    try:
        from deepface import DeepFace
    except ImportError as exc:
        raise RuntimeError(
            "DeepFace belum terpasang. Install dependency dengan: "
            "pip install deepface tensorflow opencv-python numpy"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"DeepFace gagal diinisialisasi: {exc}") from exc
    return DeepFace


def create_embedding(face_image):
    """
    # [AI/ML INFERENCE]
    Create one FaceNet512 embedding from a face crop (BGR OpenCV image).
    Menggunakan model Deep Learning DCNN (FaceNet512) untuk mengekstrak vektor fitur 512-dimensi.
    """
    if face_image is None or face_image.size == 0:
        raise RuntimeError("Embedding tidak dapat dibuat: crop wajah kosong.")

    try:
        result = get_deepface().represent(
            img_path=face_image,
            model_name=FACENET_MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )
        if not result or "embedding" not in result[0]:
            raise ValueError("DeepFace tidak mengembalikan embedding.")
        embedding = np.asarray(result[0]["embedding"], dtype=np.float32)
        if embedding.ndim != 1 or embedding.size == 0 or not np.all(np.isfinite(embedding)):
            raise ValueError("Embedding FaceNet512 tidak valid.")
        return embedding
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "Model DeepFace FaceNet512 gagal dimuat atau embedding tidak dapat dibuat: "
            f"{exc}"
        ) from exc


def cosine_distance(first_embedding, second_embedding):
    """
    # [AI/ML INFERENCE]
    Menghitung jarak kosinus (Cosine Distance) antara dua vektor embedding biometrik 512D.
    """
    first = np.asarray(first_embedding, dtype=np.float32)
    second = np.asarray(second_embedding, dtype=np.float32)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        return 1.0
    return float(1.0 - np.dot(first, second) / denominator)


def recognize_face(face_roi, known_faces):
    """
    # [AI/ML INFERENCE]
    Pengenalan wajah multi-class / open-set menggunakan ekstraksi embedding FaceNet512
    dan komparasi Cosine Distance dengan seluruh sampel di database.
    """
    if not known_faces:
        return "Unknown", 1.0

    camera_embedding = create_embedding(face_roi)
    best_distance_by_name = {}
    for known_face in known_faces:
        distance = cosine_distance(camera_embedding, known_face["embedding"])
        name = known_face["name"]
        best_distance_by_name[name] = min(best_distance_by_name.get(name, 1.0), distance)

    best_name, best_distance = min(best_distance_by_name.items(), key=lambda item: item[1])
    if best_distance > COSINE_DISTANCE_THRESHOLD:
        return "Unknown", best_distance
    return best_name, best_distance


def face_detection(frame):
    """
    # [LOGIKA PENDUKUNG / NON-AI]
    Deteksi posisi wajah menggunakan Viola-Jones / Haar Cascade klasik dengan perataan cahaya.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = normalize_lighting(gray_frame)
    faces = detect_faces_at_angles(gray_frame)
    return gray_frame, faces


def detect_faces_at_angles(gray_frame):
    """
    # [LOGIKA PENDUKUNG / NON-AI]
    Deteksi multi-sudut rotasi murni pengolahan citra geometri (WarpAffine + Haar Cascade).
    """
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


def box_iou(first_box, second_box):
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
    return intersection / union if union else 0


def merge_overlapping_boxes(boxes):
    merged_boxes = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if all(box_iou(box, existing_box) < FACE_DETECTION_IOU_LIMIT for existing_box in merged_boxes):
            merged_boxes.append(box)
    return np.array(merged_boxes, dtype=np.int32)


def normalize_lighting(gray_image):
    """
    # [LOGIKA PENDUKUNG / NON-AI]
    Normalisasi kecerahan citra konvensional (CLAHE + brightness scaling).
    """
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


def verify_registration_liveness(face_boxes):
    """
    # [LOGIKA PENDUKUNG / NON-AI]
    Liveness Check Sederhana Anti-Spoofing:
    Memeriksa pergerakan posisi pusat wajah (centroid shift) selama pengambilan sampel registrasi.
    Foto statis yang dihadapkan ke kamera tidak memiliki pergerakan alami (centroid shift ~0).
    """
    if len(face_boxes) < 2:
        return True
    centroids = [(x + w / 2.0, y + h / 2.0) for (x, y, w, h) in face_boxes]
    dx = max(c[0] for c in centroids) - min(c[0] for c in centroids)
    dy = max(c[1] for c in centroids) - min(c[1] for c in centroids)
    movement = float(np.sqrt(dx**2 + dy**2))
    return movement >= 3.0


def preprocess_face(face_roi):
    """Legacy helper for Haar/registration display; recognition uses raw BGR crops."""
    if len(face_roi.shape) == 3:
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    return normalize_lighting(face_roi)


def find_largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda item: item[2] * item[3])
