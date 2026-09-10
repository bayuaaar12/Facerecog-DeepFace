import base64
import json
import re
import time
import cv2
import numpy as np
from config import (
    EMBEDDING_FILE_VERSION,
    EMBEDDINGS_PATH,
    FACENET_MODEL_NAME,
    KNOWN_FACES_DIR,
    REGISTER_SAMPLE_COUNT,
    REGISTER_SAMPLE_INTERVAL,
    REGISTER_SAMPLE_TIMEOUT,
)
from recognition import create_embedding, face_detection, find_largest_face, verify_registration_liveness

KNOWN_FACES = []


def slugify(value):
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return normalized.strip("_") or "customer"


def face_label_from_path(image_path):
    return re.sub(r"_\d+$", "", image_path.stem)


def display_face_label(face_label):
    return face_label.replace("_", " ").title()


def load_embeddings_file():
    """Read the local embedding index, returning None when it must be rebuilt."""
    if not EMBEDDINGS_PATH.exists():
        return None
    try:
        with EMBEDDINGS_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if (
            payload.get("version") != EMBEDDING_FILE_VERSION
            or payload.get("model_name") != FACENET_MODEL_NAME
            or not isinstance(payload.get("samples"), list)
        ):
            return None
        for sample in payload["samples"]:
            if not isinstance(sample.get("label"), str) or not isinstance(sample.get("sample"), str):
                return None
            vector = np.asarray(sample.get("embedding"), dtype=np.float32)
            if vector.ndim != 1 or vector.size == 0:
                return None
        return payload["samples"]
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def embeddings_match_face_files(embeddings):
    image_samples = {path.stem for path in list_known_face_files()}
    embedded_samples = {item["sample"] for item in embeddings}
    return image_samples == embedded_samples


def rebuild_embeddings():
    """Build a FaceNet512 embedding for every image in known_faces/."""
    global KNOWN_FACES

    image_paths = list_known_face_files()
    if not image_paths:
        KNOWN_FACES = []
        payload = {
            "version": EMBEDDING_FILE_VERSION,
            "model_name": FACENET_MODEL_NAME,
            "samples": [],
        }
        with EMBEDDINGS_PATH.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        print("Folder known_faces kosong. Database embedding dikosongkan.")
        return KNOWN_FACES

    rebuilt = []
    failures = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            failures.append(f"{image_path.name} (gambar tidak dapat dibaca)")
            continue
        try:
            embedding = create_embedding(image)
        except RuntimeError as exc:
            failures.append(f"{image_path.name} ({exc})")
            continue
        rebuilt.append(
            {
                "name": face_label_from_path(image_path),
                "sample": image_path.stem,
                "embedding": embedding,
            }
        )

    if not rebuilt:
        raise RuntimeError(
            "Tidak ada embedding yang berhasil dibuat. " + "; ".join(failures)
        )

    payload = {
        "version": EMBEDDING_FILE_VERSION,
        "model_name": FACENET_MODEL_NAME,
        "samples": [
            {"label": item["name"], "sample": item["sample"], "embedding": item["embedding"].tolist()}
            for item in rebuilt
        ],
    }
    with EMBEDDINGS_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file)
    KNOWN_FACES = rebuilt

    if failures:
        print("Sebagian sampel tidak dapat dibuat embedding-nya: " + "; ".join(failures))
    print(f"Embedding {len(rebuilt)} sampel berhasil dibangun ulang.")
    return KNOWN_FACES


def load_known_faces(rebuild_if_needed=True):
    global KNOWN_FACES
    embeddings = load_embeddings_file()
    if embeddings is None or not embeddings_match_face_files(embeddings):
        if rebuild_if_needed:
            return rebuild_embeddings()
        return []

    KNOWN_FACES = [
        {"name": item["label"], "sample": item["sample"], "embedding": np.asarray(item["embedding"], dtype=np.float32)}
        for item in embeddings
    ]
    return KNOWN_FACES


def known_person_count():
    return len({face_label_from_path(path) for path in list_known_face_files()})


def save_face_locally(face_image, face_label):
    KNOWN_FACES_DIR.mkdir(exist_ok=True)
    target_path = KNOWN_FACES_DIR / f"{face_label}.jpg"
    cv2.imwrite(str(target_path), face_image)
    return target_path


def next_face_sample_path(face_label):
    existing_numbers = []
    for image_path in list_known_face_files():
        if face_label_from_path(image_path) != face_label:
            continue

        match = re.search(r"_(\d+)$", image_path.stem)
        existing_numbers.append(int(match.group(1)) if match else 1)

    next_number = max(existing_numbers, default=0) + 1
    return KNOWN_FACES_DIR / f"{face_label}_{next_number}.jpg"


def save_face_sample(face_image, face_label):
    KNOWN_FACES_DIR.mkdir(exist_ok=True)
    target_path = next_face_sample_path(face_label)
    if not cv2.imwrite(str(target_path), face_image):
        raise RuntimeError(f"Gagal menyimpan sampel wajah: {target_path}")
    return target_path


def collect_face_samples(camera, initial_gray_frame, initial_faces, face_label):
    samples = []
    face_boxes = []
    started_at = time.time()
    last_capture_at = 0.0

    selected_face = find_largest_face(initial_faces)
    if selected_face is not None and initial_gray_frame is not None:
        x, y, w, h = selected_face
        face_boxes.append((x, y, w, h))
        samples.append(cv2.cvtColor(initial_gray_frame[y : y + h, x : x + w], cv2.COLOR_GRAY2BGR))
        last_capture_at = time.time()

    while len(samples) < REGISTER_SAMPLE_COUNT:
        if time.time() - started_at > REGISTER_SAMPLE_TIMEOUT:
            break

        success, frame = camera.read()
        if not success:
            break

        if time.time() - last_capture_at < REGISTER_SAMPLE_INTERVAL:
            cv2.waitKey(1)
            continue

        frame = cv2.flip(frame, 1)
        gray_frame, faces = face_detection(frame)
        selected_face = find_largest_face(faces)
        if selected_face is None:
            continue

        x, y, w, h = selected_face
        face_boxes.append((x, y, w, h))
        samples.append(cv2.cvtColor(gray_frame[y : y + h, x : x + w], cv2.COLOR_GRAY2BGR))
        last_capture_at = time.time()

    if not verify_registration_liveness(face_boxes):
        print(
            "[WARNING Anti-Spoofing] Terdeteksi potensi foto statis tanpa pergerakan kepala alami. "
            "Pastikan subjek bergerak/berkedip secara alami saat registrasi."
        )

    saved_paths = [save_face_sample(sample, face_label) for sample in samples]
    return samples, saved_paths


def list_known_face_files():
    KNOWN_FACES_DIR.mkdir(exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png"}
    return [
        image_path
        for image_path in sorted(KNOWN_FACES_DIR.iterdir())
        if image_path.suffix.lower() in image_extensions
    ]


def list_known_face_entries():
    entries_by_label = {}
    for image_path in list_known_face_files():
        label = face_label_from_path(image_path)
        entries_by_label.setdefault(label, []).append(image_path)

    return [
        {"label": label, "files": files}
        for label, files in sorted(entries_by_label.items())
    ]


def delete_known_face(face_entry):
    global KNOWN_FACES

    for face_path in face_entry["files"]:
        try:
            face_path.unlink()
        except FileNotFoundError:
            pass

    KNOWN_FACES = rebuild_embeddings()


def image_to_base64(face_image):
    success, buffer = cv2.imencode(".jpg", face_image)
    if not success:
        raise RuntimeError("Gagal mengubah gambar wajah ke JPEG.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")
