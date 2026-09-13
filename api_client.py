import json
import time
from urllib import error, request
from urllib.parse import quote
from config import DEFAULT_API_URL, DEFAULT_DETECTION_API_URL, LARAVEL_API_KEY
from storage import image_to_base64

LAST_MEMBER_NOTIFICATION = {"label": None, "sent_at": 0.0}


def request_json(url, method="GET", payload=None, timeout=10):
    if not LARAVEL_API_KEY:
        raise RuntimeError(
            "LARAVEL_API_KEY belum diatur. Salin nilainya dari Laravel .env "
            "ke facedetec/.env."
        )

    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    http_request = request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-API-KEY": LARAVEL_API_KEY,
        },
        method=method,
    )

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Laravel API error {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Tidak bisa terhubung ke Laravel API: {exc.reason}") from exc


def post_json(url, payload, timeout=10):
    return request_json(url, method="POST", payload=payload, timeout=timeout)


def get_customer(face_label):
    return request_json(
        f"{DEFAULT_API_URL.rsplit('/register-face', 1)[0]}/{quote(face_label, safe='')}",
        timeout=5,
    )


def update_customer(face_label, name, phone, discount_percent):
    return request_json(
        f"{DEFAULT_API_URL.rsplit('/register-face', 1)[0]}/{quote(face_label, safe='')}",
        method="PATCH",
        payload={
            "name": name,
            "phone": phone,
            "discount_percent": discount_percent,
        },
        timeout=5,
    )


def notify_member_detected(face_label, cosine_distance, liveness_passed, api_url=DEFAULT_DETECTION_API_URL):
    global LAST_MEMBER_NOTIFICATION

    now = time.time()
    same_label = LAST_MEMBER_NOTIFICATION["label"] == face_label
    sent_recently = now - LAST_MEMBER_NOTIFICATION["sent_at"] < 15

    if not liveness_passed:
        return

    if same_label and sent_recently:
        return

    LAST_MEMBER_NOTIFICATION = {"label": face_label, "sent_at": now}

    try:
        post_json(
            api_url,
            {
                "face_label": face_label,
                # Laravel stores similarity (high is better); the recognition
                # engine returns cosine distance (low is better).
                "score": round(max(0.0, min(1.0, 1.0 - float(cosine_distance))), 4),
                "liveness_passed": True,
            },
            timeout=1,
        )
    except RuntimeError as exc:
        print(f"Gagal kirim notif member ke Laravel: {exc}")


def send_register_customer(
    name,
    phone,
    discount_percent,
    face_image,
    face_label,
    face_embedding,
    consent_confirmed,
    api_url=DEFAULT_API_URL,
):
    payload = {
        "name": name,
        "phone": phone,
        "discount_percent": discount_percent,
        "face_label": face_label,
        "face_image_base64": "data:image/jpeg;base64," + image_to_base64(face_image),
        "face_embedding": face_embedding,
        "consent_confirmed": consent_confirmed,
    }
    return post_json(api_url, payload, timeout=10)
