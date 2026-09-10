import json
import time
from urllib import error, request
from config import DEFAULT_API_URL, DEFAULT_DETECTION_API_URL
from storage import image_to_base64

LAST_MEMBER_NOTIFICATION = {"label": None, "sent_at": 0.0}


def post_json(url, payload, timeout=10):
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Laravel API error {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Tidak bisa terhubung ke Laravel API: {exc.reason}") from exc


def notify_member_detected(face_label, score, api_url=DEFAULT_DETECTION_API_URL):
    global LAST_MEMBER_NOTIFICATION

    now = time.time()
    same_label = LAST_MEMBER_NOTIFICATION["label"] == face_label
    sent_recently = now - LAST_MEMBER_NOTIFICATION["sent_at"] < 15

    if same_label and sent_recently:
        return

    LAST_MEMBER_NOTIFICATION = {"label": face_label, "sent_at": now}

    try:
        post_json(
            api_url,
            {
                "face_label": face_label,
                "score": round(float(score), 4),
            },
            timeout=1,
        )
    except RuntimeError as exc:
        print(f"Gagal kirim notif member ke Laravel: {exc}")


def send_register_customer(name, phone, discount_percent, face_image, face_label, api_url=DEFAULT_API_URL):
    payload = {
        "name": name,
        "phone": phone,
        "discount_percent": discount_percent,
        "face_label": face_label,
        "image_base64": image_to_base64(face_image),
    }
    return post_json(api_url, payload, timeout=10)
