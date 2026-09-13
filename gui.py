import json
import cv2
import numpy as np
from config import (
    COLOR_ACCENT,
    COLOR_ACCENT_SOFT,
    COLOR_BG,
    COLOR_BLUE,
    COLOR_BORDER,
    COLOR_MUTED,
    COLOR_PANEL,
    COLOR_PRIMARY,
    COLOR_PRIMARY_SOFT,
    COLOR_SURFACE,
    COLOR_TEXT,
    REGISTER_SAMPLE_COUNT,
)
from recognition import (
    MotionLivenessChallenge,
    create_embedding,
    face_detection,
    find_largest_face,
    open_camera,
    recognize_face,
)
from storage import (
    delete_known_face,
    display_face_label,
    image_to_base64,
    known_person_count,
    list_known_face_entries,
    load_known_faces,
    rebuild_embeddings,
    save_face_sample,
    slugify,
)
from api_client import (
    get_customer,
    notify_member_detected,
    post_json,
    send_register_customer,
    update_customer,
)


def prompt_int(label, default_value=0):
    raw_value = input(f"{label} [{default_value}]: ").strip()
    if not raw_value:
        return default_value

    try:
        return int(raw_value)
    except ValueError:
        print(f"{label} harus angka. Dipakai default {default_value}.")
        return default_value


def draw_centered_text(canvas, text, center, font_scale, color, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(center[0] - text_size[0] / 2)
    y = int(center[1] + text_size[1] / 2)
    cv2.putText(canvas, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_round_rect(canvas, rect, color, radius=14, thickness=-1, border_color=None):
    x1, y1, x2, y2 = rect
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    if thickness < 0:
        cv2.rectangle(canvas, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(canvas, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(canvas, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(canvas, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(canvas, (x2 - radius, y2 - radius), radius, color, -1)
        if border_color:
            draw_round_rect(canvas, rect, border_color, radius, 1)
        return

    cv2.line(canvas, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)


def draw_header(canvas, title, subtitle, width):
    draw_round_rect(canvas, (18, 18, width - 18, 112), COLOR_PANEL, 18, -1, COLOR_BORDER)
    draw_round_rect(canvas, (36, 42, 76, 82), COLOR_PRIMARY_SOFT, 12, -1)
    cv2.circle(canvas, (56, 62), 11, COLOR_PRIMARY, -1, cv2.LINE_AA)
    cv2.circle(canvas, (56, 62), 4, COLOR_PANEL, -1, cv2.LINE_AA)
    cv2.putText(canvas, title, (94, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.78, COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (94, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_MUTED, 1, cv2.LINE_AA)


def draw_status_chip(canvas, text, rect, color, soft_color):
    draw_round_rect(canvas, rect, soft_color, 12, -1)
    x1, y1, _x2, _y2 = rect
    cv2.circle(canvas, (x1 + 18, y1 + 18), 5, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, text, (x1 + 32, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)


def draw_menu_button(canvas, button):
    x1, y1, x2, y2 = button["rect"]
    color = button.get("color", COLOR_PANEL)
    border = button.get("border", COLOR_BORDER)
    text_color = button.get("text_color", COLOR_PRIMARY)

    draw_round_rect(canvas, (x1, y1, x2, y2), color, 15, -1, border)
    draw_centered_text(
        canvas,
        button["label"],
        ((x1 + x2) // 2, (y1 + y2) // 2),
        button.get("font_scale", 0.58),
        text_color,
        button.get("thickness", 1),
    )


def button_at_position(buttons, x, y):
    for button in buttons:
        x1, y1, x2, y2 = button["rect"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return button["value"]
    return None


def field_at_position(fields, x, y):
    for index, field in enumerate(fields):
        x1, y1, x2, y2 = field["rect"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return index
    return None


def draw_input_field(canvas, field, value, active=False):
    x1, y1, x2, y2 = field["rect"]
    border = COLOR_PRIMARY if active else COLOR_BORDER

    cv2.putText(
        canvas,
        field["label"],
        (x1, y1 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_MUTED,
        1,
        cv2.LINE_AA,
    )
    draw_round_rect(canvas, (x1, y1, x2, y2), COLOR_PANEL, 13, -1, border)

    display_value = value if value else field["placeholder"]
    text_color = COLOR_TEXT if value else (170, 170, 176)
    if active:
        display_value = f"{display_value}|"

    cv2.putText(
        canvas,
        display_value[:34],
        (x1 + 14, y1 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        text_color,
        1,
        cv2.LINE_AA,
    )


def close_window(camera):
    camera.release()
    cv2.destroyAllWindows()


def register_customer(name, phone, discount_percent, api_url, camera_index=0):
    camera = open_camera(camera_index)
    face_label = slugify(name)
    window_name = "Register Customer Face"
    response_payload = None
    frame_count = 0
    gray_frame = None
    faces = []
    face_samples = []
    saved_paths = []
    status_message = "Ambil sampel wajah secara manual."
    action = {"value": None}
    buttons = [
        {
            "label": "Capture",
            "value": "capture",
            "rect": (315, 505, 425, 555),
            "color": COLOR_PRIMARY,
            "border": COLOR_PRIMARY,
            "text_color": (255, 255, 255),
        },
        {
            "label": "Simpan",
            "value": "save",
            "rect": (438, 505, 525, 555),
            "color": COLOR_ACCENT,
            "border": COLOR_ACCENT,
            "text_color": (255, 255, 255),
            "font_scale": 0.52,
        },
        {
            "label": "Balik",
            "value": "back",
            "rect": (540, 505, 620, 555),
            "border": COLOR_BORDER,
            "text_color": COLOR_MUTED,
            "font_scale": 0.52,
        },
    ]

    def on_mouse(event, x, y, _flags, _params):
        if event == cv2.EVENT_LBUTTONDOWN:
            action["value"] = button_at_position(buttons, x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % 3 == 1:
            gray_frame, faces = face_detection(frame)

        frame = cv2.copyMakeBorder(
            frame,
            0,
            90,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=COLOR_BG,
        )
        draw_round_rect(frame, (14, 492, 626, 568), COLOR_PANEL, 18, -1, COLOR_BORDER)
        chip_color = COLOR_ACCENT if len(faces) else COLOR_MUTED
        chip_soft = COLOR_ACCENT_SOFT if len(faces) else COLOR_SURFACE
        draw_status_chip(frame, f"{len(faces)} wajah terdeteksi", (22, 524, 208, 560), chip_color, chip_soft)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_PRIMARY, 2)

        cv2.putText(
            frame,
            "Register Wajah",
            (22, 516),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_TEXT,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{len(saved_paths)}/{REGISTER_SAMPLE_COUNT} sampel",
            (225, 526),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            COLOR_ACCENT if saved_paths else COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            status_message[:34],
            (225, 548),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

        for button in buttons:
            draw_menu_button(frame, button)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or action["value"] == "back":
            break

        if key == ord("c") or action["value"] == "capture":
            action["value"] = None
            selected_face = find_largest_face(faces)
            if selected_face is None or gray_frame is None:
                print("Wajah belum terdeteksi. Coba hadapkan wajah ke kamera.")
                status_message = "Wajah belum terdeteksi."
                continue

            x, y, w, h = selected_face
            face_roi = frame[y : y + h, x : x + w].copy()
            saved_path = save_face_sample(face_roi, face_label)
            face_samples.append(face_roi)
            saved_paths.append(saved_path)
            status_message = f"Sampel {len(saved_paths)} tersimpan."
            print(f"Sampel wajah tersimpan: {saved_path}")
            continue

        if key == ord("s") or action["value"] == "save":
            action["value"] = None
            if not face_samples:
                print("Belum ada sampel wajah. Capture minimal 1 sampel dulu.")
                status_message = "Capture minimal 1 sampel dulu."
                continue

            try:
                rebuild_embeddings()
            except RuntimeError as exc:
                status_message = "Embedding gagal dibuat. Lihat terminal."
                print(f"Gagal memperbarui embedding: {exc}")
                continue
            face_roi = face_samples[0]

            consent = input(
                "Apakah customer telah menyetujui penyimpanan data wajah? ketik YA untuk lanjut: "
            ).strip().upper() == "YA"
            if not consent:
                status_message = "Registrasi dibatalkan: persetujuan diperlukan."
                print(status_message)
                continue

            try:
                embedding = create_embedding(face_roi).tolist()
                response_payload = send_register_customer(
                    name,
                    phone,
                    discount_percent,
                    face_roi,
                    face_label,
                    embedding,
                    consent,
                    api_url,
                )
                print("Customer berhasil dikirim ke Laravel.")
                print(f"{len(saved_paths)} sampel wajah tersimpan lokal.")
                print(json.dumps(response_payload, indent=2))
            except RuntimeError as exc:
                print(f"{len(saved_paths)} sampel wajah tersimpan lokal, tapi gagal kirim ke Laravel.")
                print(exc)
            break

    close_window(camera)
    return response_payload


def run_recognition(camera_index=0):
    try:
        known_faces = load_known_faces(rebuild_if_needed=True)
    except RuntimeError as exc:
        print(f"Recognition tidak dapat dimulai: {exc}")
        return

    if not known_faces:
        print("Folder known_faces kosong. Register customer terlebih dahulu.")

    camera = open_camera(camera_index)
    window_name = "Pingkal Face Recognition"
    frame_count = 0
    detections = []
    liveness = MotionLivenessChallenge()
    action = {"value": None}
    buttons = [
        {
            "label": "Balik",
            "value": "back",
            "rect": (510, 505, 620, 555),
            "border": COLOR_BORDER,
            "text_color": COLOR_MUTED,
        },
    ]

    def on_mouse(event, x, y, _flags, _params):
        if event == cv2.EVENT_LBUTTONDOWN:
            action["value"] = button_at_position(buttons, x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % 4 == 1:
            gray_frame, faces = face_detection(frame)
            detections = []

            for x, y, w, h in faces:
                face_roi = frame[y : y + h, x : x + w]
                try:
                    label, score = recognize_face(face_roi, known_faces)
                except RuntimeError as exc:
                    print(f"Gagal mengenali wajah: {exc}")
                    label, score = "Unknown", 1.0
                liveness_passed = liveness.update((x, y, w, h))
                detections.append((x, y, w, h, label, score, liveness_passed))

        for x, y, w, h, label, score, liveness_passed in detections:
            color = COLOR_BLUE if label != "Unknown" else (90, 90, 96)
            text = f"{label} ({score:.3f})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            if label != "Unknown":
                notify_member_detected(label, score, liveness_passed)

        frame = cv2.copyMakeBorder(
            frame,
            0,
            90,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=COLOR_BG,
        )

        if not known_faces:
            draw_round_rect(frame, (14, 492, 626, 568), COLOR_PANEL, 18, -1, COLOR_BORDER)
            draw_status_chip(frame, "Belum ada data", (22, 524, 170, 560), COLOR_MUTED, COLOR_SURFACE)
            cv2.putText(
                frame,
                "Folder known_faces masih kosong",
                (22, 516),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                COLOR_MUTED,
                1,
                cv2.LINE_AA,
            )
        else:
            known_label = f"{known_person_count()} orang / {len(known_faces)} sampel"
            recognized_count = sum(1 for item in detections if item[4] != "Unknown")
            status_text = "Member dikenali" if recognized_count else "Scanning aktif"
            status_color = COLOR_ACCENT if recognized_count else COLOR_PRIMARY
            status_soft = COLOR_ACCENT_SOFT if recognized_count else COLOR_PRIMARY_SOFT
            draw_round_rect(frame, (14, 492, 626, 568), COLOR_PANEL, 18, -1, COLOR_BORDER)
            draw_status_chip(frame, status_text, (22, 524, 182, 560), status_color, status_soft)
            cv2.putText(
                frame,
                "Face Recognition",
                (22, 516),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_TEXT,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                known_label,
                (205, 548),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                COLOR_MUTED,
                1,
                cv2.LINE_AA,
            )

        for button in buttons:
            draw_menu_button(frame, button)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q") or action["value"] == "back":
            break

    close_window(camera)


def show_customer_form(title, subtitle, default_discount=0, initial_values=None):
    window_name = title
    selected = {"value": None}
    active_field = {"index": 0}
    values = {
        "name": (initial_values or {}).get("name", ""),
        "phone": (initial_values or {}).get("phone", ""),
        "discount": str((initial_values or {}).get("discount", default_discount)),
    }
    status = {"text": "Isi nama, lalu klik Mulai Register."}
    fields = [
        {
            "key": "name",
            "label": "Nama customer",
            "placeholder": "Contoh: Budi",
            "rect": (120, 205, 600, 258),
        },
        {
            "key": "phone",
            "label": "No. telepon",
            "placeholder": "Opsional",
            "rect": (120, 285, 600, 338),
        },
        {
            "key": "discount",
            "label": "Diskon (%)",
            "placeholder": "0",
            "rect": (120, 365, 600, 418),
        },
    ]
    buttons = [
        {
            "label": "Mulai register",
            "value": "start",
            "rect": (120, 455, 420, 510),
            "color": COLOR_PRIMARY,
            "border": COLOR_PRIMARY,
            "text_color": (255, 255, 255),
        },
        {
            "label": "Balik",
            "value": "back",
            "rect": (440, 455, 600, 510),
            "border": COLOR_BORDER,
            "text_color": COLOR_MUTED,
        },
    ]

    def on_mouse(event, x, y, _flags, _params):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        field_index = field_at_position(fields, x, y)
        if field_index is not None:
            active_field["index"] = field_index
            return

        selected["value"] = button_at_position(buttons, x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    while selected["value"] is None:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            selected["value"] = "back"
            break

        canvas = np.full((560, 720, 3), COLOR_BG, dtype=np.uint8)
        draw_header(canvas, title, subtitle, 720)
        draw_round_rect(canvas, (92, 124, 628, 530), COLOR_PANEL, 18, -1, COLOR_BORDER)
        draw_status_chip(canvas, "Form customer", (120, 132, 270, 168), COLOR_ACCENT, COLOR_ACCENT_SOFT)
        cv2.putText(
            canvas,
            status["text"],
            (120, 176),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

        for index, field in enumerate(fields):
            draw_input_field(canvas, field, values[field["key"]], active_field["index"] == index)

        for button in buttons:
            draw_menu_button(canvas, button)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord("q")):
            selected["value"] = "back"
            break

        if key in (9, 13):
            active_field["index"] = (active_field["index"] + 1) % len(fields)
            continue

        field = fields[active_field["index"]]
        field_key = field["key"]

        if key in (8, 127):
            values[field_key] = values[field_key][:-1]
            continue

        if 32 <= key <= 126:
            char = chr(key)
            if field_key == "discount" and not char.isdigit():
                continue
            values[field_key] = (values[field_key] + char)[:40]

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass

    if selected["value"] != "start":
        return None

    name = values["name"].strip()
    if not name:
        print("Nama customer wajib diisi.")
        return None

    try:
        discount = int(values["discount"].strip() or "0")
    except ValueError:
        discount = default_discount

    return {
        "name": name,
        "phone": values["phone"].strip(),
        "discount": discount,
    }


def show_register_form(default_discount=0):
    return show_customer_form(
        "Register Wajah",
        "Simpan data member dan wajah customer.",
        default_discount,
    )


def row_at_position(rows, x, y):
    for index, rect in enumerate(rows):
        x1, y1, x2, y2 = rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            return index
    return None


def show_member_data_list():
    window_name = "Data Wajah Member"
    action = {"value": None}
    selected = {"index": 0}
    scroll = {"offset": 0}
    confirm_delete = {"value": False}
    member_details = {}
    status = {"text": "Gunakan Wheel Mouse, Panah ↑↓, atau Tombol Naek/Turun."}
    visible_rows = 7
    buttons = [
        {
            "label": "Hapus",
            "value": "delete",
            "rect": (120, 555, 240, 610),
            "color": COLOR_PRIMARY,
            "border": COLOR_PRIMARY,
            "text_color": (255, 255, 255),
        },
        {
            "label": "Edit",
            "value": "edit",
            "rect": (250, 555, 370, 610),
            "color": COLOR_ACCENT,
            "border": COLOR_ACCENT,
            "text_color": (255, 255, 255),
        },
        {
            "label": "Balik",
            "value": "back",
            "rect": (480, 555, 600, 610),
            "border": COLOR_BORDER,
            "text_color": COLOR_MUTED,
        },
        {
            "label": "Naek",
            "value": "scroll_up",
            "rect": (610, 220, 665, 360),
            "border": COLOR_BORDER,
            "text_color": COLOR_PRIMARY,
        },
        {
            "label": "Turun",
            "value": "scroll_down",
            "rect": (610, 385, 665, 525),
            "border": COLOR_BORDER,
            "text_color": COLOR_PRIMARY,
        },
    ]
    row_rects = []

    def on_mouse(event, x, y, flags, _params):
        face_entries = list_known_face_entries()
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                selected["index"] = max(0, selected["index"] - 1)
            elif flags < 0:
                selected["index"] = min(max(0, len(face_entries) - 1), selected["index"] + 1)
            confirm_delete["value"] = False
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        row_index = row_at_position(row_rects, x, y)
        if row_index is not None:
            selected["index"] = scroll["offset"] + row_index
            confirm_delete["value"] = False
            return

        if confirm_delete["value"] and 260 <= x <= 460 and 555 <= y <= 610:
            action["value"] = "confirm_delete"
            return

        action["value"] = button_at_position(buttons, x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    for face_entry in list_known_face_entries():
        try:
            member_details[face_entry["label"]] = get_customer(face_entry["label"])["customer"]
        except (RuntimeError, KeyError):
            pass

    while action["value"] != "back":
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            action["value"] = "back"
            break

        face_entries = list_known_face_entries()
        if selected["index"] >= len(face_entries):
            selected["index"] = max(0, len(face_entries) - 1)
        if selected["index"] < scroll["offset"]:
            scroll["offset"] = selected["index"]
        if selected["index"] >= scroll["offset"] + visible_rows:
            scroll["offset"] = selected["index"] - visible_rows + 1
        scroll["offset"] = max(0, min(scroll["offset"], max(0, len(face_entries) - visible_rows)))

        canvas = np.full((640, 720, 3), COLOR_BG, dtype=np.uint8)
        draw_header(canvas, "Data Wajah", "List member yang tersimpan lokal.", 720)
        draw_round_rect(canvas, (92, 124, 680, 625), COLOR_PANEL, 18, -1, COLOR_BORDER)

        count_text = f"{len(face_entries)} orang" if not face_entries else f"Item {selected['index'] + 1} / {len(face_entries)}"
        draw_status_chip(canvas, count_text, (120, 132, 280, 168), COLOR_ACCENT, COLOR_ACCENT_SOFT)
        cv2.putText(
            canvas,
            status["text"],
            (120, 188),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

        row_rects = []
        if not face_entries:
            cv2.putText(
                canvas,
                "Belum ada wajah tersimpan.",
                (120, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                COLOR_MUTED,
                1,
                cv2.LINE_AA,
            )
        else:
            visible_entries = face_entries[scroll["offset"] : scroll["offset"] + visible_rows]
            for index, face_entry in enumerate(visible_entries):
                row_y = 220 + index * 45
                rect = (120, row_y, 590, row_y + 36)
                row_rects.append(rect)
                absolute_index = scroll["offset"] + index
                is_selected = absolute_index == selected["index"]
                row_color = COLOR_PRIMARY_SOFT if is_selected else COLOR_PANEL
                border_color = COLOR_PRIMARY if is_selected else COLOR_BORDER
                text_color = COLOR_PRIMARY if is_selected else COLOR_TEXT
                draw_round_rect(canvas, rect, row_color, 10, -1, border_color)
                customer = member_details.get(face_entry["label"], {})
                display_name = customer.get("name", display_face_label(face_entry["label"]))
                discount = customer.get("discount_percent")
                cv2.putText(
                    canvas,
                    f"{absolute_index + 1}. {display_name}",
                    (138, row_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"{discount}% | {len(face_entry['files'])} sampel" if discount is not None else f"{len(face_entry['files'])} sampel",
                    (445, row_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    COLOR_MUTED,
                    1,
                    cv2.LINE_AA,
                )

            # Gambar visual Scrollbar di samping list
            if len(face_entries) > visible_rows:
                track_x1, track_y1, track_x2, track_y2 = 596, 220, 604, 525
                draw_round_rect(canvas, (track_x1, track_y1, track_x2, track_y2), COLOR_SURFACE, 4, -1, COLOR_BORDER)
                track_h = track_y2 - track_y1
                thumb_h = max(25, int(track_h * (visible_rows / len(face_entries))))
                max_scroll = len(face_entries) - visible_rows
                scroll_ratio = scroll["offset"] / max_scroll if max_scroll > 0 else 0
                thumb_y1 = track_y1 + int(scroll_ratio * (track_h - thumb_h))
                thumb_y2 = thumb_y1 + thumb_h
                draw_round_rect(canvas, (track_x1, thumb_y1, track_x2, thumb_y2), COLOR_PRIMARY, 4, -1, COLOR_PRIMARY)

        if confirm_delete["value"] and face_entries:
            selected_name = display_face_label(face_entries[selected["index"]]["label"])
            draw_round_rect(canvas, (260, 555, 460, 610), COLOR_ACCENT, 15, -1, COLOR_ACCENT)
            draw_centered_text(canvas, "Yakin", (360, 582), 0.58, (255, 255, 255), 1)
            cv2.putText(
                canvas,
                f"Hapus {selected_name[:18]}?",
                (120, 535),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                COLOR_PRIMARY,
                1,
                cv2.LINE_AA,
            )
        else:
            draw_round_rect(canvas, (260, 555, 460, 610), COLOR_SURFACE, 15, -1, COLOR_BORDER)
            draw_centered_text(canvas, "Pilih", (360, 582), 0.58, COLOR_MUTED, 1)

        for button in buttons:
            draw_menu_button(canvas, button)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30)
        raw_key = key
        char_key = key & 0xFF if key != -1 else -1

        if char_key in (ord("q"), 27):
            action["value"] = "back"
            break

        # Up Key / 'w' / 'W'
        if (raw_key in (2490368, 38, 82) or char_key in (ord("w"), ord("W"), 82)) and face_entries:
            selected["index"] = max(0, selected["index"] - 1)
            confirm_delete["value"] = False
            continue

        # Down Key / 's' / 'S'
        if (raw_key in (2621440, 40, 84) or char_key in (ord("s"), ord("S"), 84)) and face_entries:
            selected["index"] = min(len(face_entries) - 1, selected["index"] + 1)
            confirm_delete["value"] = False
            continue

        # Page Up
        if (raw_key in (2162688, 33) or char_key == 33) and face_entries:
            selected["index"] = max(0, selected["index"] - visible_rows)
            confirm_delete["value"] = False
            continue

        # Page Down
        if (raw_key in (2228224, 34) or char_key == 34) and face_entries:
            selected["index"] = min(len(face_entries) - 1, selected["index"] + visible_rows)
            confirm_delete["value"] = False
            continue

        if char_key in (ord("d"), 127, 8):
            action["value"] = "delete"
        if char_key in (ord("e"), ord("E")):
            action["value"] = "edit"
        if char_key in (13, ord("y")) and confirm_delete["value"]:
            action["value"] = "confirm_delete"

        if action["value"] == "scroll_up":
            action["value"] = None
            if face_entries:
                selected["index"] = max(0, selected["index"] - 1)
                confirm_delete["value"] = False
            continue

        if action["value"] == "scroll_down":
            action["value"] = None
            if face_entries:
                selected["index"] = min(len(face_entries) - 1, selected["index"] + 1)
                confirm_delete["value"] = False
            continue

        if action["value"] == "delete":
            action["value"] = None
            if not face_entries:
                status["text"] = "Tidak ada data yang bisa dihapus."
                continue
            confirm_delete["value"] = not confirm_delete["value"]
            status["text"] = "Klik Yakin atau tekan Enter untuk hapus." if confirm_delete["value"] else "Hapus dibatalkan."
            continue

        if action["value"] == "edit":
            action["value"] = None
            if not face_entries:
                status["text"] = "Tidak ada data yang bisa diedit."
                continue

            entry = face_entries[selected["index"]]
            try:
                customer = get_customer(entry["label"])["customer"]
            except (RuntimeError, KeyError) as exc:
                status["text"] = "Data Laravel tidak ditemukan. Register ulang customer ini."
                print(f"Gagal mengambil data customer: {exc}")
                continue

            edited = show_customer_form(
                "Edit Member",
                "Face label tidak dapat diubah agar wajah tetap cocok.",
                customer.get("discount_percent", 0),
                {
                    "name": customer.get("name", ""),
                    "phone": customer.get("phone") or "",
                    "discount": customer.get("discount_percent", 0),
                },
            )
            if edited is None:
                status["text"] = "Edit dibatalkan."
                continue

            try:
                response = update_customer(
                    entry["label"],
                    edited["name"],
                    edited["phone"],
                    edited["discount"],
                )
                member_details[entry["label"]] = response["customer"]
                status["text"] = "Nama dan diskon member berhasil diperbarui."
            except (RuntimeError, KeyError) as exc:
                status["text"] = "Gagal menyimpan perubahan ke Laravel."
                print(f"Gagal mengubah customer: {exc}")
            continue

        if action["value"] == "confirm_delete":
            action["value"] = None
            if confirm_delete["value"] and face_entries:
                deleted_name = display_face_label(face_entries[selected["index"]]["label"])
                delete_known_face(face_entries[selected["index"]])
                confirm_delete["value"] = False
                status["text"] = f"{deleted_name} sudah dihapus."
            continue

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def show_visual_menu():
    window_name = "Pingkal Face Menu"
    selected = {"value": None}
    buttons = [
        {
            "label": "Recognize",
            "value": "1",
            "rect": (120, 150, 600, 205),
            "border": COLOR_BORDER,
            "text_color": COLOR_PRIMARY,
        },
        {
            "label": "Register Wajah",
            "value": "2",
            "rect": (120, 225, 600, 280),
            "color": COLOR_PRIMARY,
            "border": COLOR_PRIMARY,
            "text_color": (255, 255, 255),
        },
        {
            "label": "List Data",
            "value": "3",
            "rect": (120, 300, 600, 355),
            "border": COLOR_BORDER,
            "text_color": COLOR_PRIMARY,
        },
        {
            "label": "Exit",
            "value": "0",
            "rect": (120, 375, 600, 430),
            "border": COLOR_BORDER,
            "text_color": COLOR_MUTED,
        },
    ]

    def on_mouse(event, x, y, _flags, _params):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected["value"] = button_at_position(buttons, x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    while selected["value"] is None:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            selected["value"] = "0"
            break

        canvas = np.full((515, 720, 3), COLOR_BG, dtype=np.uint8)
        draw_header(canvas, "Pingkal Face", "Kasir member recognition", 720)
        draw_round_rect(canvas, (92, 124, 628, 450), COLOR_PANEL, 18, -1, COLOR_BORDER)
        draw_status_chip(canvas, f"{known_person_count()} orang", (120, 466, 250, 502), COLOR_ACCENT, COLOR_ACCENT_SOFT)

        for button in buttons:
            draw_menu_button(canvas, button)

        cv2.putText(
            canvas,
            "Kamera default: index 0",
            (452, 489),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("1"), ord("2"), ord("3"), ord("0")):
            selected["value"] = chr(key)
        elif key in (ord("q"), 27):
            selected["value"] = "0"

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass
    return selected["value"]


def run_terminal_menu(args):
    while True:
        choice = show_visual_menu()

        if choice == "1":
            run_recognition(args.camera_index)
            continue

        if choice == "2":
            form_data = show_register_form(args.discount)
            if form_data is None:
                continue

            register_customer(
                form_data["name"],
                form_data["phone"],
                form_data["discount"],
                args.api_url,
                args.camera_index,
            )
            continue

        if choice == "3":
            show_member_data_list()
            continue

        if choice == "0":
            print("Keluar.")
            return

        print("Menu tidak valid.")
