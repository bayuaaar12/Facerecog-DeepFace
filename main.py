import argparse
from config import DEFAULT_API_URL
from gui import (
    register_customer,
    run_recognition,
    run_terminal_menu,
    show_member_data_list,
)
from recognition import default_camera_index
from storage import rebuild_embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    default_index = default_camera_index()
    parser.add_argument(
        "--mode",
        choices=["recognize", "register", "manage", "rebuild-embeddings"],
        help="recognize, register, manage, atau rebuild-embeddings",
    )
    parser.add_argument("--name", help="Nama customer saat mode register")
    parser.add_argument("--phone", default="", help="Nomor telepon customer")
    parser.add_argument("--discount", type=int, default=0, help="Diskon member persen")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="URL API Laravel")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=default_index,
        help=f"Index kamera OpenCV, default di perangkat ini: {default_index}",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode is None:
        run_terminal_menu(args)
        return

    if args.mode == "register":
        if not args.name:
            raise RuntimeError("Mode register butuh --name.")
        register_customer(args.name, args.phone, args.discount, args.api_url, args.camera_index)
        return

    if args.mode == "manage":
        show_member_data_list()
        return

    if args.mode == "rebuild-embeddings":
        try:
            rebuild_embeddings()
        except RuntimeError as exc:
            print(f"Gagal membangun ulang embedding: {exc}")
        return

    run_recognition(args.camera_index)


if __name__ == "__main__":
    main()
