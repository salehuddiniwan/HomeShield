"""
HomeShield entry point.

Examples
--------
# Default: starts on http://localhost:5000/. Cameras you add via the UI
# are persisted in homeshield.db and reloaded next launch.
python run_homeshield.py

# Custom port and DB
python run_homeshield.py --port 8080 --db custom.db

# Bind only to localhost (default is 0.0.0.0 so phones can hit it)
python run_homeshield.py --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from homeshield.server import create_app


def lan_ip() -> "str | None":
    """Best-effort detection of this machine's LAN IP address.

    Opens a UDP socket toward a public address (no packets are actually
    sent) so the OS picks the interface used for outbound traffic.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return None
    finally:
        s.close()


def main():
    p = argparse.ArgumentParser(description="HomeShield dashboard")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--db", default="homeshield.db")
    p.add_argument("--snapshots", default="snapshots")
    p.add_argument("--person-photos", default="person_photos")
    p.add_argument("--intruder-photos", default="intruder_photos")
    p.add_argument("--no-autostart", action="store_true",
                   help="Don't auto-start cameras on boot (useful for debugging)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    app = create_app(
        db_path=args.db,
        snapshot_dir=args.snapshots,
        person_photos_dir=args.person_photos,
        intruder_photos_dir=args.intruder_photos,
        auto_start=not args.no_autostart,
    )

    print(f"\n[HomeShield] This computer : http://localhost:{args.port}/")
    if args.host in ("0.0.0.0", "::"):
        ip = lan_ip()
        if ip:
            print(f"[HomeShield] Other devices : http://{ip}:{args.port}/")
            print("[HomeShield]   ^ open THIS url on your phone/laptop "
                  "(NOT 0.0.0.0).")
        else:
            print("[HomeShield] Other devices : could not detect LAN IP; "
                  "run `ipconfig` and use this PC's IPv4 address.")
        print("[HomeShield]   If it still won't load, allow Python through "
              "the Windows Firewall (see notes).")
    else:
        print(f"[HomeShield] Bound to {args.host} only "
              "(other devices cannot reach it).")
    print(f"[HomeShield] Database : {Path(args.db).resolve()}")
    print(f"[HomeShield] Snapshots: {Path(args.snapshots).resolve()}\n")

    app.run(host=args.host, port=args.port, threaded=True,
            debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
