"""Sign-in: remember me, session expiry, lockout, reset requests, recovery."""

import time
from types import SimpleNamespace

import pytest

from homeshield.auth import REMEMBER_DAYS, LoginThrottle, UserStore
from homeshield.cameras import CameraManager
from homeshield.db import write_conn
from homeshield.server import create_app


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("HOMESHIELD_SECRET", raising=False)
    db = str(tmp_path / "hs.db")
    app = create_app(db_path=db, snapshot_dir=str(tmp_path / "s"),
                     person_photos_dir=str(tmp_path / "p"),
                     intruder_photos_dir=str(tmp_path / "i"), auto_start=False)
    with write_conn(db) as conn:
        conn.execute("UPDATE users SET must_change = 0")
    app.config["DB"] = db
    return app


def _login(client, password="admin", remember=False, username="admin"):
    return client.post("/api/login", json={"username": username, "password": password,
                                           "remember": remember})


def _session_cookie(resp):
    return next(h for h in resp.headers.getlist("Set-Cookie") if h.startswith("session="))


def test_remember_me_sets_a_lasting_cookie_and_plain_sign_in_does_not(app):
    remembered = _session_cookie(_login(app.test_client(), remember=True))
    assert "Expires=" in remembered
    plain = _session_cookie(_login(app.test_client(), remember=False))
    assert "Expires=" not in plain                   # gone when the browser closes


def test_sessions_stop_working_when_their_time_is_up(app):
    c = app.test_client()
    _login(c)
    assert c.get("/api/me").status_code == 200
    with c.session_transaction() as s:
        assert s["exp"] - time.time() <= 12 * 3600 + 5
        s["exp"] = time.time() - 1
    assert c.get("/api/me").status_code == 401
    assert c.get("/api/status").status_code == 401

    r = app.test_client()
    _login(r, remember=True)
    with r.session_transaction() as s:
        assert s["exp"] - time.time() > (REMEMBER_DAYS - 1) * 86400


def test_wrong_passwords_lock_the_account_for_a_while(app):
    c = app.test_client()
    for _ in range(4):
        assert _login(c, password="nope").status_code == 401
    locked = _login(c, password="nope")
    assert locked.status_code == 429 and locked.get_json()["retry_after"] > 200
    assert _login(c, password="admin").status_code == 429   # even the right one, for now


def test_throttle_clears_after_success_and_counts_per_address():
    t = LoginThrottle()
    for _ in range(4):
        assert t.failed("Nadia", "10.0.0.2") == 0
    t.succeeded("nadia", "10.0.0.2")
    assert t.failed("nadia", "10.0.0.2") == 0        # count started again
    for i in range(20):
        t.failed(f"user{i}", "10.0.0.9")
    assert t.retry_after("someone-else", "10.0.0.9") > 0
    assert t.retry_after("someone-else", "10.0.0.3") == 0


def test_sign_in_reports_the_previous_sign_in(app):
    first = _login(app.test_client()).get_json()
    assert first["last_login"] is None
    second = _login(app.test_client()).get_json()
    assert second["last_login"]["at"] > 0 and second["last_login"]["ip"] == "127.0.0.1"


def test_reset_request_is_silent_about_accounts_and_shows_up_for_admins(app):
    admin = app.test_client()
    _login(admin)
    admin.post("/api/users", json={"username": "nadia", "password": "nadia1",
                                   "role": "guest", "must_change_password": False})
    anon = app.test_client()
    assert anon.post("/api/password_reset_request", json={"username": "ghost"}).get_json() == {"ok": True}
    assert anon.post("/api/password_reset_request", json={"username": "nadia"}).get_json() == {"ok": True}

    def nadia():
        return next(u for u in admin.get("/api/users").get_json()["users"] if u["username"] == "nadia")
    assert nadia()["reset_requested_at"]
    assert next(u for u in admin.get("/api/users").get_json()["users"]
                if u["username"] == "admin")["last_login_at"]

    admin.post(f"/api/users/{nadia()['user_id']}/password", json={"password": "temp1234"})
    assert nadia()["reset_requested_at"] is None and nadia()["must_change_password"] is True


def test_reset_requests_are_rate_limited(app):
    c = app.test_client()
    codes = [c.post("/api/password_reset_request", json={"username": "admin"}).status_code
             for _ in range(6)]
    assert codes[:4] == [200] * 4 and codes[-1] == 429


def test_session_key_survives_a_restart(app, tmp_path):
    again = create_app(db_path=app.config["DB"], snapshot_dir=str(tmp_path / "s"),
                       person_photos_dir=str(tmp_path / "p"),
                       intruder_photos_dir=str(tmp_path / "i"), auto_start=False)
    assert again.secret_key == app.secret_key
    assert (tmp_path / "homeshield_secret.key").exists()


def test_console_recovery_sets_a_temporary_password(app):
    store = UserStore(app.config["DB"])
    assert store.set_temporary_password("nobody") is None
    temp = store.set_temporary_password("admin")
    assert temp and store.verify("admin", temp) is not None
    assert store.get_by_username("admin")["must_change"] == 1
    assert store.verify("admin", "admin") is None


def test_wrong_password_reply_counts_down_the_tries_left(app):
    c = app.test_client()
    lefts = [_login(c, password="nope").get_json()["attempts_left"] for _ in range(4)]
    assert lefts == [4, 3, 2, 1]
    ghost = [_login(c, username="ghost", password="x").get_json()["attempts_left"] for _ in range(2)]
    assert ghost == [4, 3]                      # unknown accounts answer the same way


def test_new_password_must_differ_from_old_password_and_username(app):
    db = app.config["DB"]
    with write_conn(db) as conn:
        conn.execute("UPDATE users SET must_change = 1")
    c = app.test_client()
    _login(c)
    assert "username" in c.post("/api/change_password", json={"new_password": "ADMIN"}).get_json()["error"]
    UserStore(db).update_password(1, "old-pass1")
    with write_conn(db) as conn:
        conn.execute("UPDATE users SET must_change = 1")
    c2 = app.test_client()
    _login(c2, password="old-pass1")
    r = c2.post("/api/change_password", json={"new_password": "old-pass1"})
    assert r.status_code == 400 and "you have now" in r.get_json()["error"]
    assert c2.post("/api/change_password", json={"new_password": "fresh-pass2"}).status_code == 200


def test_setup_state_tells_a_guest_only_whether_cameras_are_watched(app):
    state = app.test_client().get("/api/setup_state").get_json()
    assert set(state) == {"default_admin", "monitoring"}
    assert state["monitoring"] is False                  # not started in tests


def test_watching_needs_a_running_manager_and_a_connected_camera():
    m = CameraManager.__new__(CameraManager)
    m._running, m._workers = False, {}
    assert not m.monitoring()
    cam = SimpleNamespace(status=SimpleNamespace(camera_connected=False))
    m._running, m._workers = True, {1: cam}
    assert not m.monitoring()                            # running, but nothing to watch
    cam.status.camera_connected = True
    assert m.monitoring()
    m._running = False
    assert not m.monitoring()


def test_the_page_itself_renders(app):
    # The page is a Jinja template: "{#" or "{{" in its CSS or JS breaks every load.
    r = app.test_client().get("/")
    assert r.status_code == 200 and b"authShell" in r.data


def test_the_page_is_plain_html_gzipped_and_revalidated(app):
    import gzip
    c = app.test_client()
    plain = c.get("/")
    assert b"__bundler" not in plain.data          # no self-unpacking bundle
    assert plain.headers["Cache-Control"] == "no-cache" and plain.headers["ETag"]

    zipped = c.get("/", headers={"Accept-Encoding": "gzip"})
    assert zipped.headers["Content-Encoding"] == "gzip"
    assert "Accept-Encoding" in zipped.headers["Vary"]
    assert gzip.decompress(zipped.data) == plain.data
    assert len(zipped.data) < len(plain.data) / 3

    again = c.get("/", headers={"If-None-Match": plain.headers["ETag"]})
    assert again.status_code == 304 and again.data == b""


def test_small_replies_are_not_gzipped(app):
    r = app.test_client().get("/api/setup_state", headers={"Accept-Encoding": "gzip"})
    assert "Content-Encoding" not in r.headers


def test_fonts_come_from_homeshield_not_the_internet(app):
    c = app.test_client()
    page = c.get("/").data
    assert b"fonts.googleapis.com" not in page and b"fonts.gstatic.com" not in page
    for name in ("chakra-petch-400-latin", "chakra-petch-600-latin",
                 "chakra-petch-700-latin", "rubik-latin"):
        assert f"/static/fonts/{name}.woff2".encode() in page
        r = c.get(f"/static/fonts/{name}.woff2")
        assert r.status_code == 200 and r.data[:4] == b"wOF2"
        r.close()


def test_console_reset_runs_without_starting_the_dashboard(app):
    # reset-password.bat calls this; it must not need the camera libraries.
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable, "-X", "importtime", "run_homeshield.py",
                          "--reset-password", "admin", "--db", app.config["DB"]],
                         cwd=root, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr[-2000:]
    assert "Temporary password for admin: " in run.stdout
    assert "homeshield.server" not in run.stderr          # importtime lists every module loaded
    missing = subprocess.run([sys.executable, "run_homeshield.py", "--reset-password", "ghost",
                              "--db", app.config["DB"]], cwd=root, capture_output=True, text=True,
                             timeout=60)
    assert missing.returncode == 1 and "No user called 'ghost'" in missing.stdout
