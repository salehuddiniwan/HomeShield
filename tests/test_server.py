"""API validation: bad input is answered with a 400 and a readable message."""

import pytest

from homeshield.db import write_conn
from homeshield.server import create_app


@pytest.fixture
def app(tmp_path):
    db = str(tmp_path / "hs.db")
    app = create_app(db_path=db,
                     snapshot_dir=str(tmp_path / "snapshots"),
                     person_photos_dir=str(tmp_path / "persons"),
                     intruder_photos_dir=str(tmp_path / "intruders"),
                     auto_start=False)
    with write_conn(db) as conn:      # skip the forced first-login password change
        conn.execute("UPDATE users SET must_change = 0")
    return app


@pytest.fixture
def admin(app):
    c = app.test_client()
    r = c.post("/api/login", json={"username": "admin", "password": "admin"})
    assert r.status_code == 200
    return c


def _add_camera(client, **body):
    return client.post("/api/cameras", json=body)


def test_add_camera_requires_a_source(admin):
    for url in ("", "   ", '""'):
        r = _add_camera(admin, name="Door", url=url)
        assert r.status_code == 400
        assert "source" in r.get_json()["error"]
    assert admin.get("/api/cameras").get_json() == []


def test_add_camera_strips_quotes_and_limits_lengths(admin):
    r = _add_camera(admin, name="  Door  ", url='"0"', location="Hall")
    assert r.status_code == 200
    cam = admin.get("/api/cameras").get_json()[0]
    assert (cam["name"], cam["url"], cam["location"]) == ("Door", "0", "Hall")

    assert _add_camera(admin, name="x" * 61, url="0").status_code == 400
    assert _add_camera(admin, name="Door", url="rtsp://" + "a" * 500).status_code == 400
    assert _add_camera(admin, name="Door", url="0", location="y" * 61).status_code == 400


def test_zone_polygon_is_validated_and_clamped(admin):
    _add_camera(admin, name="Door", url="0")
    cid = admin.get("/api/cameras").get_json()[0]["camera_id"]

    def post(polygon, camera_id=cid, name="Stove"):
        return admin.post("/api/zones", json={"zone_name": name, "camera_id": camera_id,
                                              "polygon": polygon, "zone_type": "danger"})

    assert post([[0, 0], [10, 10]]).status_code == 400                 # too few points
    assert post([[0, 0], [10, "a"], [5, 5]]).status_code == 400          # not a number
    assert post("0,0 10,10 5,5").status_code == 400                      # not a list
    assert post([[0, 0], [1, 1], [2, 2]] * 30).status_code == 400        # too many points
    assert post([[0, 0], [10, 10], [5, 5]], camera_id=cid + 99).status_code == 400
    assert post([[0, 0], [10, 10], [5, 5]], name="z" * 61).status_code == 400

    r = post([[-20, 5], [900, 10], [320, 700]])
    assert r.status_code == 200
    zone = admin.get("/api/zones").get_json()[0]
    assert zone["polygon"] == [[0.0, 5.0], [640.0, 10.0], [320.0, 480.0]]


def test_person_category_and_name_are_checked_before_capture(admin):
    r = admin.post("/api/persons", json={"name": "Aisyah", "category": "pet", "camera_id": 1})
    assert r.status_code == 400 and "Category" in r.get_json()["error"]
    r = admin.post("/api/persons", json={"name": "n" * 61, "category": "adult", "camera_id": 1})
    assert r.status_code == 400 and "too long" in r.get_json()["error"]


def test_guest_may_fetch_still_frames(app, admin):
    r = admin.post("/api/users", json={"username": "viewer", "password": "viewer1",
                                       "role": "guest", "must_change_password": False})
    assert r.status_code == 200
    guest = app.test_client()
    assert guest.post("/api/login", json={"username": "viewer",
                                          "password": "viewer1"}).status_code == 200
    # No such camera: 404, not 401/403.
    assert guest.get("/frame_snap/1").status_code == 404
    assert guest.post("/api/cameras", json={"url": "0"}).status_code == 403


def test_guests_never_see_camera_sources(app, admin):
    _add_camera(admin, name="Gate", url="rtsp://admin:secret@192.168.1.9:554/stream1")
    admin.post("/api/users", json={"username": "viewer", "password": "viewer1",
                                   "role": "guest", "must_change_password": False})
    guest = app.test_client()
    guest.post("/api/login", json={"username": "viewer", "password": "viewer1"})
    cams = guest.get("/api/cameras").get_json()
    assert cams and all("url" not in c for c in cams)
    status = guest.get("/api/status").get_json()
    assert all("url" not in c for c in status["cameras"].values())
    assert "secret" in admin.get("/api/cameras").get_data(as_text=True)   # admins still manage them


def test_describe_source_drops_credentials():
    from homeshield.cameras import describe_source
    assert describe_source("rtsp://admin:secret@192.168.1.9:554/stream1?x=1") == "rtsp://192.168.1.9:554"
    assert describe_source("0") == "webcam 0"
    assert describe_source("http://cam.local/video") == "http://cam.local"


def test_setup_state_reports_the_default_login_only_while_it_works(tmp_path):
    db = str(tmp_path / "fresh.db")
    app = create_app(db_path=db, snapshot_dir=str(tmp_path / "s"),
                     person_photos_dir=str(tmp_path / "p"),
                     intruder_photos_dir=str(tmp_path / "i"), auto_start=False)
    c = app.test_client()
    assert c.get("/api/setup_state").get_json()["default_admin"] is True
    c.post("/api/login", json={"username": "admin", "password": "admin"})
    assert c.post("/api/change_password", json={"new_password": "better-one"}).status_code == 200
    assert c.get("/api/setup_state").get_json()["default_admin"] is False
