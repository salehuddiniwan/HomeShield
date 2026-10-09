"""Incidents: grouping, escalation, resolution, upgrade, and per-incident WhatsApp."""

import time

import pytest

from homeshield.db import init_db, write_conn
from homeshield.events import Event, EventBus
from homeshield.incidents import INCIDENT_GAP_S, UPGRADE_USER, IncidentStore
from homeshield.notify import WhatsAppNotifier, compose, parse_phones
from homeshield.server import create_app

T0 = 1_800_000_000.0


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "hs.db")
    init_db(path)
    return path


@pytest.fixture
def bus(db):
    return EventBus(db_path=db)


def emit(bus, etype, ts, cam=1, conf=0.6, snap=None):
    """Publish synchronously (the background thread is bypassed)."""
    ev = Event(event_type=etype, ts=ts, camera_id=cam, camera_name=f"Cam {cam}",
               confidence=conf, snapshot_path=snap)
    bus._do_publish(ev, None)
    return ev


def test_bursts_group_until_a_quiet_gap(bus, db):
    for i in range(10):                       # fire every 5 s for 45 s
        emit(bus, "fire_detected", T0 + i * 5, conf=0.35 + i * 0.05)
    later = emit(bus, "fire_detected", T0 + 45 + INCIDENT_GAP_S + 1)

    incs = IncidentStore(db).list()
    assert len(incs) == 2
    first = incs[-1]
    assert first["event_count"] == 10
    assert first["peak_confidence"] == pytest.approx(0.8)
    assert first["duration_s"] == pytest.approx(45)
    assert later.incident["change"] == "opened"


def test_cameras_and_kinds_are_separate_and_normal_events_never_group(bus, db):
    emit(bus, "fire_detected", T0, cam=1)
    emit(bus, "fire_detected", T0 + 1, cam=2)
    emit(bus, "intruder_detected", T0 + 2, cam=1)
    sys_ev = emit(bus, "system", T0 + 3, cam=1)
    assert sys_ev.incident is None
    assert len(IncidentStore(db).list()) == 3


def test_fall_stages_form_one_incident_headlined_by_the_worst(bus, db):
    emit(bus, "fall_detected", T0, conf=0.7)
    emit(bus, "lying_motionless", T0 + 30, conf=0.6)
    emit(bus, "inactivity", T0 + 60, conf=0.6)
    (inc,) = IncidentStore(db).list()
    assert inc["kind"] == "fall"
    assert inc["headline_type"] == "lying_motionless"
    assert inc["severity"] == "critical"
    assert inc["event_count"] == 3


def test_after_acknowledging_new_events_join_quietly(bus, db):
    store = IncidentStore(db)
    first = emit(bus, "fire_detected", T0)
    store.resolve(first.incident_id, outcome="acknowledged", by="aisyah")
    more = emit(bus, "fire_detected", T0 + 10)
    assert more.incident["change"] == "updated"
    inc = store.get(first.incident_id)
    assert inc["status"] == "acknowledged" and inc["event_count"] == 2


def test_escalation_reopens_a_closed_incident_and_keeps_who_closed_it(bus, db):
    store = IncidentStore(db)
    fall = emit(bus, "fall_detected", T0)
    store.resolve(fall.incident_id, outcome="acknowledged", by="aisyah", note="checked")
    worse = emit(bus, "lying_motionless", T0 + 40)
    assert worse.incident["change"] == "escalated"
    inc = store.get(fall.incident_id)
    assert inc["status"] == "open" and inc["escalated"] is True
    assert inc["resolved_by"] == "aisyah"


def test_peak_snapshot_follows_the_most_confident_event(bus, db):
    emit(bus, "fire_detected", T0, conf=0.4, snap="a.jpg")
    emit(bus, "fire_detected", T0 + 5, conf=0.9, snap="b.jpg")
    emit(bus, "fire_detected", T0 + 10, conf=0.5, snap="c.jpg")
    (inc,) = IncidentStore(db).list()
    assert inc["snapshot_path"] == "b.jpg"
    detail = IncidentStore(db).get(inc["incident_id"], with_events=True)
    assert [e["snapshot_path"] for e in detail["events"]] == ["a.jpg", "b.jpg", "c.jpg"]


def test_board_puts_open_first_and_counts_critical(bus, db):
    store = IncidentStore(db)
    a = emit(bus, "zone_entry", T0, cam=1)
    emit(bus, "fire_detected", T0 + 1, cam=2)
    c = emit(bus, "intruder_detected", T0 + 2, cam=3)
    store.resolve(c.incident_id, outcome="false_alarm", by="sam")
    board = store.board()
    assert [i["kind"] for i in board["open"]] == ["fire", "zone"]   # critical first
    assert board["open_total"] == 2 and board["open_critical"] == 1
    assert [i["incident_id"] for i in board["recent"]] == [c.incident_id]
    assert store.open_counts() == {"open_incidents": 2, "open_critical": 1}
    assert a.incident["severity"] == "attention"


def test_stats_count_false_alarms_per_detector_and_camera(bus, db):
    store = IncidentStore(db)
    now = time.time()
    x = emit(bus, "fire_detected", now - 100, cam=1)
    emit(bus, "fire_detected", now - 100, cam=2)
    store.resolve(x.incident_id, outcome="false_alarm", by="sam")
    stats = store.stats(days=7)
    fire = next(k for k in stats["by_kind"] if k["key"] == "fire")
    assert (fire["total"], fire["false_alarm"], fire["open"]) == (2, 1, 1)
    assert {c["key"] for c in stats["by_camera"]} == {"Cam 1", "Cam 2"}


def test_backfill_groups_old_events_and_closes_stale_ones(db):
    now = time.time()
    with write_conn(db) as conn:
        rows = [(now - 7200, "fire_detected"), (now - 7195, "fire_detected"),
                (now - 60, "fall_detected"), (now - 50, "system")]
        for ts, etype in rows:
            conn.execute(
                "INSERT INTO events (ts, created_at, event_type, camera_id, "
                "camera_name, confidence) VALUES (?, '', ?, 1, 'Hall', 0.7)",
                (ts, etype))
    store = IncidentStore(db)
    assert store.backfill() == 2
    assert store.backfill() == 0                     # runs once per event
    by_kind = {i["kind"]: i for i in store.list()}
    assert by_kind["fire"]["status"] == "acknowledged"
    assert by_kind["fire"]["resolved_by"] == UPGRADE_USER
    assert by_kind["fall"]["status"] == "open"       # recent: still needs a look


def test_a_late_old_event_does_not_join_a_newer_incident(bus, db):
    emit(bus, "fire_detected", T0)
    old = emit(bus, "fire_detected", T0 - 2 * 86400)
    early = emit(bus, "fire_detected", T0 - 30)          # within the gap: joins
    assert old.incident["change"] == "opened"
    assert early.incident["change"] == "updated"
    assert early.incident["started_ts"] == T0 - 30
    assert len(IncidentStore(db).list()) == 2


def test_clear_removes_incidents_too(bus, db):
    emit(bus, "fire_detected", T0)
    bus.clear()
    assert IncidentStore(db).list() == []


# ---- WhatsApp ---------------------------------------------------------------

class _Settings(dict):
    def get(self, k, default=None):
        return super().get(k, default)


def test_parse_phones_cleans_and_dedupes():
    raw = "+60 12-345 6789, whatsapp:+60198765432;\n+60123456789, 0123, hello"
    assert parse_phones(raw) == ["+60123456789", "+60198765432"]


def test_whatsapp_goes_out_once_per_incident_and_on_escalation(bus, monkeypatch):
    for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM"):
        monkeypatch.setenv(k, "x")
    sent = []
    notifier = WhatsAppNotifier(_Settings(alert_phones="+60123456789"),
                                sender=lambda to, body: sent.append((to, body)))
    bus.add_incident_listener(notifier.on_incident)

    for i in range(12):                          # one fire, 12 events
        emit(bus, "fire_detected", T0 + i * 5, cam=1)
    emit(bus, "fall_detected", T0, cam=2)
    emit(bus, "lying_motionless", T0 + 30, cam=2)  # escalation

    deadline = time.time() + 3
    while len(sent) < 3 and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.1)
    assert len(sent) == 3
    assert "Fire or smoke detected on Cam 1" in sent[0][1]
    assert sent[2][1].startswith("HomeShield - ESCALATED: Person lying motionless")


def test_whatsapp_stays_off_without_credentials(bus, monkeypatch):
    for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM"):
        monkeypatch.delenv(k, raising=False)
    sent = []
    notifier = WhatsAppNotifier(_Settings(alert_phones="+60123456789"),
                                sender=lambda to, body: sent.append(to))
    notifier.on_incident("opened", {"headline_type": "fire_detected"})
    time.sleep(0.1)
    assert sent == [] and notifier.status()["ready"] is False


def test_compose_names_what_where_and_how_sure():
    msg = compose("opened", {"headline_type": "zone_entry", "camera_name": "Kitchen",
                             "last_ts": T0, "peak_confidence": 0.874})
    assert "Child entered a danger zone on Kitchen" in msg and "87%" in msg


# ---- API ----------------------------------------------------------------------

@pytest.fixture
def app(tmp_path):
    path = str(tmp_path / "api.db")
    app = create_app(db_path=path, snapshot_dir=str(tmp_path / "s"),
                     person_photos_dir=str(tmp_path / "p"),
                     intruder_photos_dir=str(tmp_path / "i"), auto_start=False)
    with write_conn(path) as conn:
        conn.execute("UPDATE users SET must_change = 0")
    app.config["DB"] = path
    return app


def _login(app, user, pw):
    c = app.test_client()
    assert c.post("/api/login", json={"username": user, "password": pw}).status_code == 200
    return c


def test_resolve_api_validates_and_records_the_user(app):
    bus = EventBus(db_path=app.config["DB"])
    ev = emit(bus, "fire_detected", time.time())
    admin = _login(app, "admin", "admin")
    admin.post("/api/users", json={"username": "nadia", "password": "nadia1",
                                   "role": "guest", "must_change_password": False})
    guest = _login(app, "nadia", "nadia1")
    url = f"/api/incidents/{ev.incident_id}/resolve"

    assert guest.post(url, json={"outcome": "maybe"}).status_code == 400
    assert guest.post(url, json={"outcome": "false_alarm", "note": "n" * 201}).status_code == 400
    assert guest.post("/api/incidents/9999/resolve",
                      json={"outcome": "acknowledged"}).status_code == 404

    r = guest.post(url, json={"outcome": "false_alarm", "note": "  cooking steam "})
    assert r.status_code == 200
    body = r.get_json()
    assert (body["status"], body["resolved_by"], body["note"]) == ("false_alarm", "nadia", "cooking steam")

    board = guest.get("/api/incidents/board").get_json()
    assert board["open_total"] == 0 and board["recent"][0]["incident_id"] == ev.incident_id
    assert guest.get("/api/status").get_json()["open_incidents"] == 0
    assert guest.get("/api/incidents?status=bogus").status_code == 400
    assert guest.post("/api/events/clear").status_code == 403          # clearing stays admin-only
    detail = guest.get(f"/api/incidents/{ev.incident_id}").get_json()
    assert len(detail["events"]) == 1


def test_notify_test_explains_missing_setup(app, monkeypatch):
    for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM"):
        monkeypatch.delenv(k, raising=False)
    admin = _login(app, "admin", "admin")
    r = admin.post("/api/notify/test")
    assert r.status_code == 400 and "Account SID" in r.get_json()["error"]
    assert admin.get("/api/notify/status").get_json()["ready"] is False


SID = "AC" + "0123456789abcdef" * 2
TOKEN = "fedcba9876543210" * 2


def test_twilio_details_saved_in_settings_never_leave_the_server(app, monkeypatch):
    for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM"):
        monkeypatch.delenv(k, raising=False)
    admin = _login(app, "admin", "admin")
    r = admin.post("/api/settings", json={"twilio_account_sid": SID, "twilio_auth_token": TOKEN,
                                          "twilio_whatsapp_from": "whatsapp:+1 415-523-8886",
                                          "alert_phones": "+60123456789"})
    assert r.status_code == 200
    for body in (r.get_data(as_text=True), admin.get("/api/settings").get_data(as_text=True)):
        assert TOKEN not in body
    s = admin.get("/api/settings").get_json()
    assert s["twilio_token_saved"] is True and s["twilio_token_hint"] == TOKEN[-4:]
    assert s["twilio_whatsapp_from"] == "+14155238886"
    status = admin.get("/api/notify/status").get_json()
    assert status["ready"] is True and status["uses_env"] is False

    # a blank token keeps the saved one; the clear flag removes it
    admin.post("/api/settings", json={"twilio_auth_token": "", "twilio_account_sid": SID})
    assert admin.get("/api/settings").get_json()["twilio_token_saved"] is True
    admin.post("/api/settings", json={"twilio_auth_token_clear": True})
    assert admin.get("/api/settings").get_json()["twilio_token_saved"] is False
    assert admin.get("/api/notify/status").get_json()["missing"] == ["Auth token"]


def test_twilio_fields_are_validated(app):
    admin = _login(app, "admin", "admin")
    for bad in ({"twilio_account_sid": "SK123"}, {"twilio_auth_token": "short"},
                {"twilio_whatsapp_from": "0123"}):
        r = admin.post("/api/settings", json=bad)
        assert r.status_code == 400, bad
    assert admin.get("/api/settings").get_json()["twilio_account_sid"] == ""


def test_saved_twilio_details_win_over_env(monkeypatch):
    from homeshield.notify import resolve_config
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC_from_env")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "env-token")
    monkeypatch.delenv("TWILIO_WHATSAPP_FROM", raising=False)
    cfg = resolve_config(_Settings(twilio_account_sid=SID))
    assert cfg["twilio_account_sid"] == {"value": SID, "source": "settings"}
    assert cfg["twilio_auth_token"] == {"value": "env-token", "source": "env"}
    assert cfg["twilio_whatsapp_from"]["source"] == ""
