"""WhatsApp alerts through Twilio: one message per incident, not per event.

A fire in view produces an event every few seconds; texting each one would
bury the family's phones. Messages go out when an incident opens and again
if it escalates (e.g. a fall becomes "lying motionless").

Off unless a Twilio Account SID, auth token and WhatsApp sender are known
and Settings > Notifications lists at least one number. The three Twilio
values come from Settings > Notifications, or else from the environment /
.env (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM). Sending runs on its own thread so a slow network never delays
the dashboard; failures are logged, never raised into the pipeline.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from typing import Any

log = logging.getLogger(__name__)

TWILIO_URL = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
ENV_KEYS = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM")
# (settings key, environment variable, name shown to people)
CONFIG_FIELDS = (
    ("twilio_account_sid", "TWILIO_ACCOUNT_SID", "Account SID"),
    ("twilio_auth_token", "TWILIO_AUTH_TOKEN", "Auth token"),
    ("twilio_whatsapp_from", "TWILIO_WHATSAPP_FROM", "WhatsApp sender"),
)
MAX_PER_MINUTE = 20          # a cap on cost if something misfires
_PHONE_RE = re.compile(r"^\+[1-9]\d{6,14}$")

LABELS = {
    "fall_detected": "Fall detected",
    "lying_motionless": "Person lying motionless",
    "inactivity": "No movement for a long time",
    "fire_detected": "Fire or smoke detected",
    "intruder_detected": "Intruder detected (unknown face)",
    "zone_entry": "Child entered a danger zone",
}


def parse_phones(raw: str) -> list[str]:
    """'+60 12-345 6789, +60198765432' -> ['+60123456789', '+60198765432']."""
    out: list[str] = []
    for part in re.split(r"[,;\n]+", raw or ""):
        p = re.sub(r"[\s\-().]", "", part)
        if p.lower().startswith("whatsapp:"):
            p = p[len("whatsapp:"):]
        if _PHONE_RE.match(p) and p not in out:
            out.append(p)
    return out


def compose(change: str, inc: dict[str, Any]) -> str:
    what = LABELS.get(inc.get("headline_type", ""), "Alert")
    where = inc.get("camera_name") or "a camera"
    when = time.strftime("%H:%M", time.localtime(inc.get("last_ts") or time.time()))
    conf = int(round(float(inc.get("peak_confidence") or 0) * 100))
    lead = "HomeShield - ESCALATED: " if change == "escalated" else "HomeShield: "
    return (f"{lead}{what} on {where} at {when} (confidence {conf}%). "
            f"Open the dashboard to view it live and respond.")


def resolve_config(settings=None) -> dict[str, dict[str, str]]:
    """Each Twilio value and where it came from: 'settings', 'env' or ''."""
    out = {}
    for key, env, _label in CONFIG_FIELDS:
        saved = str((settings.get(key, "") if settings is not None else "") or "").strip()
        from_env = os.environ.get(env, "").strip()
        out[key] = ({"value": saved, "source": "settings"} if saved else
                    {"value": from_env, "source": "env" if from_env else ""})
    return out


def missing_config(settings=None) -> list[str]:
    cfg = resolve_config(settings)
    return [label for key, _env, label in CONFIG_FIELDS if not cfg[key]["value"]]


def send_whatsapp(to: str, body: str, cfg: dict, timeout: float = 10.0) -> None:
    """Send one message; raises RuntimeError with Twilio's reason on failure."""
    sid = cfg["twilio_account_sid"]["value"]
    token = cfg["twilio_auth_token"]["value"]
    sender = cfg["twilio_whatsapp_from"]["value"]
    if not sender.startswith("whatsapp:"):
        sender = "whatsapp:" + sender
    data = urllib.parse.urlencode(
        {"From": sender, "To": "whatsapp:" + to, "Body": body}).encode()
    req = urllib.request.Request(TWILIO_URL.format(sid=sid), data=data, method="POST")
    auth = base64.b64encode(f"{sid}:{token}".encode()).decode()
    req.add_header("Authorization", "Basic " + auth)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
    except urllib.error.HTTPError as e:
        try:
            reason = json.loads(e.read().decode()).get("message") or str(e)
        except Exception:
            reason = str(e)
        raise RuntimeError(reason) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(f"could not reach Twilio ({e})") from None


class WhatsAppNotifier:
    def __init__(self, settings, sender=None):
        self.settings = settings
        # Tests pass a fake sender(to, body); the real one reads the current
        # Twilio details on every send, so saving Settings applies at once.
        self._send = sender or (lambda to, body: send_whatsapp(
            to, body, resolve_config(self.settings)))
        self._q: queue.Queue = queue.Queue(maxsize=100)
        self._sent_at: deque[float] = deque()
        threading.Thread(target=self._loop, name="hs-whatsapp",
                         daemon=True).start()

    def phones(self) -> list[str]:
        return parse_phones(str(self.settings.get("alert_phones", "") or ""))

    def status(self) -> dict[str, Any]:
        missing = missing_config(self.settings)
        phones = self.phones()
        sources = {v["source"] for v in resolve_config(self.settings).values()}
        return {"ready": not missing and bool(phones),
                "missing": missing, "phones": len(phones),
                "uses_env": "env" in sources}

    def on_incident(self, change: str, inc: dict[str, Any]) -> None:
        """EventBus listener: queue a message for new or escalated incidents."""
        if change not in ("opened", "escalated") or missing_config(self.settings):
            return
        phones = self.phones()
        if not phones:
            return
        try:
            self._q.put_nowait((compose(change, inc), phones))
        except queue.Full:
            log.warning("WhatsApp queue full; dropped incident %s",
                        inc.get("incident_id"))

    def send_test(self) -> dict[str, Any]:
        """Synchronous test from Settings; returns per-number results."""
        body = ("HomeShield test message: WhatsApp alerts are set up. You will "
                "get one message when an incident starts and one if it escalates.")
        sent, failed = 0, []
        for to in self.phones():
            try:
                self._send(to, body)
                sent += 1
            except Exception as e:
                failed.append({"to": to, "error": str(e)})
        return {"sent": sent, "failed": failed}

    def _loop(self) -> None:
        while True:
            body, phones = self._q.get()
            for to in phones:
                now = time.time()
                while self._sent_at and now - self._sent_at[0] > 60:
                    self._sent_at.popleft()
                if len(self._sent_at) >= MAX_PER_MINUTE:
                    log.warning("WhatsApp rate cap reached (%d/min); skipped %s",
                                MAX_PER_MINUTE, to)
                    continue
                self._sent_at.append(now)
                try:
                    self._send(to, body)
                    log.info("WhatsApp alert sent to %s", to)
                except Exception as e:
                    log.warning("WhatsApp alert to %s failed: %s", to, e)


def load_dotenv(path) -> None:
    """Read KEY=VALUE lines into os.environ without overriding real env vars."""
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
