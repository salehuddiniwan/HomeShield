"""
HomeShield Alerter — sends WhatsApp notifications via Twilio with snapshot images.
"""
import os
import time
import threading
from datetime import datetime
from config import Config

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("[WARN] twilio not installed — WhatsApp alerts will be logged only")


class Alerter:
    """Manages alert dispatch with cooldown to prevent spam."""

    def __init__(self):
        self.twilio_client = None
        self.last_alert_time = {}  # event_type:camera_id -> timestamp
        self.alert_log = []  # in-memory log of recent alerts

        if TWILIO_AVAILABLE and Config.TWILIO_ACCOUNT_SID and Config.TWILIO_AUTH_TOKEN:
            try:
                self.twilio_client = TwilioClient(
                    Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN
                )
                print("[INFO] Twilio WhatsApp client initialized")
            except Exception as e:
                print(f"[WARN] Twilio init failed: {e}")

    def should_alert(self, event_type, camera_id):
        """Check cooldown. Returns True if enough time has passed."""
        key = f"{event_type}:{camera_id}"
        last = self.last_alert_time.get(key, 0)
        return (time.time() - last) > Config.ALERT_COOLDOWN_SECONDS

    def send_alert(self, event_type, camera_name, person_category, confidence,
                   snapshot_path=None, camera_id=None, details=""):
        """Send WhatsApp alert. Non-blocking (threaded)."""

        key = f"{event_type}:{camera_id}"
        self.last_alert_time[key] = time.time()

        alert_info = {
            "event_type": event_type,
            "camera_name": camera_name,
            "person_category": person_category,
            "confidence": round(confidence, 2),
            "snapshot_path": snapshot_path,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "sent": False,
        }

        # Format message
        event_labels = {
            "fall_detected": "FALL DETECTED",
            "lying_motionless": "PERSON LYING MOTIONLESS",
            "inactivity": "PROLONGED INACTIVITY",
            "zone_entry": "CHILD ENTERED RESTRICTED ZONE",
        }
        label = event_labels.get(event_type, event_type.upper())

        severity = "CRITICAL" if event_type in ("fall_detected", "lying_motionless") else "WARNING"

        msg = (
            f"{'🚨' if severity == 'CRITICAL' else '⚠️'} *HomeShield Alert*\n\n"
            f"*{severity}: {label}*\n\n"
            f"📍 Camera: {camera_name}\n"
            f"👤 Person: {person_category}\n"
            f"📊 Confidence: {confidence:.0%}\n"
            f"🕐 Time: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}\n"
        )
        if details:
            msg += f"ℹ️ Details: {details}\n"
        msg += "\n_Please check on your family member immediately._"

        threading.Thread(
            target=self._dispatch, args=(msg, snapshot_path, alert_info), daemon=True
        ).start()

        return alert_info

    def _dispatch(self, message, snapshot_path, alert_info):
        """Actually send via Twilio (runs in background thread)."""
        if not Config.ALERT_PHONE_NUMBERS:
            print(f"[ALERT] No phone numbers configured. Message:\n{message}")
            alert_info["sent"] = False
            self.alert_log.append(alert_info)
            return

        if self.twilio_client is None:
            print(f"[ALERT] Twilio not configured. Message:\n{message}")
            alert_info["sent"] = False
            self.alert_log.append(alert_info)
            return

        for phone in Config.ALERT_PHONE_NUMBERS:
            try:
                kwargs = {
                    "body": message,
                    "from_": Config.TWILIO_WHATSAPP_FROM,
                    "to": f"whatsapp:{phone}" if not phone.startswith("whatsapp:") else phone,
                }

                # Attach snapshot if available and hosted
                if snapshot_path and os.path.exists(snapshot_path):
                    # Note: Twilio requires a public URL for media.
                    # For local deployment, you'd need ngrok or a public server.
                    # For now we send text-only; see README for media setup.
                    pass

                self.twilio_client.messages.create(**kwargs)
                alert_info["sent"] = True
                print(f"[ALERT] WhatsApp sent to {phone}")

            except Exception as e:
                print(f"[ERROR] WhatsApp send failed to {phone}: {e}")
                alert_info["sent"] = False

        self.alert_log.append(alert_info)

    def get_recent_alerts(self, count=20):
        return self.alert_log[-count:]
