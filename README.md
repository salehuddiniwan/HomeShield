# 🛡️ HomeShield

**Intelligent ML-Powered CCTV Surveillance System for Elderly & Child Safety**

HomeShield is a real-time computer vision surveillance system that autonomously detects dangerous situations involving elderly persons and young children in the home environment, sending instant WhatsApp alerts to family members.

## Features

- **Real-time person detection** — YOLOv8 nano model detects people in live camera feeds
- **Pose estimation** — MediaPipe extracts 33 body keypoints for activity analysis
- **Fall detection** — Rule-based analysis of body angle, height ratio, and hip velocity
- **Inactivity monitoring** — Alerts when no movement is detected for a configurable period
- **Child zone monitoring** — Define restricted areas; alerts when a child enters
- **Person classification** — Heuristic classification of detected persons as elderly, child, or adult
- **WhatsApp alerts** — Instant notifications via Twilio with event details
- **Web dashboard** — Live camera feeds, event log, zone editor, and settings
- **Multi-camera support** — Up to 4 RTSP cameras simultaneously
- **Local processing** — All ML inference runs locally; no cloud dependency

## Requirements

- Python 3.9+
- Webcam or RTSP-capable IP camera (e.g., TP-Link Tapo C200)
- 8GB RAM minimum
- Optional: NVIDIA GPU for faster inference

## Quick Start

```bash
# 1. Clone / download the project
cd homeshield

# 2. Run setup (creates venv, installs deps, downloads YOLOv8)
chmod +x setup.sh
./setup.sh

# 3. Edit configuration
cp .env.example .env
nano .env  # Add your Twilio credentials and phone numbers

# 4. Start the system
python app.py

# 5. Open dashboard
# http://localhost:5000
```

## Project Structure

```
homeshield/
├── app.py               # Flask app — routes, video streaming, main loop
├── config.py            # Configuration from environment variables
├── database.py          # SQLite — events, cameras, zones, settings
├── detector.py          # YOLOv8 + MediaPipe + fall detection pipeline
├── camera_manager.py    # Threaded multi-camera RTSP capture
├── alerter.py           # Twilio WhatsApp notification dispatch
├── requirements.txt     # Python dependencies
├── setup.sh             # One-click setup script
├── .env.example         # Environment variable template
├── templates/
│   └── index.html       # Web dashboard (single-page app)
├── static/
│   └── snapshots/       # Saved alert images
└── models/              # Custom ML models (future)
```

## Configuration

All settings are in `.env`. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL` | `yolov8n.pt` | YOLOv8 model file |
| `YOLO_CONFIDENCE` | `0.5` | Minimum detection confidence |
| `FALL_THRESHOLD` | `0.80` | Fall detection confidence threshold |
| `INACTIVITY_SECONDS` | `300` | Seconds before inactivity alert |
| `ALERT_COOLDOWN` | `60` | Minimum seconds between repeat alerts |
| `PROCESS_FPS` | `8` | Frames per second to process |
| `TWILIO_ACCOUNT_SID` | — | Twilio account SID |
| `TWILIO_AUTH_TOKEN` | — | Twilio auth token |
| `ALERT_PHONE_NUMBERS` | — | Comma-separated WhatsApp numbers |

## Camera Setup

### Using a webcam (for testing)
The default configuration uses your laptop webcam (device `0`). No setup needed.

### Using IP cameras
1. Get your camera's RTSP URL (check camera manual)
2. Common formats:
   - TP-Link Tapo: `rtsp://username:password@192.168.1.x:554/stream1`
   - Hikvision: `rtsp://admin:password@192.168.1.x:554/Streaming/Channels/101`
   - Generic: `rtsp://192.168.1.x:554/live`
3. Add cameras via the dashboard Settings page, or edit `.env`

## WhatsApp Alert Setup

1. Create a [Twilio account](https://www.twilio.com/try-twilio)
2. Activate the [WhatsApp Sandbox](https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn)
3. Follow Twilio's instructions to connect your phone number to the sandbox
4. Copy your Account SID and Auth Token to `.env`
5. Add recipient phone numbers (with country code) to `ALERT_PHONE_NUMBERS`

## How Detection Works

### Fall Detection
The system uses a rule-based approach analyzing MediaPipe pose keypoints:

1. **Body angle** — Torso angle from vertical (>55° = horizontal/fallen)
2. **Height ratio** — Vertical vs horizontal body extent (<1.2 = lying down)
3. **Hip velocity** — Rate of hip descent (>0.8 = rapid fall)
4. **Temporal analysis** — Checks if person was recently upright before becoming horizontal

A fall is confirmed when: body is horizontal + lying flat + rapid descent detected.

### Person Classification
Simple heuristic based on bounding box height relative to frame:
- **Child**: height < 35% of frame
- **Adult**: height > 55% of frame
- **Elderly**: height between 35–55% (default assumption for home setting)

### Zone Monitoring
Users define polygon zones on camera feeds via the dashboard. The system checks if a child's bounding box center falls within the polygon using ray-casting.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard |
| GET | `/video_feed/<id>` | MJPEG stream |
| GET | `/api/status` | System status |
| GET | `/api/events` | Event log |
| GET | `/api/cameras` | Camera list |
| POST | `/api/cameras` | Add camera |
| DELETE | `/api/cameras/<id>` | Remove camera |
| GET | `/api/zones` | Zone list |
| POST | `/api/zones` | Add zone |
| DELETE | `/api/zones/<id>` | Remove zone |
| POST | `/api/system/start` | Start detection |
| POST | `/api/system/stop` | Stop detection |
| GET | `/api/settings` | Get settings |
| POST | `/api/settings` | Update settings |

## License

Academic project — IIUM KICT Final Year Project.

## Author

[Your Name] — International Islamic University Malaysia
