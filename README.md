# 🛡️ HomeShield v2.0 — GPU Accelerated

**Intelligent ML-Powered CCTV Surveillance System for Elderly & Child Safety**

HomeShield is a real-time computer vision surveillance system that autonomously detects dangerous situations involving elderly persons and young children in the home environment, sending instant WhatsApp alerts to family members.

Built with YOLOv8, MediaPipe, and Flask — runs on both CPU and NVIDIA GPU with automatic CUDA detection.

## ✨ Features

- **Real-time person detection** — YOLOv8 with GPU acceleration (8-15ms inference on CUDA)
- **Pose estimation** — MediaPipe extracts 33 body keypoints for activity analysis
- **Fall detection** — Rule-based analysis of body angle, height ratio, and hip velocity
- **Inactivity monitoring** — Alerts when no movement is detected for a configurable period
- **Child zone monitoring** — Define restricted danger areas; alerts when a child enters
- **Person classification** — Heuristic classification of detected persons as elderly, child, or adult
- **WhatsApp alerts** — Instant notifications via Twilio with event snapshots
- **Web dashboard** — Live camera feeds, event log, zone editor, and settings
- **Multi-camera support** — Up to 4 RTSP cameras simultaneously
- **GPU acceleration** — Automatic CUDA detection, FP16 half-precision, model warmup
- **Performance monitoring** — Real-time FPS and inference timing overlay
- **Local processing** — All ML inference runs locally; no cloud dependency

## 🚀 Performance (CPU vs GPU)

| Metric | CPU (Intel i5) | GPU (RTX 3060) |
|--------|---------------|----------------|
| YOLOv8 inference | ~80-120ms | ~8-15ms |
| Processing FPS | 5-8 fps | 15-30 fps |
| MediaPipe complexity | 1 (full) | 2 (heavy, auto-upgraded) |
| FP16 half-precision | Not available | Enabled (2x speed) |
| Fall detection response | ~2-3 seconds | <1 second |

## 📋 Requirements

### Minimum (CPU Mode)
- Python 3.9 - 3.11
- 8GB RAM
- Webcam or RTSP IP camera
- Windows / macOS / Linux

### Recommended (GPU Mode)
- NVIDIA GPU (RTX 20xx / 30xx / 40xx series)
- CUDA Toolkit 11.8 or 12.1
- 16GB RAM
- Dedicated IP cameras (TP-Link Tapo C200/C210)

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/salehuddiniwan/HomeShield.git
cd HomeShield
```

### 2. Create Anaconda environment
```bash
conda create --name homeshield python=3.11 -y
conda activate homeshield
```

### 3. Install PyTorch with CUDA (GPU users)
```bash
# For NVIDIA GPU with CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For NVIDIA GPU with CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only (no NVIDIA GPU):
pip install torch torchvision
```

### 4. Install dependencies
```bash
pip install flask opencv-python ultralytics mediapipe==0.10.14 numpy python-dotenv twilio
```

### 5. Download YOLOv8 model
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 6. Configure environment
```bash
copy .env.example .env    # Windows
cp .env.example .env      # macOS/Linux
```
Edit `.env` with your Twilio credentials and phone numbers.

### 7. Run HomeShield
```bash
python app.py
```

### 8. Open dashboard
```
http://localhost:5000
```

## 📁 Project Structure

```
HomeShield/
├── app.py                 # Flask app — routes, video streaming, processing loop
├── config.py              # Configuration with GPU settings
├── database.py            # SQLite — events, cameras, zones, settings
├── detector.py            # YOLOv8 + MediaPipe + fall detection (GPU-accelerated)
├── camera_manager.py      # Threaded multi-camera RTSP capture
├── alerter.py             # Twilio WhatsApp notification dispatch
├── requirements.txt       # Python dependencies
├── setup.sh               # One-click setup script (macOS/Linux)
├── .env.example           # Environment variable template
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── templates/
│   └── index.html         # Web dashboard (single-page app)
└── static/
    └── snapshots/         # Saved alert snapshot images
```

## ⚙️ Configuration

All settings are in `.env`. Key options:

### GPU Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `auto` | `auto` = use GPU if available, `cuda` = force GPU, `cpu` = force CPU |
| `USE_HALF_PRECISION` | `true` | FP16 mode — 2x faster on RTX GPUs |

### Detection Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL` | `yolov8n.pt` | Model: `yolov8n.pt` (fast), `yolov8s.pt` (balanced), `yolov8m.pt` (accurate) |
| `YOLO_CONFIDENCE` | `0.5` | Minimum detection confidence (0.0 - 1.0) |
| `YOLO_IMG_SIZE` | `640` | Input image size for YOLOv8 |
| `POSE_MODEL_COMPLEXITY` | `1` | MediaPipe: 0=lite, 1=full, 2=heavy (auto-upgraded on GPU) |

### Alert Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `FALL_THRESHOLD` | `0.80` | Fall detection confidence threshold |
| `INACTIVITY_SECONDS` | `300` | Seconds before inactivity alert |
| `ALERT_COOLDOWN` | `60` | Minimum seconds between repeat alerts |
| `PROCESS_FPS` | `15` | Frames per second (GPU: 15-30, CPU: 5-8) |

### Twilio WhatsApp
| Variable | Default | Description |
|----------|---------|-------------|
| `TWILIO_ACCOUNT_SID` | — | Twilio account SID |
| `TWILIO_AUTH_TOKEN` | — | Twilio auth token |
| `TWILIO_WHATSAPP_FROM` | `whatsapp:+14155238886` | Twilio sandbox number |
| `ALERT_PHONE_NUMBERS` | — | Comma-separated phone numbers with country code |

## 📷 Camera Setup

### Webcam (testing)
Default camera URL is `0` (built-in webcam). No setup needed.

### Phone as IP Camera (DroidCam)
1. Install **DroidCam** on your Android phone
2. Connect phone and laptop to same WiFi
3. Open DroidCam — note the IP and port
4. Add camera URL in HomeShield: `http://PHONE_IP:4747/video`

### RTSP IP Cameras
| Camera | RTSP URL Format |
|--------|----------------|
| TP-Link Tapo C200/C210 | `rtsp://user:pass@IP:554/stream1` |
| Hikvision | `rtsp://admin:pass@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://admin:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| Reolink | `rtsp://admin:pass@IP:554/h264Preview_01_main` |

### Video File (testing with datasets)
Use a video file path as the camera URL:
```
C:\Users\Admin\Desktop\test_videos\fall_video.avi
```

## 🔔 WhatsApp Alert Setup

1. Create a [Twilio account](https://www.twilio.com/try-twilio) (free)
2. Go to **Messaging > Try it out > Send a WhatsApp message**
3. Send the join code from your phone to the Twilio sandbox number
4. Copy Account SID and Auth Token to `.env`
5. Add recipient phone numbers to `ALERT_PHONE_NUMBERS`

> **Note:** Sandbox connection expires after 72 hours. Re-send join code before demos.

## 🧠 How Detection Works

### Fall Detection Pipeline
```
Camera Frame → YOLOv8 (person detection) → MediaPipe (pose estimation) → Fall Analysis → Alert
```

The fall detector analyzes three signals from MediaPipe pose keypoints:
1. **Body angle** — Torso angle from vertical (>55° = horizontal/fallen)
2. **Height ratio** — Vertical vs horizontal body extent (<1.2 = lying down)
3. **Hip velocity** — Rate of hip descent (>0.8 = rapid fall)

A fall is confirmed when: body is horizontal + lying flat + rapid descent detected.

### Person Classification
Heuristic based on bounding box height relative to frame:
- **Child**: height < 35% of frame
- **Adult**: height > 55% of frame
- **Elderly**: height between 35–55%

### Zone Monitoring
Users define polygon zones via the dashboard. The system uses ray-casting to check if a child's bounding box center falls within a restricted zone.

## 🖥️ GPU Verification

Check if your GPU is working:
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

When HomeShield starts with GPU, you will see:
```
==================================================
  GPU STATUS
==================================================
  Device:     NVIDIA GeForce RTX 3060
  VRAM:       12.0 GB total, 11.4 GB free
  CUDA:       12.1
  cuDNN:      Yes
  FP16:       Yes (enabled)
  Mode:       GPU ACCELERATED
==================================================
```

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard |
| GET | `/video_feed/<id>` | MJPEG video stream |
| GET | `/api/status` | System status, camera info, people count |
| GET | `/api/events` | Event log (filterable by type and camera) |
| GET | `/api/cameras` | List all cameras |
| POST | `/api/cameras` | Add camera (JSON: name, url, location) |
| DELETE | `/api/cameras/<id>` | Remove a camera |
| GET | `/api/zones` | List danger zones |
| POST | `/api/zones` | Add zone (JSON: zone_name, camera_id, polygon) |
| DELETE | `/api/zones/<id>` | Remove a zone |
| POST | `/api/system/start` | Start detection pipeline |
| POST | `/api/system/stop` | Stop detection pipeline |
| GET | `/api/settings` | Get current settings |
| POST | `/api/settings` | Update settings |

## 🧪 Testing Datasets

Free fall detection video datasets for testing:
- [Le2i Fall Dataset](http://le2i.cnrs.fr/Fall-detection-Dataset) — 221 home environment videos
- [UR Fall Detection](http://fenix.ur.edu.pl/~mkepski/ds/uf.html) — 70 videos, 2 camera angles
- [FallVision 2025](https://data.mendeley.com/datasets/wkgxspfrps) — Falls from bed, chair, standing
- [Fall Detection Dataset (GitHub)](https://github.com/YifeiYang210/Fall_Detection_dataset) — Curated links

## 📜 License

Academic project — IIUM KICT Final Year Project (FYP).

## 👤 Author

**MOHAMMAD SALEHUDDIN IWAN**
Bachelor of Computer Science
Kulliyyah of Information and Communication Technology
International Islamic University Malaysia

---

*HomeShield v2.0 — April 2026*
