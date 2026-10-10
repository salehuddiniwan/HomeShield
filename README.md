# 🛡️ HomeShield

**HomeShield** is a centralized dashboard for **real-time CCTV monitoring** with three GPU-accelerated anomaly detectors running side-by-side on every camera feed. One unified pipeline draws bounding-box overlays on the live MJPEG stream, persists every alert to SQLite with an annotated snapshot, and pushes updates to the dashboard via Server-Sent Events.

Built as a Final-Year Project at the **International Islamic University Malaysia (IIUM), Kulliyyah of Information and Communication Technology**.

> One pipeline. Three detectors. Per-camera worker threads sharing one set of YOLO / ONNX models on the GPU.

---

## ✨ Features

### Detection
- 🤸 **Fall detection** — any Ultralytics YOLO pose checkpoint (`yolov8`, `yolo11`, `yolo26`) feeds a research-tuned **7-state finite-state machine**: *Standing → Walking → Sitting → Fall_Detected → Lying_After_Fall → Lying_Motionless → Inactivity*. A two-stage decision (peak descent velocity **plus** sustained horizontal posture) rejects controlled sit-downs and false positives.
- 🔥 **Fire & smoke detection** — custom YOLO weights covering fire and smoke classes, with **per-class cooldown** to debounce repeated alerts.
- 👤 **Face / intruder detection** — InsightFace **ArcFace (buffalo_l)** generates 512-dimensional embeddings, cosine-matched against your registered Persons gallery. Unknown faces are auto-logged to the **Intruders** list with a snapshot.

### Identity & alerts
- **Registered persons** gallery — enrol family members from a still photo or a live capture.
- **Auto-intruder logging** — anyone not in the gallery is snapshot-logged with timestamp and camera.
- **Real-time alert stream** — Server-Sent Events push new events to the dashboard the instant they're published.
- **Incidents, not a wall of rows** — detections are grouped into incidents (one camera, one kind of trouble, no gap over 2 minutes), each with its peak confidence and best snapshot. Anyone signed in can **acknowledge** an incident or mark it a **false alarm**, with an optional note; the record keeps who did it, and a 7-day false-alarm count per detector and camera builds up on the Events page.
- **WhatsApp alerts (optional)** — one message through Twilio when an incident starts and one more if it escalates (e.g. a fall becomes "lying motionless"), never one per detection.
- **Annotated snapshots** — every event is saved as a JPEG with the bounding box / pose skeleton / face label burned in.

### Zones
- **Polygon zones per camera** drawn directly on the live preview.
- **Safe zones** suppress lying / inactivity alerts (e.g. on a bed or sofa).
- **Danger zones** trigger a child-entry alert when a person crosses into them (kitchen, balcony, pool, etc.).

### Platform
- **Multi-camera** — webcams, IP cameras (RTSP), DroidCam, IP Webcam, or a video file for testing.
- **Live MJPEG streams** with detector overlays.
- **SQLite event log** in WAL mode + filesystem snapshots.
- **Hot-reloadable settings** — toggle detectors, swap pose models, change FPS / imgsz / thresholds without restarting cameras.
- **Mobile-friendly UI** — responsive Flask dashboard you can hit from your phone on the same network.

---

## 📋 Requirements

### Recommended setup (what this is tuned for)
- **OS:** Windows 11
- **Python:** 3.11 (Anaconda virtual environment)
- **GPU:** NVIDIA GPU with **CUDA 12.x** drivers (tested on RTX-class hardware)
- **PyTorch:** 2.x with CUDA build
- **RAM:** 16 GB+
- **Disk:** ~5 GB for weights, snapshots, and the SQLite database

### Minimum (CPU-only, no intruder detection)
You can run HomeShield on a machine without a GPU, but with caveats:
- **CPU-only** PyTorch and ONNX Runtime work for **Fall** and **Fire** detection — expect 5–15 FPS on a modern laptop CPU at small image sizes.
- **Face / intruder detection should be disabled** — InsightFace inference on CPU is too slow to keep up with a live feed and will tank the pose framerate.
- 8 GB RAM minimum.
- Lower the **imgsz** and **target FPS** in *Settings → Performance* aggressively (e.g. imgsz 416, FPS cap 10).

### Python version note
**Python 3.11 is required.** The bundled InsightFace wheel (`insightface-0.7.3-cp311-cp311-win_amd64.whl`) is compiled specifically for **CPython 3.11 on Windows x64**. Other Python versions will fail to install it from source unless you have MSVC build tools configured. If you must use a different Python version, you'll need to compile InsightFace yourself or find a matching pre-built wheel.

---

## ⚡ Installation

### 1. Get Python 3.11
Install Python 3.11 via Anaconda (recommended) or the official installer:

```bash
# With Anaconda
conda create -n homeshield python=3.11
conda activate homeshield
```

Verify:

```bash
python --version
# Python 3.11.x
```

### 2. Clone and create a venv

```bash
git clone https://github.com/salehuddiniwan/HomeShield.git
cd HomeShield

# If you skipped the conda step above, create a plain venv:
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install PyTorch (do this BEFORE requirements.txt)
**This step is critical.** If you let pip resolve PyTorch from `requirements.txt`, it will pull the **CPU-only** build. Install PyTorch from a CUDA index **first**, and note which CUDA version you picked; step 5 has to match it:

```bash
# CUDA 12 (recommended; NVIDIA driver 525 or newer)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 13 (NVIDIA driver 580 or newer)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# CPU-only fallback (no NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Run `nvidia-smi` to see your driver version. Verify CUDA is detected:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda)"
```

### 4. Install InsightFace (Windows-specific)
On Windows without MSVC build tools, pip cannot compile InsightFace from source. Use the **bundled pre-built wheel** instead:

```bash
pip install Face_Detection/insightface-0.7.3-cp311-cp311-win_amd64.whl
```

(Use the `cp310` wheel on Python 3.10.) On macOS / Linux, or Windows with MSVC installed, `pip install insightface==0.7.3` works directly and step 5 handles it.

### 5. Install the rest

`requirements.txt` installs HomeShield in editable mode with the **CUDA 12** build of ONNX Runtime (which runs the face models):

```bash
pip install -r requirements.txt
```

If you installed the CUDA 13 or CPU build of PyTorch in step 3, install the matching extra instead:

```bash
pip install -e .[plot,cuda13]
```

```bash
pip install -e .[plot,cpu]
```

ONNX Runtime's CUDA version must match PyTorch's (`onnxruntime-gpu` 1.27+ needs CUDA 13, 1.26 and older need CUDA 12). With a mismatch, face recognition silently falls back to the CPU; HomeShield logs a warning naming the fix, and `/healthz` reports `face_device`.

The other pins avoid real breakages:
- `numpy<2.0`: InsightFace wheels are compiled against NumPy 1.x.
- `opencv-python` and `opencv-python-headless` are both kept on 4.10.x. Ultralytics needs the first and InsightFace's `albumentations` the second, and both install into the same `cv2` folder.
- `lap` is required by the person tracker (Ultralytics 8.3 and 8.4 both import it). HomeShield also turns off Ultralytics' auto-installer, which otherwise runs whatever `pip` is on your PATH and can install into a different Python.

Whichever OpenCV package was written last provides `cv2`. The dashboard works with either. Only the standalone fall runner's `--show` window needs the GUI build; if it reports that the window is unavailable, restore it with:

```bash
pip install --force-reinstall --no-deps "opencv-python~=4.10.0"
```

Verify the full stack loads cleanly:

```bash
python -c "import torch, cv2, ultralytics, insightface, onnxruntime, lap; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('cv2', cv2.__version__); print('onnxruntime', onnxruntime.__version__, onnxruntime.get_available_providers())"
```

Optionally run the tests (no GPU or weights needed):

```bash
pip install pytest
```

```bash
python -m pytest
```

### 6. Configure
- Drop YOLO **pose weights** (`*-pose.pt`) into `Fall_Detection/weights/`. The dashboard's **Settings → Fall detection** dropdown auto-populates from whatever's in that folder.
- Make sure **Fire_Detection/best.pt** exists (the custom fire/smoke weights).
- The first time you run the app, it creates `homeshield.db`, `snapshots/`, `person_photos/`, and `intruder_photos/` automatically.
- *(Optional)* WhatsApp alerts: in **Settings → Notifications → Twilio account**, paste your **Account SID**, **Auth token** and **WhatsApp sender** number from the Twilio Console, add the phone numbers under **WhatsApp alerts**, save, and use **Send test message**. The auth token is write-only: it is stored in `homeshield.db` and never shown again (**Remove saved token** deletes it). The **WhatsApp sender** is Twilio's WhatsApp number (on a trial, the one shown on the Console's **Try out WhatsApp** page), not your own phone. Each receiving phone must be a verified tester and first send the join code from that page (e.g. `join twilio-trial`) to the sender number. Without these values, WhatsApp stays off and nothing leaves the machine.

  **Twilio trial accounts only send Twilio's own message templates** (free text fails with "ContentSid Required"). Copy the template's `contentSid` (starts with `HX`) from the code sample on the Try out WhatsApp page into **Message template**. Trial templates have fixed wording, so the message tells you to check HomeShield rather than what happened. After upgrading Twilio you can get your own template approved, e.g. *"HomeShield: {{1}} on {{2}} at {{3}} (confidence {{4}})"*, and tick **My template has {{1}} to {{4}}** so alerts fill in what, where, when and how sure. Leave Message template empty to send HomeShield's own text (works on upgraded accounts within WhatsApp's 24-hour reply window).

  You can also keep the details in a git-ignored `.env` file next to `run_homeshield.py` (values saved in Settings take priority):
  ```
  TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  TWILIO_AUTH_TOKEN=your_auth_token
  TWILIO_WHATSAPP_FROM=+14155238886
  ```

### 7. Run

```bash
python run_homeshield.py
```

Then open the dashboard at **http://localhost:5000/**.

Common flags:

```bash
# Custom port and DB
python run_homeshield.py --port 8080 --db custom.db

# Bind only to localhost (default 0.0.0.0 lets phones on the LAN reach it)
python run_homeshield.py --host 127.0.0.1

# Don't auto-start cameras on boot (useful for debugging)
python run_homeshield.py --no-autostart
```

---

## 📸 Camera setup

HomeShield accepts any source OpenCV's `VideoCapture` understands — webcams, RTSP / HTTP streams, and video files. Add cameras from **Live feeds → Add camera** in the dashboard.

### Built-in webcam
Use the integer device index. `0` is your default webcam:

```
Source: 0
```

If you have multiple webcams, try `1`, `2`, etc.

### IP cameras over RTSP (Tapo, VIGI, Hikvision, Dahua, …)
HomeShield works with any IP camera that exposes an **RTSP** stream. The general form is:

```
Source: rtsp://<user>:<password>@<camera-ip>:<port>/<stream-path>
```

Most consumer cameras use port `554`. Many manufacturers offer a high-quality main stream and a lower-bitrate sub-stream — **the sub-stream is strongly recommended for 24/7 monitoring** because it stays well within the GPU / CPU budget when you're running multiple cameras side-by-side. Below are the URL patterns for the brands HomeShield has been validated against.

#### TP-Link Tapo (C100, C200, C210, C220, C310, C320WS, etc.)
1. Open the **Tapo app → your camera → Camera Settings → Advanced Settings → Camera Account**.
2. Toggle **Camera Account** on and set a username + password (this is the RTSP credential, separate from your Tapo login).
3. Use the URL:

```
Source: rtsp://<user>:<password>@<camera-ip>:554/stream1   # high quality (2K / 1080p)
Source: rtsp://<user>:<password>@<camera-ip>:554/stream2   # lower quality, lower bandwidth
```

#### TP-Link VIGI (C300, C400, C540, NVR channels)
1. In the **VIGI Security Manager** (or web UI) → **Settings → Network → Advanced → RTSP** — make sure RTSP is enabled and note the port (default `554`).
2. Create an ONVIF / RTSP account under **Settings → System → User Management**.
3. URL pattern:

```
Source: rtsp://<user>:<password>@<camera-ip>:554/stream1   # main stream
Source: rtsp://<user>:<password>@<camera-ip>:554/stream2   # sub stream
```

For VIGI NVR channels, append the channel number, e.g. `…/stream1?channel=2`.

#### Hikvision (DS-2CD…, Ezviz under the hood)
1. Web UI → **Configuration → Network → Advanced Settings → Integration Protocol** — enable **ONVIF** and create an ONVIF user, or use the admin account.
2. Web UI → **Configuration → System → Security → Authentication** — set **RTSP Authentication** to `digest/basic`.
3. URL pattern:

```
Source: rtsp://<user>:<password>@<camera-ip>:554/Streaming/Channels/101   # main stream, ch 1
Source: rtsp://<user>:<password>@<camera-ip>:554/Streaming/Channels/102   # sub stream, ch 1
```

For multi-channel NVRs, the channel encoding is `<channel><stream>` — e.g. `201` = channel 2 main, `202` = channel 2 sub.

#### Dahua (IPC-HDW…, Lorex, Amcrest rebrands)
1. Web UI → **Setup → Network → Connection → RTSP** — confirm port (default `554`).
2. Create or reuse an admin / operator account under **Setup → System → Account**.
3. URL pattern:

```
Source: rtsp://<user>:<password>@<camera-ip>:554/cam/realmonitor?channel=1&subtype=0   # main stream
Source: rtsp://<user>:<password>@<camera-ip>:554/cam/realmonitor?channel=1&subtype=1   # sub stream
```

`subtype=0` is the main stream, `subtype=1` is the sub stream. Change `channel=` for NVR channels.

#### Generic ONVIF / other brands
If your camera isn't listed above, check the manufacturer's docs for "RTSP URL" or "ONVIF stream URI". Tools like **ONVIF Device Manager** or **VLC → Media → Open Network Stream** can probe a camera and reveal its working RTSP path. Once you have the URL, paste it into HomeShield's **Live feeds → Add camera → Source** field exactly as-is.

> 💡 **TCP vs UDP:** if the stream connects but stutters or freezes, OpenCV may be defaulting to UDP transport on a lossy network. You can force TCP by exporting `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` before launching HomeShield.

### Android phone as camera (DroidCam, IP Webcam)
Both apps expose a phone's camera as a network stream:

- **DroidCam** (Wi-Fi mode):
  ```
  Source: http://<phone-ip>:4747/video
  ```
- **IP Webcam**:
  ```
  Source: http://<phone-ip>:8080/video
  ```

Make sure both devices are on the same Wi-Fi network and the phone screen stays on.

### Video file for testing
Drop in any local video file to dry-run the detectors without a live camera:

```
Source: C:/path/to/sample_fall.mp4
Source: ./test_videos/kitchen_fire.mp4
```

The pipeline loops the file and plays it at its real frame rate. Detection timing uses the video's own clock, so results match what a live camera would produce even if your GPU processes the file slower than real time.

---

## 🎯 Using the system

### Register your family members
1. Make sure face recognition is on (**Settings → Face recognition**); registering needs it.
2. Go to **People → Register a person** and enter the person's name and category.
3. Pick a camera, ask them to look straight at it, and press **Capture from camera** once the preview says **FACE OK**.
4. Repeat for every household member.

> ⚠️ **Important:** Register everyone soon after turning face recognition on. Until then, household members show up under **People → Intruders seen**; you can register them straight from there.

### Set up zones
1. Open **Zones**, pick a camera from the list.
2. Click on the live preview to lay polygon vertices, then close the polygon.
3. Tag the zone:
   - **Safe zone** → suppresses lying / inactivity alerts inside it (use this for beds, sofas, recliners).
   - **Danger zone** → triggers a child-entry alert when anyone enters (kitchen, balcony, pool, stairs).
4. Save. Zones apply live, no restart needed.

### Watch the live feed
**Live feeds** is the main dashboard:
- A one-line verdict at the top says **All clear** or how many incidents need attention (it says **Not monitoring** when the system is stopped, rather than claiming all is well).
- Each camera shows the annotated MJPEG stream with bounding boxes, pose skeletons, and face labels overlaid.
- The incident board lists open incidents first, then the ones recently handled, and updates in real time via Server-Sent Events.

### Handle incidents
Click (or tap) an incident to open it: the most confident snapshot, how long it lasted, how many detections it grouped, and a timeline of the first, peak, escalating and last detections.
- **View live** jumps to that camera.
- **Acknowledge** — you've seen it and dealt with it. **False alarm** — the detector was wrong. Either can carry a short note, and either can be changed later.
- If a closed incident gets worse (a fall turns into "lying motionless"), it reopens marked **Escalated**.
- **Incidents** shows every incident with status, type and camera filters, the 7-day false-alarm counts, and a **Detection log** view of the individual detections. **Clear history** (admins only) deletes incidents and the log.

### Handle intruders
When an unknown face appears, an entry pops into **People → Intruders seen** with a snapshot and timestamp. From there you can:
- **Register** them (name and category) if it's someone you know.
- **Dismiss** the entry if it was a false positive (poor lighting, motion blur, partial face).
- **Delete** a dismissed entry and its photo for good (tick **Show dismissed** to see them).

---

## ⚙️ Tuning for your hardware

All knobs live in **Settings** and apply live without restarting cameras. Changing the pose / fire model file or the FP16 switch reloads that model automatically.

| Setting | Effect | When to adjust |
|---|---|---|
| **Pose model** | Smaller (`yolo11n-pose`) = faster, less accurate. `yolo26x-pose` = slowest, best accuracy. | Drop to `n` on CPU or low-end GPU. |
| **imgsz** | Inference resolution (320 / 416 / 640). | Lower for more FPS, higher for far-away subjects. |
| **Target FPS cap** | Caps the capture loop. | Set to 10–15 on CPU. |
| **FP16 (half-precision)** | Halves model memory on NVIDIA GPUs. At one frame per call the speed gain is small (~3% for `yolo11x-pose` on an RTX 4070); detections are unchanged (keypoints within 0.1 px of FP32). | Leave **on**; it is ignored on CPU. |
| **Frame skipping** | Pose runs every frame; fire every 2nd; face every 5th. | Bump face skip if face inference is the bottleneck. |
| **Fire / Face enabled** | Toggle the heavier detectors. | Disable face on CPU-only setups. |
| **Sensitivity (fall)** | Adjusts the FSM thresholds for descent velocity and lying duration. | Raise if false-falls are common; lower if real falls are missed. |
| **Cooldowns** | Min seconds between repeat alerts of the same class. | Raise for noisy environments. |

Advanced settings (not in the UI yet; set them with `POST /api/settings`):

| Key | Default | Effect |
|---|---|---|
| `fire_confirm_frames` / `fire_confirm_window` | 3 / 5 | Fire or smoke must be detected in 3 of the last 5 fire checks before alerting. Rejects one-frame false positives (lamps, sunsets, orange clothing). |
| `face_min_size` / `face_min_det_score` | 40 px / 0.6 | Faces smaller or less certain than this are shown as `?` and never matched, so they cannot raise intruder alerts. Also applied when enrolling a person. |
| `intruder_confirm_frames` | 2 | An unknown face must appear in this many consecutive face checks before the intruder alert. |

---

## 🏗️ Architecture

HomeShield is a single Flask process that runs one **CameraManager** with one worker thread per camera. Every worker shares a single set of YOLO / InsightFace models on the GPU and publishes results to an async **EventBus** that handles SQLite writes, snapshot encoding, and Server-Sent Events to the browser.

Key design choices:
- **Per-camera worker threads** share one set of YOLO / ONNX models on the GPU. Adding a camera does not duplicate VRAM.
- **Per-camera tracking** — each camera keeps its own ByteTrack state even though the pose model is shared, so one camera's frames never update or expire another camera's person IDs.
- **Newest-frame capture** — live cameras are read on a separate grabber thread and the pipeline always takes the newest frame, so a slow GPU never makes the feed (and the alerts) drift behind real time. Video files are instead processed frame by frame and timed by the file's own clock.
- **Frame skipping**: pose runs every frame to keep the FSM responsive; fire runs every 2nd frame; face every 5th.
- **Async face inference** — face detection runs on its own daemon thread per camera so the capture loop runs at pose-only speed regardless of how slow `app.get()` is. A recognised face is attached to the tracked person, so fall events name the person and child danger-zone alerts work.
- **Lazy stream encoding** — a frame is JPEG-encoded only when someone is watching that camera, on the viewer's thread.
- **Non-blocking event publishing** offloads snapshot encoding and SQLite writes to a daemon thread, so the capture loop never waits on disk I/O.
- **Server-side incidents** — the publisher files each detection into its incident in the same SQLite transaction (`homeshield/incidents.py`), so every phone and laptop sees the same incidents and the same resolutions. Events logged by older versions are grouped once on upgrade.
- **Hot reloading** — toggling `fire_enabled` / `face_enabled`, switching model files or tweaking imgsz / FPS / thresholds applies live without restarting cameras.

### Tests

```bash
python -m pytest
```

Install it first with `pip install pytest`. The suite runs in a few seconds without a GPU or model weights. It covers the fall FSM on synthetic skeleton tracks (falls vs. sitting, bending and lying down slowly, at 3–30 FPS), per-camera tracker isolation, fire and intruder confirmation, zones, face matching, the frame grabber, incident grouping/escalation/resolution, per-incident WhatsApp, and API input validation.

---

## 📁 Project structure

```
FYP/
├── Fall_Detection/                 # Pose weights + fall-detection guide
│   ├── weights/                    #   drop any *-pose.pt YOLO weights here
│   │   ├── yolo11n-pose.pt
│   │   ├── yolo11m-pose.pt
│   │   ├── yolo11x-pose.pt
│   │   ├── yolo26n-pose.pt
│   │   ├── yolo26m-pose.pt
│   │   └── yolo26x-pose.pt
│   └── README.md
│
├── Fire_Detection/                 # Fire/smoke weights
│   └── best.pt                     #   custom-trained weights
│
├── Face_Detection/                 # Bundled InsightFace wheels (Windows)
│   ├── insightface-0.7.3-cp310-cp310-win_amd64.whl
│   └── insightface-0.7.3-cp311-cp311-win_amd64.whl
│
├── homeshield/                     # The unified Flask app
│   ├── __init__.py
│   ├── server.py                   #   Flask API + MJPEG endpoints + SSE
│   ├── auth.py                     #   user accounts + login / session middleware
│   ├── pipeline.py                 #   per-camera pipeline + Models + FaceWorker
│   ├── cameras.py                  #   multi-camera lifecycle manager
│   ├── events.py                   #   async EventBus + SQLite log + snapshots
│   ├── incidents.py                #   detections grouped into incidents + resolutions
│   ├── notify.py                   #   per-incident WhatsApp alerts (Twilio, opt-in)
│   ├── persons.py                  #   registered persons + intruder log
│   ├── zones.py                    #   polygon zone storage + point-in-polygon
│   ├── settings.py                 #   hot-reloadable settings store
│   ├── annotator.py                #   bounding boxes, pose skeleton, labels
│   ├── db.py                       #   SQLite (WAL mode) connection + schema
│   ├── paths.py                    #   model weights folder locations
│   ├── detectors/                  #   detection back-ends (each runnable via python -m)
│   │   ├── fall/                   #     YOLO pose features + 7-state FSM + drawing
│   │   ├── fire.py                 #     YOLO fire/smoke wrapper + CLI
│   │   └── face.py                 #     InsightFace embedding + matching helpers
│   ├── static/fonts/               #   Chakra Petch + Rubik, served locally (OFL)
│   └── templates/
│       └── index.html              #   single-page dashboard UI (plain HTML, CSS and JS)
│
├── tests/                          # pytest suite (+ synthetic skeleton tracks)
├── run_homeshield.py               # Entry point (argparse + create_app)
├── reset-password.bat              # Double-click recovery when no admin can sign in
├── pyproject.toml                  # Package metadata + pinned, ABI-consistent deps
├── requirements.txt                # Installs the package (-e .[plot,cuda12])
├── homeshield.db                   # SQLite event/persons/zones/users store (created on first run)
├── snapshots/                      # Annotated event JPEGs (gitignored)
├── person_photos/                  # Enrolment photos for known persons (gitignored)
├── intruder_photos/                # Auto-logged unknown-face snapshots (gitignored)
├── .gitignore
├── .gitattributes
└── README.md                       # ← you are here
```

---

## 🔐 Privacy & security

- **All inference runs locally.** No frames, embeddings, or events ever leave your machine. There is no cloud component, no telemetry, and no third-party API calls during detection — HomeShield only touches the network to pull RTSP/HTTP streams from cameras you explicitly add.
- **The one opt-in exception: WhatsApp alerts.** If you add Twilio credentials, each new or escalated incident sends a short text (what, which camera, when, confidence) through Twilio to the numbers you list. No images or face data are sent. Leave the Twilio details empty (in Settings and `.env`) to keep HomeShield fully offline. A token saved in Settings sits in `homeshield.db`, so treat that file as a secret (it already holds camera passwords).
- **Built-in user authentication.** The dashboard ships with a login layer (`homeshield/auth.py`) backed by hashed passwords in the local SQLite DB. The first run seeds `admin` / `admin` and forces a new password at the first sign-in; every API and stream endpoint then requires an authenticated session. Admins manage accounts and reset passwords in **Settings → Users**.
  - **Remember me** keeps a device signed in for 30 days; without it, sign-in lasts until the browser closes (12 hours at most).
  - **Lockout:** 5 wrong passwords for one account from one device within 10 minutes pause that account's sign-in from there for 5 minutes (20 from one device across accounts pause the device).
  - **Forgot password:** there is no email. The sign-in page sends the request to the admins, who see **RESET REQUESTED** in Settings → Users and give a temporary password. If the only admin forgets, run `python run_homeshield.py --reset-password <username>` on the HomeShield computer; it prints a temporary password that must be changed at sign-in.
  - Sessions are signed with `homeshield_secret.key`, created beside the database on first run (git-ignored) so sign-ins survive a restart. Delete it to sign everyone out, or set `HOMESHIELD_SECRET` to use your own key.
- **Face embeddings, not photos**, are used for matching at runtime. The 512-dim ArcFace vectors live in the local SQLite DB. Original enrolment photos are kept in `person_photos/` so you can re-enrol after a model swap.
- **Intruder snapshots** are stored in `intruder_photos/` on disk. Delete them whenever you want — the entry in the UI will disappear with them. The same applies to event snapshots in `snapshots/`.
- **Network exposure.** By default the server binds to `0.0.0.0:5000`, so any authenticated user on your local network can reach the dashboard. To restrict it to the same machine, run with `--host 127.0.0.1`. For remote access, **do not expose port 5000 directly to the public internet** — put HomeShield behind a reverse proxy with HTTPS (Caddy, nginx, Cloudflare Tunnel) or a private network (Tailscale, WireGuard).
- **RTSP credentials** for IP cameras are stored in plaintext inside the SQLite DB so that camera workers can reconnect after a restart. Treat `homeshield.db` like any other secret file — anyone with read access to it can pull your camera passwords. Back it up to an encrypted volume if you're storing it outside your machine.
- **Defence in depth.** Make sure your IP cameras themselves use **strong, unique RTSP passwords** (not the default `admin/admin`), keep their firmware up to date, and put them on a VLAN or isolated Wi-Fi SSID that cannot reach the rest of your LAN. HomeShield is only as secure as the cameras feeding it.

---

## 🧪 Performance notes

Indicative numbers on an **RTX-class GPU + Ryzen / Intel desktop CPU**, measured in the dashboard's FPS chip:

| Setup | Cameras | Pose model | imgsz | Detectors on | Avg FPS / cam |
|---|---|---|---|---|---|
| RTX 3060, FP16 | 1× 1080p | `yolo11n-pose` | 640 | Fall | ~55 |
| RTX 3060, FP16 | 1× 1080p | `yolo11n-pose` | 640 | Fall + Fire + Face | ~28 |
| RTX 3060, FP16 | 3× 1080p | `yolo11n-pose` | 640 | Fall + Fire + Face | ~14 each |
| RTX 3060, FP16 | 1× 1080p | `yolo26x-pose` | 640 | Fall + Fire + Face | ~9 |
| Laptop CPU only | 1× 720p | `yolo11n-pose` | 416 | Fall + Fire | ~8 |
| Laptop CPU only | 1× 720p | `yolo11n-pose` | 416 | Fall + Fire + Face | ~2 (unusable) |

Bottlenecks, in practice:
- **Face inference on CPU** is the single biggest performance cliff — keep it on GPU or off entirely.
- **RTSP decode** (especially 2K / 4K main streams on Tapo, Hikvision, or Dahua) eats a surprising amount of CPU. Prefer the **sub-stream** for 24/7 use across all brands.
- **Disk I/O** never blocks the capture loop because event publishing is fully async.

---

## 🔧 Troubleshooting

**Forgot the admin password**
If another admin can sign in, they can reset it in **Settings → Users**. Otherwise, on the HomeShield computer, double-click **`reset-password.bat`** in the project folder and type the username (Enter means `admin`). It finds HomeShield's Python on its own; if it can't, set `HOMESHIELD_PYTHON` to that `python.exe`, or run `conda activate homeshield` then `python run_homeshield.py --reset-password admin`. Sign in with the temporary password it prints and choose a new one.

**`numpy.dtype size changed, may indicate binary incompatibility`**
You ended up on NumPy 2.x. Pin back to NumPy 1.x: `pip install "numpy<2.0"`.

**`torch.cuda.is_available()` is `False`**
You either installed the CPU-only PyTorch build, or your NVIDIA driver / CUDA runtime is mismatched. Reinstall PyTorch from the CUDA index you chose in step 3 (e.g. `--index-url https://download.pytorch.org/whl/cu130`) **after** uninstalling the existing torch.

**InsightFace fails to install on Windows**
Use the bundled wheel: `pip install Face_Detection/insightface-0.7.3-cp311-cp311-win_amd64.whl`. The wheel is built for **Python 3.11 / Windows x64** specifically.

**Pose model dropdown is empty**
HomeShield only lists files in `Fall_Detection/weights/` whose names end in `-pose.pt`. Heavy `x` variants exceed GitHub's per-file limit and are excluded by `.gitignore`. Download them from <https://github.com/ultralytics/assets/releases> or train your own.

**RTSP camera connects but stutters**
You're probably on the main / high-quality stream. Switch to the camera's sub-stream (e.g. `/stream2` on Tapo/VIGI, `/Streaming/Channels/102` on Hikvision, `subtype=1` on Dahua). If it still stutters, try forcing TCP transport via `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`, and check that your CPU isn't pinned at 100% from H.264/H.265 decode.

**Phone (DroidCam / IP Webcam) won't connect**
Both devices must be on the **same Wi-Fi network** (not Wi-Fi vs guest network, not cellular). Open the URL in a browser first to confirm the stream is reachable, then paste it into HomeShield.

**Every family member gets logged as an intruder**
They haven't been registered yet. Register them from **People → Intruders seen** (or with **Register a person**); new detections will then name them.

**Dashboard reachable from your laptop but not your phone**
Server is bound to `127.0.0.1`. Restart with `--host 0.0.0.0` (the default), and check Windows Firewall isn't blocking inbound TCP/5000.

**Fall alerts on someone just sitting down**
Lower the **fall sensitivity** in Settings, or raise the descent-velocity threshold. The FSM also requires sustained horizontal posture, so very brief lie-downs shouldn't trigger.

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` (or the header in `run_homeshield.py`) for the full text.

Third-party model weights and libraries retain their own licenses:
- **Ultralytics YOLO** — AGPL-3.0 (commercial use requires an Ultralytics enterprise license).
- **InsightFace** — MIT.
- **PyTorch, OpenCV, Flask, NumPy, SciPy, Shapely** — their respective open-source licenses.

---

## 🙏 Acknowledgements

- **Developer:** Mohammad Salehuddin bin Iwan
- **Supervisor:** Andi Fitriah binti Abdul Kadir
- **Institution:** International Islamic University Malaysia (IIUM), **Kulliyyah of Information and Communication Technology**
- **Project title:** Final Year Project — *HomeShield: Centralized Dashboard for Real-Time CCTV Monitoring and Anomaly Detection*

Built on the shoulders of:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — pose, fire, and smoke detection.
- [InsightFace](https://github.com/deepinsight/insightface) — ArcFace face recognition.
- [PyTorch](https://pytorch.org/), [ONNX Runtime](https://onnxruntime.ai/), [OpenCV](https://opencv.org/), [Flask](https://flask.palletsprojects.com/), [Shapely](https://shapely.readthedocs.io/).

Special thanks to the open-source CV community for making real-time anomaly detection on consumer hardware actually viable.
