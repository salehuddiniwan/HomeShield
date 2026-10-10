# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: a non-technical homeowner or parent who monitors cameras passively and acts on alerts. Comfortable with everyday technology but has no background in video surveillance or machine learning. Typically accesses the dashboard from a phone or laptop on the home Wi-Fi.

Secondary: an academic examiner or supervisor evaluating the FYP — looking for demonstrable technical depth alongside a polished, understandable interface.

The admin who sets the system up may be more technically capable, but the day-to-day user experience must be legible to a non-expert family member with no training.

## Product Purpose

HomeShield is a locally-hosted CCTV monitoring dashboard that runs three GPU-accelerated anomaly detectors — fall detection, fire/smoke detection, and face/intruder detection — side-by-side on every connected camera feed. A single unified pipeline annotates and streams live MJPEG video to the browser, logs every alert to SQLite with an annotated snapshot, and pushes real-time updates via Server-Sent Events.

Success means a homeowner can monitor their family's safety from their phone without understanding the underlying models, while the admin can configure cameras, register household members, draw detection zones, and tune thresholds without touching code.

## Positioning

The only home CCTV dashboard that runs three production-quality anomaly detectors concurrently on consumer hardware with zero cloud dependency — no subscription, no data leaving the home, no third-party API calls during detection. The fall detector's 7-state finite-state machine (Standing → Walking → Sitting → Fall_Detected → Lying_After_Fall → Lying_Motionless → Inactivity) is tuned to reject controlled sit-downs and false positives through a two-stage decision on descent velocity and sustained horizontal posture — a research contribution, not just a YOLO wrapper.

## Operating Context

- Runs as a persistent Flask process on the household's Windows 11 machine (NVIDIA GPU recommended)
- Family members and the admin access the dashboard from phones and laptops on the same local Wi-Fi network
- Cameras include built-in webcams, RTSP IP cameras (Tapo, VIGI, Hikvision, Dahua), Android phones via DroidCam / IP Webcam, and video files for testing
- Designed for 24/7 background operation; camera connections auto-reconnect; event log accumulates indefinitely
- Admin registers household members before enabling face detection; intruder snapshots are manually reviewed and actioned from the dashboard

## Capabilities and Constraints

**Detectors:**
- Fall detection: 7-state FSM on YOLO pose keypoints; configurable sensitivity, inactivity timeout, and safe/danger zones
- Fire & smoke: custom YOLO weights, per-class cooldown, confirmation over a sliding window to reject one-frame false positives
- Face/intruder: InsightFace ArcFace (buffalo_l) 512-dim embeddings, cosine-matched against a registered Persons gallery; unknown faces auto-logged as intruders

**Zones:** polygon safe zones (suppress lying/inactivity alerts on beds/sofas) and danger zones (child-entry alerts for kitchens, balconies, stairs)

**Performance:** per-camera worker threads share one set of YOLO/ONNX models on the GPU; adding a camera does not duplicate VRAM. Async face inference, lazy JPEG encoding, and non-blocking event publishing keep the capture loop from stalling on I/O

**Platform constraints:** Windows-primary (bundled InsightFace wheel for CPython 3.11 x64); face detection on CPU is unusable in practice and should be disabled on CPU-only setups

**Settings are hot-reloadable** — toggling detectors, swapping models, or changing thresholds applies live without restarting cameras

**Auth:** session-based login backed by hashed passwords in local SQLite; admin and guest roles

**Undecided:** no WCAG conformance target established; no accessibility audit performed

## Brand Commitments

- Product name: **HomeShield**
- Logo: `Icon/LOGO.svg` (exists; not locked to current execution)
- Companion icons (Detection, Fall, Fire, Face, Camera, Register, Notification) drawn inline in the page; the source SVGs were moved out of the repo to `../FYPClaude_archive/Icon/`
- No locked color palette, typography, or visual style — all open for design decisions

## Evidence on Hand

- `README.md` — complete feature specification, installation guide, hardware tuning table, RTSP brand reference
- UML diagrams (component, class, activity, sequence, deployment, FSMs) are archived outside the repo in `../FYPClaude_archive/UML_Diagrams/`
- `tests/` — synthetic fall-track pytest suite covering the FSM, per-camera tracker isolation, fire/intruder confirmation, zones, and face matching
- `homeshield/templates/index.html` — the shipped UI: one plain HTML page with its CSS and JS inline
- `homeshield/static/fonts/` — self-hosted Chakra Petch and Rubik

No testimonials, customer quotes, usage metrics, or third-party benchmarks on hand; do not fabricate them.

## Product Principles

1. **Protect without intrusion** — all inference is local; no frames, embeddings, or events ever leave the home network
2. **Trustworthy alerts** — multi-stage confirmation (FSM thresholds, sliding-window fire confirm, intruder consecutive frames) prevents false positives from eroding trust
3. **Accessible by design** — a non-technical family member should never need to understand the underlying ML to feel safe using it; complexity is the admin's problem, not the family's
4. **Transparent in operation** — every detection is annotated with overlays, persisted with a snapshot, and visible in the event log; nothing happens silently
5. **Hardware-honest** — the system exposes its hardware requirements clearly and degrades gracefully (disabling face detection on CPU) rather than silently performing below expectation

## Accessibility & Inclusion

Mobile-responsive UI (Flask, accessible from phones on the local network). No specific WCAG conformance target has been established for this project.
