# Smart Attendance System

AI-powered face-recognition attendance system for university deployments.  
Images are captured by ESP32-CAM kiosks (or any browser webcam), processed on a local server with ArcFace recognition, and surfaced through a student portal and admin dashboard.

---

## Architecture

```
ESP32-CAM / Browser Webcam
        │  HTTP POST /ingest/image  (X-Device-Key auth)
        ▼
  FastAPI Server — port 8000
   ├── /register/webcam   face enrolment (5-angle KYC capture)
   ├── /checkin           live recognition kiosk
   ├── /student           student attendance portal
   ├── /admin             admin dashboard + fairness KPIs
   └── /docs              OpenAPI
        │
        ▼  ImageJob rows  (SQLite dev / PostgreSQL prod)
  Batch Worker  (runs every ~2 h)
   ├── YuNet detection     OpenCV Zoo ONNX
   ├── ArcFace 512-d embed InsightFace w600k_mbf.onnx
   └── Cosine similarity   per-demographic threshold
        │
        ▼
  AttendanceEvent  →  student portal reflects within 12 h
```

**Racial fairness** is a first-class concern.  
Per-group thresholds (FairFace 7 labels) target F1 ≥ 0.75 for every demographic group with ≤ 15 pp inter-group gap.

---

## Quick Start

### Prerequisites

- Python 3.11 or 3.12
- Git

### Windows

```powershell
git clone <repo-url>
cd develop-an-efficient-and-scalable-backend
.\scripts\setup.ps1
python start_server.py
```

### Linux / macOS

```bash
git clone <repo-url>
cd develop-an-efficient-and-scalable-backend
bash scripts/setup.sh
python start_server.py
```

The setup script copies `.env.example` → `.env`, creates `work/` directories, installs dependencies, and downloads the AI models (~14 MB).

---

## Pages

| URL | Purpose |
|-----|---------|
| `/` | Home — links to all portals |
| `/register/webcam` | Enrol a student's face (5-angle guided capture) |
| `/checkin` | Face-recognition kiosk (simulates ESP32-CAM) |
| `/student` | Student attendance portal |
| `/admin` | Admin dashboard — log, manual override, devices, fairness |
| `/docs` | FastAPI interactive API docs |

---

## Configuration

Copy `.env.example` to `.env` and edit as needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | SQLite | Use PostgreSQL URL for production |
| `MATCH_THRESHOLD` | `0.35` | ArcFace cosine similarity cut-off |
| `RACE_THRESHOLDS` | `{}` | Per-group overrides e.g. `{"black": 0.30}` |
| `DETECTION_SCORE_THRESHOLD` | `0.60` | YuNet face-detection confidence minimum |
| `AUTO_CREATE_TABLES` | `true` | Set `false` when using Alembic migrations |

---

## Production — Docker Compose

```bash
# Build and start API + batch worker + PostgreSQL
docker-compose up --build -d

# Download AI models into the Docker volume
docker-compose exec api python -m worker.download_models

# Tail logs
docker-compose logs -f api worker
```

| Service | Port | Role |
|---------|------|------|
| `api` | 8000 | FastAPI + HTML pages |
| `worker` | — | Batch processing loop |
| `db` | 5432 | PostgreSQL 16 |

---

## Project Structure

```
├── api/
│   ├── main.py                 All API routes
│   ├── index.html              Landing page
│   ├── checkin.html            Face-recognition kiosk
│   ├── admin.html              Admin dashboard
│   ├── student.html            Student portal
│   └── webcam_register.html    Face enrolment
│
├── core/
│   ├── config.py               Pydantic settings (reads .env)
│   ├── database.py             SQLAlchemy async engine
│   ├── models.py               ORM models
│   └── math_utils.py           l2_normalize helper
│
├── worker/
│   ├── model_adapter.py        YuNet + ArcFace ONNX pipeline
│   ├── registration_updater.py Processes pending face registrations
│   ├── runner.py               Polling worker loop
│   └── download_models.py      Model download script
│
├── migrations/
│   └── 001_initial.sql         Full PostgreSQL schema + indexes
│
├── tests/
│   ├── fairness/               Per-demographic F1 evaluation
│   ├── load/                   Locust load tests
│   ├── stress/                 Data seeding + throughput
│   └── validation/             Recognition accuracy evaluation
│
├── firmware/
│   └── esp32_cam/esp32_cam.ino Arduino sketch for ESP32-CAM
│
├── scripts/
│   ├── setup.ps1               Windows first-run setup
│   └── setup.sh                Linux/macOS first-run setup
│
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt            Production dependencies
├── requirements-dev.txt        Development + test dependencies
├── .env.example                Documented configuration template
└── start_server.py             Entry point
```

---

## AI Pipeline

### Detection — YuNet (OpenCV Zoo)
- Model: `face_detection_yunet_2023mar.onnx` (234 KB)
- Outputs 5-point landmarks for affine alignment
- CPU inference via OpenCV

### Recognition — ArcFace MobileNet
- Model: `w600k_mbf.onnx` (13 MB, InsightFace buffalo_sc)
- 512-dimensional L2-normalised embeddings
- Similarity: cosine similarity via dot product

### Fairness
- 7-group FairFace labels stored per embedding
- Per-group thresholds configurable via `RACE_THRESHOLDS`
- Admin → Fairness KPIs shows live per-group accuracy bars
- Target: F1 ≥ 0.75, inter-group gap ≤ 15 pp

---

## Makefile

```bash
make setup        # Install deps + download models
make run          # Start server
make worker       # Start batch worker
make test         # Run pytest suite
make docker-up    # Build + start Docker stack
make docker-down  # Stop Docker stack
make clean        # Remove __pycache__ / .pyc
make reset-db     # Delete SQLite DB (dev only)
```

---

## ESP32-CAM Hardware

Flash `firmware/esp32_cam/esp32_cam.ino` via Arduino IDE.  
Edit the configuration at the top of the sketch:

```cpp
const char* WIFI_SSID      = "YOUR_NETWORK";
const char* WIFI_PASSWORD  = "YOUR_PASSWORD";
const char* SERVER_HOST    = "192.168.1.100";  // server LAN IP
const int   SERVER_PORT    = 8000;
const char* DEVICE_API_KEY = "<key from Admin > Devices tab>";
const char* CLASS_ID       = "CS301";
```

Register the device via **Admin → Devices → Register Device** to get its API key.
