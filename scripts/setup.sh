#!/usr/bin/env bash
# Smart Attendance System — Linux/macOS first-time setup
# Run from the project root: bash scripts/setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ""
echo "Smart Attendance System — Setup"
echo "================================"
echo ""

# 1. Copy .env if not present
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[1/4] Created .env from .env.example"
    echo "      Edit .env to set DATABASE_URL if using PostgreSQL."
else
    echo "[1/4] .env already exists — skipping"
fi

# 2. Create work directories
mkdir -p work/models work/spool work/registrations work/logs
echo "[2/4] Created work/ directories"

# 3. Install Python dependencies
echo "[3/4] Installing Python packages..."
pip install -r requirements.txt
echo "      Done."

# 4. Download AI models
echo "[4/4] Downloading AI models (YuNet + ArcFace)..."
python -m worker.download_models

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Start the API:   python start_server.py"
echo "  2. Open browser:    http://localhost:8000"
echo "  3. Register faces:  http://localhost:8000/register/webcam"
echo "  4. Face kiosk:      http://localhost:8000/checkin"
echo "  5. Admin panel:     http://localhost:8000/admin"
echo ""
