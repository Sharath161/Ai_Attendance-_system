# Smart Attendance System — Windows first-time setup
# Run from the project root: .\scripts\setup.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

Write-Host ""
Write-Host "Smart Attendance System — Setup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# 1. Copy .env if not present
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "[1/4] Created .env from .env.example" -ForegroundColor Green
    Write-Host "      Edit .env to set your DATABASE_URL if using PostgreSQL." -ForegroundColor Yellow
} else {
    Write-Host "[1/4] .env already exists — skipping" -ForegroundColor DarkGray
}

# 2. Create work directories
foreach ($dir in @("work\models", "work\spool", "work\registrations", "work\logs")) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
Write-Host "[2/4] Created work/ directories" -ForegroundColor Green

# 3. Install Python dependencies
Write-Host "[3/4] Installing Python packages..." -ForegroundColor Cyan
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
Write-Host "      Dependencies installed." -ForegroundColor Green

# 4. Download AI models
Write-Host "[4/4] Downloading AI models (YuNet + ArcFace)..." -ForegroundColor Cyan
python -m worker.download_models
if ($LASTEXITCODE -ne 0) { throw "Model download failed" }

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Start the API:    python start_server.py"
Write-Host "  2. Open browser:     http://localhost:8000"
Write-Host "  3. Register a face:  http://localhost:8000/register/webcam"
Write-Host "  4. Face kiosk:       http://localhost:8000/checkin"
Write-Host "  5. Admin dashboard:  http://localhost:8000/admin"
Write-Host ""
