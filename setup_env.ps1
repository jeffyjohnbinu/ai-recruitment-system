# =============================================================================
# setup_env.ps1 — bootstrap the AI Recruitment System dev environment (Windows)
# Usage:  powershell -ExecutionPolicy Bypass -File setup_env.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host "==> Checking Python version"
python --version

Write-Host "==> Creating virtual environment in .\.venv"
python -m venv .venv

Write-Host "==> Activating virtual environment"
.\.venv\Scripts\Activate.ps1

Write-Host "==> Upgrading pip"
pip install --upgrade pip

Write-Host "==> Installing project dependencies"
pip install -r requirements.txt

if ($args[0] -eq "--dev") {
    Write-Host "==> Installing dev dependencies"
    pip install -r requirements-dev.txt
    pre-commit install
}

Write-Host "==> Downloading spaCy English model"
python -m spacy download en_core_web_sm

if (-Not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env — fill in your API keys before running the system."
}

Write-Host ""
Write-Host "Environment ready. Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Run tests with: pytest"
