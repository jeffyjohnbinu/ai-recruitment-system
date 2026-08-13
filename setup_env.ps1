$ErrorActionPreference = "Stop"

Write-Host "==> Checking Python version"
python --version

Write-Host "==> Upgrading pip"
python -m pip install --upgrade pip

Write-Host "==> Installing project dependencies"
python -m pip install -r requirements.txt

Write-Host "==> Installing dev dependencies"
python -m pip install -r requirements-dev.txt

Write-Host "==> Installing pre-commit"
pre-commit install

Write-Host "==> Downloading spaCy English model"
python -m spacy download en_core_web_sm

if (-Not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env - fill in your API keys before running the system."
}

Write-Host ""
Write-Host "Environment ready."
Write-Host "Run tests with: pytest"