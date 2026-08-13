#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — bootstrap the AI Recruitment System dev environment
# Usage:  bash setup_env.sh
# =============================================================================
set -e  # exit on first error

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

echo "==> Checking Python version"
$PYTHON_BIN --version

echo "==> Creating virtual environment in ./${VENV_DIR}"
$PYTHON_BIN -m venv "$VENV_DIR"

echo "==> Activating virtual environment"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing project dependencies"
pip install -r requirements.txt

if [ "$1" == "--dev" ]; then
  echo "==> Installing dev dependencies"
  pip install -r requirements-dev.txt
  pre-commit install || true
fi

echo "==> Downloading spaCy English model (used by parsers/)"
python -m spacy download en_core_web_sm || echo "WARNING: spaCy model download failed — check network access."

echo "==> Creating .env from template (if not present)"
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env — fill in your API keys before running the system."
fi

echo ""
echo "✅ Environment ready."
echo "   Activate it any time with:  source ${VENV_DIR}/bin/activate"
echo "   Run the test suite with:    pytest"
