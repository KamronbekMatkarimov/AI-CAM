#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f requirements.txt ]; then
  echo "requirements.txt topilmadi. Iltimos, loyiha ildiz papkasida ishlating."
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir -p outputs/raw outputs/processed outputs/crops outputs/logs outputs/uploads outputs/uploads_processed

API_KEY="${API_KEY:-abc123}"
export API_KEY

echo "CamAI serverini ishga tushurish..."
exec python app/main.py
