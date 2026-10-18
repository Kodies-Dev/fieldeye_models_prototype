#!/usr/bin/env bash
# Przygotowanie projektu na Macu z Apple Silicon - jednym poleceniem.
#
#   cd stitch-live
#   bash scripts/setup_mac.sh
#
# Co robi: sprawdza Pythona, tworzy .venv, instaluje biblioteki, pokazuje,
# czy GPU (MPS) jest widoczne, uruchamia testy i - jesli jest folder dane/ -
# buduje oba zbiory treningowe. Mozna uruchamiac wielokrotnie.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/5 Python"
PY=""
for kand in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$kand" >/dev/null 2>&1; then
    if "$kand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
      PY="$kand"; break
    fi
  fi
done
if [ -z "$PY" ]; then
  echo "Brak Pythona 3.10+ (systemowy python3 na macOS bywa za stary dla PyTorcha)."
  echo "Zainstaluj np.:  brew install python@3.12   albo instalator z python.org"
  exit 1
fi
echo "   uzywam: $PY ($($PY --version))"

echo "== 2/5 Srodowisko .venv"
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q

echo "== 3/5 Urzadzenie"
python -m models.cli srodowisko

echo "== 4/5 Testy"
python -m pytest tests -q

echo "== 5/5 Zbiory treningowe"
if [ -d dane/sudars/dataset ]; then
  python -m normalization.build --sudars dane/sudars/dataset -o zbior_naziemne --tryb-klas chwasty
else
  echo "   pomijam Sudars - brak dane/sudars/dataset"
fi
if [ -d dane/weedsgalore-dataset ]; then
  python -m normalization.build --weedsgalore dane/weedsgalore-dataset -o zbior_lotnicze --tryb-klas chwasty
else
  echo "   pomijam WeedsGalore - brak dane/weedsgalore-dataset"
fi

echo
echo "Gotowe. W kazdym nowym terminalu aktywuj srodowisko:"
echo "   source .venv/bin/activate"
