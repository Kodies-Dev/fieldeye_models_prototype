#!/usr/bin/env bash
# fieldeye-detect: zdjecia z katalogu -> model chwastow -> wynik w terminalu (Linux, macOS, Raspberry Pi).
#   bash scripts/fieldeye-detect.sh ZDJECIA WYNIKI MODEL         # MODEL: sciezka z nazwa pliku (.pt/.onnx/.hef)
#   bash scripts/fieldeye-detect.sh ZDJECIA WYNIKI MODEL --raz   # przetwarza i konczy
#   bash scripts/fieldeye-detect.sh ZDJECIA WYNIKI               # model koncowy z projektu
# Wszystkie opcje: bash scripts/fieldeye-detect.sh --help
set -e
REPO="$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")/.." && pwd)"
[ -f "$REPO/.venv/bin/activate" ] && source "$REPO/.venv/bin/activate"
# bez "cd": sciezki ZDJECIA i WYNIKI licza sie od miejsca, z ktorego uruchamiasz skrypt
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m mission.detect_cli "$@"
