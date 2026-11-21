#!/usr/bin/env bash
# Pakuje wszystko, czego potrzeba na Raspberry Pi, do ~/Downloads/fieldeye_pi.zip:
# kod (bez treningu i danych treningowych), 132 kafelki testowe EWIS1 z etykietami,
# model .hef (jesli go znajdzie) i skrypt na_pi.sh do testow i budowy binarki na Pi.
#
#   bash scripts/spakuj_na_pi.sh
#   bash scripts/spakuj_na_pi.sh --hef SCIEZKA/fieldeye_final_1280_h8.hef
#   bash scripts/spakuj_na_pi.sh --bez-testow
set -euo pipefail
HEF=""; TESTY=1
while [ $# -gt 0 ]; do
  case "$1" in
    --hef) HEF="$2"; shift ;;
    --bez-testow) TESTY=0 ;;
    *) echo "nieznana opcja: $1"; exit 2 ;;
  esac
  shift
done
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
[ -f app.py ] || { echo "uruchom z projektu FIELDEYE"; exit 1; }
CEL="$HOME/Downloads/fieldeye_pi.zip"
TMP="$(mktemp -d)"
PACZKA="$TMP/fieldeye_pi"
trap 'rm -rf "$TMP"' EXIT
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python

echo "=== 1. Testy przed spakowaniem"
if [ $TESTY -eq 1 ]; then
  "$PY" -m pytest tests/test_detect_cli.py tests/test_spray_map.py tests/test_binary.py -q 2>&1 | tail -1
  "$PY" -m pytest tests/test_detect_cli.py tests/test_spray_map.py tests/test_binary.py -q > /dev/null 2>&1 \
    || { echo "testy nie przeszly - nie pakuje (pomin: --bez-testow)"; exit 1; }
else
  echo "  pominiete"
fi

echo "=== 2. Kod"
mkdir -p "$PACZKA"
for k in mission stitchlive models scripts configs; do
  [ -d "$k" ] && cp -R "$k" "$PACZKA/"
done
find "$PACZKA" \( -name "__pycache__" -o -name "weights" \) -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$PACZKA" \( -name "*.pyc" -o -name ".DS_Store" \) -type f -delete 2>/dev/null || true
for f in pyproject.toml README.md; do [ -f "$f" ] && cp "$f" "$PACZKA/"; done
echo "  $(find "$PACZKA" -name '*.py' | wc -l | tr -d ' ') plikow .py, $(find "$PACZKA/scripts" -name '*.sh' | wc -l | tr -d ' ') skryptow .sh"

echo "=== 3. Dane testowe"
KAF="zbior_ewis1_podzial/dataset"
if [ -d "$KAF/images/test" ]; then
  mkdir -p "$PACZKA/$KAF/images" "$PACZKA/$KAF/labels"
  cp -R "$KAF/images/test" "$PACZKA/$KAF/images/"
  [ -d "$KAF/labels/test" ] && cp -R "$KAF/labels/test" "$PACZKA/$KAF/labels/"
  echo "  $(ls "$PACZKA/$KAF/images/test" | wc -l | tr -d ' ') kafelkow EWIS1 z etykietami"
else
  echo "  UWAGA: brak $KAF/images/test - paczka bez danych testowych"
fi

echo "=== 4. Model .hef"
if [ -z "$HEF" ]; then
  HEF="$(find . ~/modele ~/Desktop ~/Downloads -maxdepth 3 -name '*.hef' -not -path '*/.venv*' 2>/dev/null | head -1 || true)"
fi
if [ -n "$HEF" ] && [ -f "$HEF" ]; then
  cp "$HEF" "$PACZKA/"
  echo "  $(basename "$HEF")  (z: $HEF)"
elif [ -f model_fin.zip ] && unzip -l model_fin.zip | grep -qi '\.hef'; then
  unzip -jqo model_fin.zip '*.hef' -d "$PACZKA/"
  echo "  $(ls "$PACZKA"/*.hef | xargs -n1 basename)  (z: model_fin.zip)"
else
  echo "  UWAGA: nie znaleziono .hef - kolega uzyje swojego (wtedy: cp SCIEZKA/*.hef do folderu)"
fi

echo "=== 5. Skrypt dla Pi i instrukcja"
cp scripts/na_pi.sh "$PACZKA/na_pi.sh"
chmod +x "$PACZKA/na_pi.sh" "$PACZKA"/scripts/*.sh 2>/dev/null || true
cat > "$PACZKA/README_PI.txt" <<'EOF'
FIELDEYE na Raspberry Pi 5 + AI HAT+ (Hailo-8)

Wymagania (jednorazowo):
  sudo apt update
  sudo apt install -y hailo-all python3-venv python3-pip python3-opencv python3-numpy unzip
  (model .hef jest skompilowany pod HailoRT 4.23 - sprawdz: hailortcli --version)

Rozpakowanie:
  mkdir -p ~/fieldeye_test && cd ~/fieldeye_test
  unzip -oq ~/Downloads/fieldeye_pi.zip
  cd fieldeye_pi

Kolejno:
  bash na_pi.sh sprawdz     czy jest Hailo, Python, OpenCV, numpy, model, dane
  bash na_pi.sh test        132 kafelki przez model .hef (bez GPS); porownanie z Makiem: 4313 chwastow z .pt
  bash na_pi.sh gps         symulowany lot z telemetria GPS + mapa oprysku + ocena namiarow
  bash na_pi.sh na-zywo     dron co 0,7 s i detekcja na biezaco - czy Hailo nadaza (kolejka nie rosnie)
  bash na_pi.sh binarka     buduje dist/fieldeye-detect (jeden plik) i sprawdza go

Wyniki: /tmp/fieldeye_*; logi z kazdego kroku: /tmp/fieldeye_*.log - prosze odeslac podsumowania.
EOF
ls -la "$PACZKA" | tail -n +4 | awk '{print "  " $NF}'

echo "=== 6. ZIP"
rm -f "$CEL"
( cd "$TMP" && zip -rq "$CEL" fieldeye_pi -x "*.DS_Store" )
echo "  $CEL  ($(du -h "$CEL" | cut -f1))"
echo
echo "Wyslij koledze $CEL. Na Pi: unzip -oq fieldeye_pi.zip && cd fieldeye_pi && bash na_pi.sh sprawdz"
