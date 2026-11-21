#!/usr/bin/env bash
# Binarka fieldeye-detect przez PyInstaller.
#
# BUDUJ NA TYM SYSTEMIE, NA KTORYM MA DZIALAC: binarka z Maca nie ruszy na Ubuntu,
# a z Ubuntu na PC (x86) nie ruszy na Raspberry Pi (ARM).
#
#   bash scripts/build_binary.sh onnx              # Ubuntu / Mac: ONNX Runtime, bez PyTorcha (domyslnie)
#   bash scripts/build_binary.sh hailo             # Raspberry Pi z zainstalowanym hailo-all: model .hef
#   bash scripts/build_binary.sh pt                # PyTorch + ultralytics: dziala wszedzie, ale 1-2 GB
#   bash scripts/build_binary.sh onnx --katalog    # katalog z programem i bibliotekami zamiast jednego pliku
#
# Wynik (domyslnie): JEDEN PLIK dist/fieldeye-detect z bibliotekami w srodku - do uruchomienia
# wystarczy on, model i zdjecia. Start trwa ok. 1-2 s dluzej (rozpakowanie bibliotek do
# katalogu tymczasowego). Z --katalog: dist/fieldeye-detect/fieldeye-detect + _internal/
# (szybszy start, przenosi sie caly katalog).
# Model nie jest zaszyty w binarce: podaje sie go przy uruchomieniu (sciezka z nazwa
# pliku) albo kladzie obok programu (fieldeye_final_1280_h8.hef / fieldeye_final.onnx).
#
# Uruchomienie: dist/fieldeye-detect/fieldeye-detect ZDJECIA WYNIKI MODEL
set -euo pipefail
WARIANT=${1:-onnx}
JEDEN="--onefile"
[ "${2:-}" = "--katalog" ] && JEDEN=""
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
VENV=".venv-build-$WARIANT"

echo "=== 1. Srodowisko $VENV"
case $WARIANT in
    hailo) python3 -m venv --system-site-packages "$VENV" ;;     # hailo_platform, numpy, opencv z systemu (apt)
    onnx|pt) python3 -m venv "$VENV" ;;
    *) echo "nieznany wariant: $WARIANT (onnx | hailo | pt)"; exit 2 ;;
esac
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip
case $WARIANT in
    onnx)  python -m pip install -q pyinstaller numpy opencv-python-headless onnxruntime ;;
    hailo) python -m pip install -q pyinstaller
           python -c "import hailo_platform, cv2, numpy" \
               || { echo "brak hailo_platform/opencv: sudo apt install hailo-all python3-opencv"; exit 1; } ;;
    pt)    python -m pip install -q pyinstaller -r requirements.txt ;;
esac

echo "=== 2. PyInstaller ($WARIANT, $([ -n "$JEDEN" ] && echo "jeden plik" || echo "katalog"))"
rm -rf dist/fieldeye-detect build/fieldeye-detect          # poprzednia binarka (plik albo katalog)
WYKLUCZ=(--exclude-module streamlit --exclude-module tkinter --exclude-module matplotlib
         --exclude-module pandas --exclude-module scipy --exclude-module pyproj --exclude-module IPython)
DODAJ=()
case $WARIANT in
    onnx)  WYKLUCZ+=(--exclude-module torch --exclude-module torchvision --exclude-module ultralytics)
           DODAJ=(--collect-binaries onnxruntime) ;;
    hailo) WYKLUCZ+=(--exclude-module torch --exclude-module torchvision --exclude-module ultralytics
                     --exclude-module onnxruntime)
           DODAJ=(--collect-all hailo_platform) ;;
    pt)    DODAJ=(--collect-data ultralytics) ;;
esac
pyinstaller --noconfirm --clean $JEDEN --name fieldeye-detect --paths "$REPO" \
    "${WYKLUCZ[@]}" "${DODAJ[@]}" scripts/fieldeye_detect_main.py > "build-$WARIANT.log" 2>&1 \
    || { echo "PyInstaller nie przeszedl - szczegoly: build-$WARIANT.log"; exit 1; }

echo "=== 3. Wynik"
if [ -n "$JEDEN" ]; then PROG="dist/fieldeye-detect"; else PROG="dist/fieldeye-detect/fieldeye-detect"; fi
"$PROG" --help > /dev/null && echo "  dziala: $PROG"
du -sh "dist/fieldeye-detect" | sed 's/^/  rozmiar: /'
echo "  uruchomienie: $PROG ZDJECIA WYNIKI MODEL   (MODEL: sciezka z nazwa pliku .onnx / .hef / .pt)"
