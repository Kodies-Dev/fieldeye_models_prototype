#!/usr/bin/env bash
# Sklada samowystarczalny folder FIELDEYE: binarka + modele + przyklady + uruchom.sh + README.
# Do uruchomienia z tego folderu nie jest potrzebny projekt, Python ani biblioteki.
#
#   bash scripts/przygotuj_folder.sh "$HOME/Documents/test binarki fieldeye"
#   bash scripts/przygotuj_folder.sh CEL --od-zera         # binarka budowana calkiem od nowa
#   bash scripts/przygotuj_folder.sh CEL --bez-budowania   # uzyj binarki z dist/
#   bash scripts/przygotuj_folder.sh CEL --onnx PLIK.onnx  # gotowy model ONNX zamiast eksportu z .pt
#
# Binarka (wariant onnx) dziala na systemie, na ktorym ja zbudowano (Mac -> Mac, Ubuntu -> Ubuntu).
set -euo pipefail
[ $# -ge 1 ] || { echo "Uzycie: bash scripts/przygotuj_folder.sh FOLDER_DOCELOWY [--od-zera] [--bez-budowania] [--onnx PLIK]"; exit 2; }
CEL="$1"; shift
OD_ZERA=0; BUDUJ=1; ONNX=""
while [ $# -gt 0 ]; do
  case "$1" in
    --od-zera) OD_ZERA=1 ;;
    --bez-budowania) BUDUJ=0 ;;
    --onnx) ONNX="$2"; shift ;;
    *) echo "nieznana opcja: $1"; exit 2 ;;
  esac
  shift
done
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
[ -f app.py ] || { echo "uruchom z projektu FIELDEYE"; exit 1; }
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
KAFELKI="zbior_ewis1_podzial/dataset/images/test"

echo "=== 1. Binarka"
if [ $OD_ZERA -eq 1 ]; then
  rm -rf dist build .venv-build-onnx fieldeye-detect.spec build-onnx.log
fi
if [ $BUDUJ -eq 1 ]; then
  bash scripts/build_binary.sh onnx
fi
[ -f dist/fieldeye-detect ] || { echo "brak dist/fieldeye-detect - zbuduj binarke (bez --bez-budowania)"; exit 1; }

echo "=== 2. Folder: $CEL"
mkdir -p "$CEL"
# czyscimy tylko to, co sami tworzymy - nic innego w folderze nie ruszamy
rm -rf "$CEL/fieldeye-detect" "$CEL/modele" "$CEL/przyklad" "$CEL/wyniki" "$CEL/wyniki_gps" \
       "$CEL/wyniki_na_zywo" "$CEL/lot_na_zywo" "$CEL/lot_na_zywo_prawda.geojson" \
       "$CEL/uruchom.sh" "$CEL/README.txt" "$CEL/WERSJA.txt"
mkdir -p "$CEL/modele" "$CEL/przyklad/ewis1/images" "$CEL/przyklad/ewis1/labels"
cp dist/fieldeye-detect "$CEL/"
echo "  binarka"

if [ -n "$ONNX" ]; then
  cp "$ONNX" "$CEL/modele/fieldeye_final.onnx"
else
  "$PY" -m models.cli onnx models/weights/fieldeye_final.pt -o "$CEL/modele/fieldeye_final.onnx" \
      --wejscie 1280 --opset 11 > "$CEL/modele/.eksport.log" 2>&1 \
    || { echo "eksport ONNX nie wyszedl - szczegoly: $CEL/modele/.eksport.log (albo podaj --onnx PLIK)"; exit 1; }
  rm -f "$CEL/modele/.eksport.log"
fi
cp models/weights/fieldeye_final.pt "$CEL/modele/"
HEF="$(ls hailo/final/*.hef fieldeye_final_1280_h8.hef 2>/dev/null | head -1 || true)"
[ -n "$HEF" ] && cp "$HEF" "$CEL/modele/"
echo "  modele: $(ls "$CEL/modele" | tr '\n' ' ')"

if [ -d "$KAFELKI" ]; then
  PIERWSZY="$(ls "$KAFELKI" | grep -E '_k[0-9]+_[0-9]+\.jpg$' | head -1)"
  BAZA="${PIERWSZY%_k*}"
  for f in "$KAFELKI/${BAZA}"_k*.jpg; do
    cp "$f" "$CEL/przyklad/ewis1/images/"
    et="${KAFELKI/images/labels}/$(basename "${f%.jpg}").txt"
    [ -f "$et" ] && cp "$et" "$CEL/przyklad/ewis1/labels/"
  done
  echo "  przyklad: $(ls "$CEL/przyklad/ewis1/images" | wc -l | tr -d ' ') kafelkow jednego zdjecia EWIS1 z etykietami"
  "$CEL/fieldeye-detect" dron "$CEL/przyklad/lot_gps" --ewis1 "$CEL/przyklad/ewis1/images" --interwal 0 \
      --prawda "$CEL/przyklad/lot_gps_prawda.geojson" > /dev/null
  echo "  przyklad: lot z telemetria GPS ($(ls "$CEL/przyklad/lot_gps" | grep -c '\.jpg$') zdjec) + prawdziwe polozenia chwastow"
else
  echo "  UWAGA: brak $KAFELKI - folder bez przykladow (rozpakuj fieldeye_dane.zip w projekcie)"
fi

cat > "$CEL/uruchom.sh" <<'EOF'
#!/usr/bin/env bash
# FIELDEYE - uruchamianie z tego folderu (bez projektu i Pythona).
#   bash uruchom.sh                  przyklad: 12 kafelkow -> wyniki/ (zdjecia z ramkami)
#   bash uruchom.sh gps              przyklad: lot z telemetria GPS -> wyniki_gps/ (mapa oprysku) + ocena namiarow
#   bash uruchom.sh na-zywo          symulowany dron w locie, detekcja na biezaco -> wyniki_na_zywo/
#   bash uruchom.sh ZDJECIA WYNIKI   wlasne zdjecia: obserwuje ZDJECIA, Ctrl+C konczy (dalsze opcje: --help)
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$DIR/fieldeye-detect"
MODEL="$DIR/modele/fieldeye_final.onnx"
case "${1:-}" in
  "")
    "$BIN" "$DIR/przyklad/ewis1/images" "$DIR/wyniki" "$MODEL" --raz ;;
  gps)
    rm -rf "$DIR/wyniki_gps"
    "$BIN" "$DIR/przyklad/lot_gps" "$DIR/wyniki_gps" "$MODEL" --raz
    echo; echo "Ocena namiarow oprysku (punkt blizej niz 10 cm od prawdziwego chwastu = trafiony):"
    "$BIN" porownaj "$DIR/wyniki_gps/oprysk.geojson" "$DIR/przyklad/lot_gps_prawda.geojson" ;;
  na-zywo)
    rm -rf "$DIR/lot_na_zywo" "$DIR/wyniki_na_zywo" "$DIR/lot_na_zywo_prawda.geojson"
    "$BIN" "$DIR/lot_na_zywo" "$DIR/wyniki_na_zywo" "$MODEL" --timeout 6 &
    DET=$!
    trap 'kill $DET 2>/dev/null' EXIT
    sleep 2
    "$BIN" dron "$DIR/lot_na_zywo" --ewis1 "$DIR/przyklad/ewis1/images" --interwal 0.7 > /dev/null
    wait $DET || true
    trap - EXIT
    echo; echo "Ocena namiarow oprysku:"
    "$BIN" porownaj "$DIR/wyniki_na_zywo/oprysk.geojson" "$DIR/lot_na_zywo_prawda.geojson" ;;
  -h|--help)
    sed -n '2,6p' "$0" ;;
  *)
    [ $# -ge 2 ] || { sed -n '2,6p' "$0"; exit 2; }
    "$BIN" "$1" "$2" "$MODEL" "${@:3}" ;;
esac
EOF
chmod +x "$CEL/uruchom.sh" "$CEL/fieldeye-detect"

cat > "$CEL/README.txt" <<'EOF'
FIELDEYE - wykrywanie chwastow na zdjeciach z drona i mapa oprysku
==================================================================

Zawartosc
  fieldeye-detect     program (jeden plik, biblioteki w srodku; bez Pythona)
  modele/             fieldeye_final.onnx - model dla tego programu
                      fieldeye_final.pt   - ten sam model w PyTorch (do programu w wariancie pt)
                      *.hef (jesli jest)  - ten sam model dla Raspberry Pi z Hailo-8 (wariant hailo)
  przyklad/ewis1/     12 kafelkow jednego zdjecia z drona z prawdziwymi chwastami (etykiety)
  przyklad/lot_gps/   symulowany lot: zdjecia + telemetria GPS (IMG_0001.jpg + IMG_0001.geojson)
  uruchom.sh          gotowe uruchomienia (ponizej)

Szybki start
  bash uruchom.sh             12 kafelkow -> wyniki/ (zdjecia z zaznaczonymi chwastami)
  bash uruchom.sh gps         lot z GPS -> wyniki_gps/ (mapa oprysku) + ocena namiarow
  bash uruchom.sh na-zywo     symulowany dron w locie, detekcja na biezaco
  bash uruchom.sh ZDJECIA WYNIKI   wlasne zdjecia (obserwuje folder, Ctrl+C konczy)

Bezposrednio
  ./fieldeye-detect ZDJECIA WYNIKI modele/fieldeye_final.onnx [--raz] [--tta] [--json] [--help]
  ./fieldeye-detect dron FOLDER --ewis1 przyklad/ewis1/images [--lat .. --lon ..] [--szum-gps-m 1.5]
  ./fieldeye-detect porownaj WYNIKI/oprysk.geojson PRAWDA.geojson

Program obserwuje folder ZDJECIA i kazde nowe zdjecie (w kolejnosci zapisu) przepuszcza
przez model; zdjecie z zaznaczonymi chwastami zapisuje do WYNIKI (folder powstaje sam).

Telemetria GPS: obok zdjecia plik o tej samej nazwie z rozszerzeniem .geojson
(20261109T142859.998Z.jpg -> 20261109T142859.998Z.geojson), w formacie drona:
  {"type": "FeatureCollection", "features": [{"type": "Feature",
    "geometry": {"type": "Point", "coordinates": [DLUGOSC, SZEROKOSC, WYSOKOSC_NPM]},
    "properties": {"image": "...jpg", "time_utc": "...", "fix": true, "fix_quality": 2,
      "heading_deg": 90.0, "roll_deg": 0.0, "pitch_deg": 0.0,
      "survey": {"agl_m": 2.0, "gsd_mm_per_px": 0.623}}}]}
  (heading_deg - kurs drona od polnocy; agl_m - wysokosc nad polem; opis: mission/spray_map.py)
w WYNIKI powstaje mapa oprysku, odswiezana po kazdym zdjeciu:
  oprysk.geojson / oprysk.csv   punkty do oprysku (ten sam chwast z kilku zdjec = jeden punkt)
  chwasty.geojson               wszystkie wykrycia;  trasa.geojson - trasa drona
Mape mozna obejrzec, przeciagajac pliki .geojson na strone geojson.io albo w QGIS.
Dokladnosc namiarow zalezy od GPS: bez RTK ok. 1-2 m, z RTK kilka cm.

Uwagi
  - Program dziala na systemie, na ktorym go zbudowano (Mac z procesorem Apple / Ubuntu / Raspberry Pi).
  - Gdy macOS blokuje program pobrany z internetu: xattr -dr com.apple.quarantine fieldeye-detect
  - Nie uruchamiac dwuklikiem - z Terminala, z argumentami.
EOF

{
  echo "FIELDEYE - wersja folderu"
  echo "Zbudowano:  $(date '+%Y-%m-%d %H:%M')  na $(uname -sm)"
  echo "Kod:        $(git rev-parse --short HEAD 2>/dev/null || echo '?')$( [ -n "$(git status --porcelain 2>/dev/null)" ] && echo ' + zmiany lokalne')"
  echo "Model:      fieldeye_final (dane_ewis1_bt), YOLOv8n, wejscie 1280, 1 klasa: chwast"
  echo "Wyniki:     zwykla kamera RGB (EWIS1, niewidziany dzien): AP50 0,817, czulosc 78 %, precyzja 84 % (z TTA)"
  echo "            kamera multispektralna (WeedsGalore, caly sezon): AP50 0,714"
} > "$CEL/WERSJA.txt"

echo "=== 3. Sprawdzenie z folderu docelowego"
if [ -d "$CEL/przyklad/lot_gps" ]; then
  ( cd "$CEL" && bash uruchom.sh gps | tail -4 )
fi
echo
echo "Gotowe: $CEL"
ls -la "$CEL" | tail -n +4 | awk '{print "  " $NF}'
echo "Dalej: cd \"$CEL\" && bash uruchom.sh"
