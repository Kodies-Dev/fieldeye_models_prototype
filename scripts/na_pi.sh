#!/usr/bin/env bash
# FIELDEYE na Raspberry Pi (Hailo-8) - uruchamiac w rozpakowanym folderze fieldeye_pi.
#
#   bash na_pi.sh sprawdz     czy jest Hailo, Python, OpenCV, numpy, model, dane
#   bash na_pi.sh test        132 kafelki EWIS1 przez model .hef (bez GPS)
#   bash na_pi.sh gps         symulowany lot z telemetria GPS -> mapa oprysku -> ocena namiarow
#   bash na_pi.sh na-zywo     dron co 0,7 s i detekcja na biezaco (czy Hailo nadaza)
#   bash na_pi.sh binarka     buduje dist/fieldeye-detect (PyInstaller, wariant hailo) i sprawdza go
#   bash na_pi.sh wszystko    sprawdz + test + gps + na-zywo + binarka
#
# Wyniki w /tmp/fieldeye_*, pelne logi w /tmp/fieldeye_*.log.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
HEF="$(ls "$DIR"/*.hef 2>/dev/null | head -1)"
KAF="$DIR/zbior_ewis1_podzial/dataset/images/test"
PY=python3

naglowek() { echo; echo "=== $1"; }
wymagaj_hef() {
  [ -n "$HEF" ] || { echo "brak pliku .hef w $DIR - skopiuj model: cp SCIEZKA/fieldeye_final_1280_h8.hef $DIR/"; exit 1; }
}

sprawdz() {
  naglowek "Sprawdzenie"
  local ok=1
  if command -v hailortcli > /dev/null; then
    hailortcli fw-control identify 2>&1 | grep -iE "board|device|firmware|hailo" | head -3 | sed 's/^/  /'
    echo "  HailoRT: $(hailortcli --version 2>&1 | head -1)   (HEF skompilowany pod 4.23)"
  else
    echo "  BRAK hailortcli - sudo apt install hailo-all"; ok=0
  fi
  "$PY" -c "import hailo_platform" 2> /dev/null && echo "  hailo_platform: OK" || { echo "  BRAK hailo_platform (pakiet hailo-all)"; ok=0; }
  "$PY" -c "import cv2; print('  OpenCV:', cv2.__version__)" 2> /dev/null || { echo "  BRAK OpenCV - sudo apt install python3-opencv"; ok=0; }
  "$PY" -c "import numpy; print('  numpy:', numpy.__version__)" 2> /dev/null || { echo "  BRAK numpy - sudo apt install python3-numpy"; ok=0; }
  [ -n "$HEF" ] && echo "  model: $(basename "$HEF")" || { echo "  BRAK modelu .hef w folderze"; ok=0; }
  [ -d "$KAF" ] && echo "  dane: $(ls "$KAF" | wc -l) kafelkow testowych" || { echo "  BRAK danych testowych"; ok=0; }
  "$PY" -c "import mission.detect_cli, mission.spray_map, mission.drone_sim" 2> /dev/null \
    && echo "  kod FIELDEYE: OK" || { echo "  BLAD importu kodu FIELDEYE"; ok=0; }
  [ $ok -eq 1 ] && echo "  WSZYSTKO GOTOWE" || { echo "  popraw braki powyzej"; return 1; }
}

test_bez_gps() {
  wymagaj_hef
  naglowek "Test: 132 kafelki przez $(basename "$HEF") (bez GPS)"
  rm -rf /tmp/fieldeye_wyniki
  "$PY" -m mission.detect_cli "$KAF" /tmp/fieldeye_wyniki "$HEF" --raz > /tmp/fieldeye_test.log 2>&1
  sed -n '2,4p' /tmp/fieldeye_test.log | cut -c1-110
  echo "  ..."
  tail -2 /tmp/fieldeye_test.log
  echo "  (na Macu z modelem .pt: 132 zdjec, 4313 chwastow - z .hef powinno wyjsc podobnie)"
}

test_gps() {
  wymagaj_hef
  naglowek "Test: symulowany lot z GPS"
  rm -rf /tmp/fieldeye_lot /tmp/fieldeye_lot_prawda.geojson /tmp/fieldeye_wyniki_gps
  "$PY" -m mission.drone_sim /tmp/fieldeye_lot --ewis1 "$KAF" --interwal 0 > /tmp/fieldeye_dron.log 2>&1
  head -2 /tmp/fieldeye_dron.log | cut -c1-110
  "$PY" -m mission.detect_cli /tmp/fieldeye_lot /tmp/fieldeye_wyniki_gps "$HEF" --raz > /tmp/fieldeye_gps.log 2>&1
  tail -2 /tmp/fieldeye_gps.log
  echo "  Ocena namiarow:"
  "$PY" -m mission.spray_map /tmp/fieldeye_wyniki_gps/oprysk.geojson /tmp/fieldeye_lot_prawda.geojson
  echo "  Mapa: /tmp/fieldeye_wyniki_gps/oprysk.geojson (np. przeciagnac na geojson.io)"
}

na_zywo() {
  wymagaj_hef
  naglowek "Test na zywo: zdjecie co 0,7 s, detekcja na biezaco"
  rm -rf /tmp/fieldeye_live /tmp/fieldeye_live_prawda.geojson /tmp/fieldeye_wyniki_live
  "$PY" -m mission.detect_cli /tmp/fieldeye_live /tmp/fieldeye_wyniki_live "$HEF" --timeout 8 > /tmp/fieldeye_live.log 2>&1 &
  local det=$!
  trap 'kill $det 2> /dev/null' EXIT
  sleep 3
  "$PY" -m mission.drone_sim /tmp/fieldeye_live --ewis1 "$KAF" --interwal 0.7 > /dev/null 2>&1
  wait $det
  trap - EXIT
  grep -E "\.jpg " /tmp/fieldeye_live.log | tail -3 | cut -c1-140
  tail -2 /tmp/fieldeye_live.log
  local maks
  maks="$(grep -oE 'w kolejce: [0-9]+' /tmp/fieldeye_live.log | awk '{print $3}' | sort -n | tail -1)"
  echo "  najdluzsza kolejka: ${maks:-0} zdjec  (0-2 = Hailo nadaza; rosnie = nie nadaza)"
}

binarka() {
  wymagaj_hef
  naglowek "Binarka (wariant hailo)"
  bash scripts/build_binary.sh hailo || { echo "  budowa nie wyszla - prosze odeslac koniec pliku: $DIR/build-hailo.log"; return 1; }
  rm -rf /tmp/fieldeye_wyniki_bin
  ./dist/fieldeye-detect "$KAF" /tmp/fieldeye_wyniki_bin "$HEF" --raz > /tmp/fieldeye_bin.log 2>&1 \
    && tail -1 /tmp/fieldeye_bin.log \
    || { echo "  binarka nie ruszyla:"; tail -15 /tmp/fieldeye_bin.log; return 1; }
  echo "  binarka: $DIR/dist/fieldeye-detect  ($(du -h dist/fieldeye-detect | cut -f1))"
  echo "  uzycie:  ./dist/fieldeye-detect ZDJECIA WYNIKI $(basename "$HEF")"
}

case "${1:-}" in
  sprawdz) sprawdz ;;
  test) test_bez_gps ;;
  gps) test_gps ;;
  na-zywo) na_zywo ;;
  binarka) binarka ;;
  wszystko) sprawdz && test_bez_gps && test_gps && na_zywo && binarka ;;
  *) sed -n '2,10p' "$0" ;;
esac
