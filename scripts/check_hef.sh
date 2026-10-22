#!/usr/bin/env bash
# Raspberry Pi 5 + AI HAT+ (Hailo-8): sprawdzenie modelu koncowego FIELDEYE.
#
# Model: hailo/fieldeye_final_1280_h8.hef. Jesli go nie ma, skrypt wypakowuje go
# z model_fin.zip (YOLOv8n, 1 klasa, wejscie 1280, skompilowany pod HailoRT 4.23).
# Zbior testowy do oceny AP50 (bez niego ocena jest pomijana):
#   zbior_ewis1_podzial/dataset/data.yaml   dzien EWIS1 16.05.2022, odlozony od treningu
# Jednorazowo:
#   sudo apt install hailo-all python3-opencv python3-yaml python3-numpy
#
# Uruchomienie:
#   bash scripts/check_hef.sh                     model koncowy, test EWIS1
#   bash scripts/check_hef.sh model.hef data.yaml inny model albo zbior
set -euo pipefail
cd "$(dirname "$0")/.."

DOMYSLNY=hailo/fieldeye_final_1280_h8.hef
HEF=${1:-$DOMYSLNY}
DANE=${2:-zbior_ewis1_podzial/dataset/data.yaml}

if [ ! -f "$HEF" ] && [ "$HEF" = "$DOMYSLNY" ] && [ -f model_fin.zip ]; then
    echo "=== Wypakowuje $HEF z model_fin.zip"
    mkdir -p hailo
    unzip -j -o -q model_fin.zip "model_fin/$(basename "$HEF")" -d hailo/
fi
if [ ! -f "$HEF" ]; then
    echo "brak $HEF" >&2
    exit 1
fi
n=$(basename "$HEF" .hef)

echo "=== Akcelerator"
hailortcli fw-control identify

echo; echo "=== $HEF"
sha256sum "$HEF"
hailortcli parse-hef "$HEF" | head -25

echo; echo "=== Przepustowosc NPU (losowe dane, gorna granica kafelkow na sekunde)"
hailortcli run "$HEF"

if [ -f "$DANE" ]; then
    mkdir -p hailo
    echo; echo "=== Ocena na $DANE: bez TTA i z TTA"
    python3 -m models.cli ocen-detektor "$HEF" "$DANE" --prog 0.15 -o hailo/${n}_hef_det.json
    python3 -m models.cli ocen-detektor "$HEF" "$DANE" --prog 0.15 --tta -o hailo/${n}_hef_tta_det.json

    echo; echo "=== Porownanie z modelem .pt (ten sam kod oceny)"
    domyslne=0
    [ "$HEF" = "$DOMYSLNY" ] && [ -z "${2:-}" ] && domyslne=1
    python3 - "$n" "$domyslne" <<'PY'
import json
import sys
from pathlib import Path

n, domyslne = sys.argv[1], sys.argv[2] == "1"

def czytaj(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None

# Wyniki .pt modelu koncowego na tescie EWIS1 (16.05.2022) z dokumentacji modelu
# w model_fin.zip, rozdzial 3. Plik z Maca (hailo/wyniki_mac/), jesli jest, ma pierwszenstwo.
WZOR_DOKUMENTACJA = {"": {"AP50": 0.812, "przy_progu": {"czulosc": 0.791}},
                     "_tta": {"AP50": 0.817, "przy_progu": {"czulosc": 0.784}}}

print(f"\n  {'wariant':<10}{'AP50 .pt':>10}{'AP50 .hef':>11}{'różnica':>9}{'czułość .pt':>13}{'czułość .hef':>14}  ocena")
for sufiks, wariant in (("", "bez TTA"), ("_tta", "z TTA")):
    hef = czytaj(f"hailo/{n}_hef{sufiks}_det.json")
    pt = czytaj(f"hailo/wyniki_mac/{n}_pt{sufiks}_det.json")
    if pt is None and domyslne:
        pt = WZOR_DOKUMENTACJA[sufiks]
    if not (pt and hef):
        print(f"  {wariant:<10}  brak wyników .pt do porównania - pominięty")
        continue
    roz = hef["AP50"] - pt["AP50"]
    if roz >= -0.03:
        ocena = "OK: strata typowa dla kwantyzacji"
    elif roz >= -0.10:
        ocena = "do sprawdzenia: większa strata niż zwykle"
    else:
        ocena = "PROBLEM: sprawdź NMS i zdjęcia kalibracyjne, rozważ pełną optymalizację"
    print(f"  {wariant:<10}{pt['AP50']:>10.3f}{hef['AP50']:>11.3f}{roz:>+9.3f}"
          f"{pt['przy_progu']['czulosc']:>13.3f}{hef['przy_progu']['czulosc']:>14.3f}  {ocena}")
PY
else
    echo; echo "=== brak $DANE - pomijam ocene AP50"
    echo "Zbior testowy EWIS1 buduje sie na komputerze (dokumentacja modelu w model_fin.zip, rozdzial 6.6)"
    echo "i kopiuje na Pi razem z katalogiem zbior_ewis1_podzial/."
fi

echo
echo "Misja z tym modelem (configs/mission_hailo.yaml):"
echo "  python3 -m mission.cli pelna zdjecia/ -o wynik_misji/ --config configs/mission_hailo.yaml"
