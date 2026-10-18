#!/usr/bin/env bash
# Raspberry Pi 5 + AI HAT+ (Hailo-8): sprawdzenie skompilowanych modeli FIELDEYE.
#
# Potrzebne w katalogu stitch-live:
#   hailo/wg_sudars_640.hef, hailo/wg_sudars_1280.hef   (z kompilacji, scripts/HAILO.md)
#   hailo/test/, hailo/wyniki_mac/                        (z scripts/finish_models.sh na Macu)
# Jednorazowo:
#   sudo apt install hailo-all python3-opencv python3-yaml python3-numpy
#
# Uruchomienie:
#   bash scripts/check_hef.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Akcelerator"
hailortcli fw-control identify

for H in hailo/wg_sudars_640.hef hailo/wg_sudars_1280.hef; do
    if [ ! -f "$H" ]; then
        echo; echo "=== brak $H - pomijam"
        continue
    fi
    n=$(basename "$H" .hef)
    echo; echo "=== $H"
    hailortcli parse-hef "$H" | head -25
    python3 -m models.cli ocen-detektor "$H" hailo/test/data.yaml --prog 0.15 -o hailo/${n}_hef_det.json
done

echo; echo "=== Porownanie z modelem .pt (ten sam kod oceny)"
python3 - <<'PY'
import json
from pathlib import Path

def czytaj(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None

pary = (("wg_sudars_640", "hailo/wyniki_mac/wg_sudars_640_pt_det.json"),
        ("wg_sudars_1280", "hailo/wyniki_mac/wg_sudars_pt_det.json"))
print(f"\n  {'model':<16}{'AP50 .pt':>10}{'AP50 .hef':>11}{'różnica':>9}{'czułość .pt':>13}{'czułość .hef':>14}  ocena")
for n, mac in pary:
    pt, hef = czytaj(mac), czytaj(f"hailo/{n}_hef_det.json")
    if not (pt and hef):
        print(f"  {n:<16}  brak wyników - pominięty")
        continue
    roz = hef["AP50"] - pt["AP50"]
    if roz >= -0.03:
        ocena = "OK: strata typowa dla kwantyzacji"
    elif roz >= -0.10:
        ocena = "do sprawdzenia: większa strata niż zwykle"
    else:
        ocena = "PROBLEM: sprawdź --classes 1, NMS w .alls, zdjęcia kalibracyjne"
    print(f"  {n:<16}{pt['AP50']:>10.3f}{hef['AP50']:>11.3f}{roz:>+9.3f}"
          f"{pt['przy_progu']['czulosc']:>13.3f}{hef['przy_progu']['czulosc']:>14.3f}  {ocena}")
PY

echo
echo "Model do misji wpisz w configs/mission_hailo.yaml i uruchom:"
echo "  python3 -m mission.cli pelna zdjecia/ -o wynik_misji/ --config configs/mission_hailo.yaml"
