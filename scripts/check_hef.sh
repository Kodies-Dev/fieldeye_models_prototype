#!/usr/bin/env bash
# Sprawdzenie modelu koncowego FIELDEYE na Raspberry Pi 5 z AI HAT+ (Hailo-8).
#
#   bash scripts/check_hef.sh                                    # domyslne pliki ponizej
#   bash scripts/check_hef.sh moj_model.hef inny_test/data.yaml  # wlasne sciezki
#
# Domyslnie (z katalogu repozytorium):
#   fieldeye_final_1280_h8.hef     model koncowy (przekazywany osobno, nie w gicie)
#   test_ewis1/data.yaml           paczka testowa: odlozony dzien EWIS1 (132 kafelki)
#
# Kroki: akcelerator i firmware, opis pliku HEF, szybkosc, dokladnosc na tescie
# (bez TTA i z TTA) i porownanie z modelem .pt z Maca.
set -uo pipefail
cd "$(dirname "$0")/.."
HEF=${1:-fieldeye_final_1280_h8.hef}
TEST=${2:-test_ewis1/data.yaml}
W=wyniki_hef
mkdir -p "$W"
krok() { echo; echo "=== $*"; }

krok "1. Akcelerator"
hailortcli fw-control identify || { echo "Brak akceleratora Hailo albo HailoRT - sprawdz AI HAT+ i pakiet hailo-all."; exit 1; }

krok "2. Plik HEF: $HEF"
[ -f "$HEF" ] || { echo "Brak pliku $HEF - skopiuj go do katalogu repozytorium."; exit 1; }
hailortcli parse-hef "$HEF" | head -30 || { echo "parse-hef nie przeszedl: plik niezgodny z HailoRT na tym Pi."; exit 1; }

krok "3. Szybkosc (kafelki 1280x1280 na sekunde)"
hailortcli run "$HEF" || echo "  hailortcli run nie przeszedl"

krok "4. Dokladnosc na tescie: $TEST"
if [ ! -f "$TEST" ]; then
    echo "  Brak $TEST - pomijam. Rozpakuj test_ewis1.zip do katalogu repozytorium."
    exit 0
fi
python3 -m models.cli ocen-detektor "$HEF" "$TEST" --split test --prog 0.15 -o "$W/hef.json" \
    || { echo "  ocena bez TTA nie przeszla"; exit 1; }
python3 -m models.cli ocen-detektor "$HEF" "$TEST" --split test --prog 0.15 --tta -o "$W/hef_tta.json" \
    || { echo "  ocena z TTA nie przeszla"; exit 1; }

krok "5. Porownanie z modelem .pt (Mac, ten sam test)"
python3 - <<'PY'
import json
from pathlib import Path
# Wyniki modelu .pt dane_ewis1_bt na odlozonym dniu EWIS1 (16.05.2022), prog 0,15.
PT = {"bez TTA": (0.812, 0.791, 0.811), "z TTA": (0.817, 0.784, 0.843)}
PLIKI = {"bez TTA": "wyniki_hef/hef.json", "z TTA": "wyniki_hef/hef_tta.json"}
print(f"\n  {'':<9}{'AP50 .pt':>10}{'AP50 .hef':>11}{'różnica':>9}{'czułość .pt':>13}{'czułość .hef':>14}{'prec. .hef':>12}  ocena")
for k, (ap, cz, pr) in PT.items():
    h = json.loads(Path(PLIKI[k]).read_text())
    roz = h["AP50"] - ap
    ocena = ("dobrze" if roz >= -0.02 else
             "akceptowalnie (kwantyzacja na poziomie 0)" if roz >= -0.05 else
             "za duza strata - sprawdzic kompilacje i kalibracje")
    print(f"  {k:<9}{ap:>10.3f}{h['AP50']:>11.3f}{roz:>+9.3f}{cz:>13.3f}"
          f"{h['przy_progu']['czulosc']:>14.3f}{h['przy_progu']['precyzja']:>12.3f}  {ocena}")
print("\n  NMS na ukladzie odrzuca wykrycia ponizej 0,05, wiec AP50 HEF-a jest liczone z nieco krotszej")
print("  listy niz dla .pt - roznica rzedu 0,01 moze wynikac tylko z tego.")
PY