#!/usr/bin/env bash
# check_hef.sh - sprawdzenie modelu koncowego FIELDEYE na Raspberry Pi 5 z AI HAT+ (Hailo-8).
#
# Po co:
#   Po kompilacji na WSL (scripts/compile_hef.py) trzeba sprawdzic na prawdziwym Pi, czy plik .hef
#   dziala z HailoRT, jak szybko liczy i ile dokladnosci kosztowala kwantyzacja do INT8.
#   Dokladnosc liczona jest TYM SAMYM kodem co na Macu (models.cli ocen-detektor),
#   wiec liczby .hef i .pt mozna porownac wprost.
#
# Jak dziala (po kolei):
#   1. akcelerator    - hailortcli fw-control identify (czy Hailo-8 i firmware odpowiadaja)
#   2. plik HEF       - hailortcli parse-hef (wejscie 1280x1280x3, wyjscie z NMS; zgodnosc z HailoRT)
#   3. szybkosc       - hailortcli run (kafelki 1280x1280 na sekunde)
#   4. dokladnosc     - ocen-detektor na tescie EWIS1 (dzien 16.05.2022, 132 kafelki), prog 0.15,
#                       bez TTA i z TTA -> wyniki_hef/hef.json, hef_tta.json
#   5. porownanie     - AP50, czulosc i precyzja .hef kontra .pt z Maca:
#                       strata do 0,02 = dobrze, do 0,05 = akceptowalnie, wiecej = sprawdzic kompilacje
#
# Argumenty:
#   $1 - plik .hef, domyslnie fieldeye_final_1280_h8.hef (w katalogu repozytorium, nie w gicie)
#   $2 - data.yaml testu, domyslnie test_ewis1/data.yaml (paczka testowa)
#        albo zbior_ewis1_podzial/dataset/data.yaml (z final_model.sh)
#
# Zwraca (zapisuje):
#   wyniki_hef/hef.json, wyniki_hef/hef_tta.json + tabela porownania w terminalu
#
# Uzywa:
#   hailortcli (pakiet hailo-all), python3 -m models.cli ocen-detektor -> mission/detector.py HailoDetector
#   Opis: scripts/HAILO.md, krok 3.
#
#   bash scripts/check_hef.sh                                    # domyslne pliki powyzej
#   bash scripts/check_hef.sh moj_model.hef inny_test/data.yaml  # wlasne sciezki
set -uo pipefail                     # bez -e: bledy obslugiwane recznie, z czytelnym komunikatem
cd "$(dirname "$0")/.."              # glowny folder repo
HEF=${1:-fieldeye_final_1280_h8.hef}
TEST=${2:-test_ewis1/data.yaml}
W=wyniki_hef                         # tu ida wyniki oceny
mkdir -p "$W"
krok() { echo; echo "=== $*"; }    # naglowek kroku w terminalu

krok "1. Akcelerator"
hailortcli fw-control identify || { echo "Brak akceleratora Hailo albo HailoRT - sprawdz AI HAT+ i pakiet hailo-all."; exit 1; }

krok "2. Plik HEF: $HEF"
[ -f "$HEF" ] || { echo "Brak pliku $HEF - skopiuj go do katalogu repozytorium."; exit 1; }
hailortcli parse-hef "$HEF" | head -30 || { echo "parse-hef nie przeszedl: plik niezgodny z HailoRT na tym Pi."; exit 1; }

krok "3. Szybkosc (kafelki 1280x1280 na sekunde)"
hailortcli run "$HEF" || echo "  hailortcli run nie przeszedl"    # tylko ostrzezenie - idziemy dalej

krok "4. Dokladnosc na tescie: $TEST"
if [ ! -f "$TEST" ]; then
    echo "  Brak $TEST - pomijam. Rozpakuj test_ewis1.zip do katalogu repozytorium."
    exit 0
fi
# ten sam prog i test co dla .pt w final_model.sh - wyniki da sie porownac
python3 -m models.cli ocen-detektor "$HEF" "$TEST" --split test --prog 0.15 -o "$W/hef.json" \
    || { echo "  ocena bez TTA nie przeszla"; exit 1; }
python3 -m models.cli ocen-detektor "$HEF" "$TEST" --split test --prog 0.15 --tta -o "$W/hef_tta.json" \
    || { echo "  ocena z TTA nie przeszla"; exit 1; }

krok "5. Porownanie z modelem .pt (Mac, ten sam test)"
# tabela: AP50 .pt / .hef / roznica, czulosc .pt / .hef, precyzja .hef i slowna ocena straty
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
