#!/usr/bin/env bash
# Drugi test: odlozony dzien lotu z EWIS1 - maj, inne pole niz WeedsGalore, zwykla kamera RGB.
#
# Ocenia modele, ktore NIGDY nie widzialy EWIS1 (wg_sudars, sezon_nadprob i ich drugie ziarna).
# Modele uczone na calym EWIS1 (dane_ewis1, dane_ewis1_sorgo) sa wylaczone - widzialy ten test.
# Z TRENUJ=1 trenuje tez model z EWIS1 bez odlozonego dnia i ocenia go na obu testach.
#
#   bash scripts/ewis1_test.sh                                  # kilka minut
#   TEST_DNI="20220509" bash scripts/ewis1_test.sh              # wybrany dzien
#   TRENUJ=1 caffeinate -i bash scripts/ewis1_test.sh           # + trening, ok. 4 h
set -euo pipefail
cd "$(dirname "$0")/.."
E=${EWIS1_DIR:-"dane/EWIS1 Maize, Sorghum & Weed Dataset"}
[ -d "$E" ] || E=dane/ewis1
Z=zbior_ewis1_podzial
W=wyniki/ewis1_test
mkdir -p "$W"
krok() { echo; echo "=== $*"; }

krok "1. Dni lotu (kukurydza)"
WYBOR=$(python - "$E" <<'PY'
import collections, os, re, sys
n = collections.Counter(m.group(1) for p in os.listdir(os.path.join(sys.argv[1], "images"))
                        if "maize" in p.lower() and (m := re.search(r"_(20\d{6})_", p)))
for d, k in sorted(n.items()):
    print(f"  {d}: {k} zdjec", file=sys.stderr)
cel = sum(n.values()) / 3                 # okolo 1/3 zdjec do testu
print(min(n, key=lambda d: (abs(n[d] - cel), d)))
PY
)
TEST_DNI=${TEST_DNI:-$WYBOR}
echo "  dzien testowy: $TEST_DNI"

krok "2. Zbior z odlozonym dniem"
if [ -f $Z/test_dni.txt ] && [ "$(cat $Z/test_dni.txt)" = "$TEST_DNI" ]; then
    echo "  $Z juz jest dla $TEST_DNI - pomijam"
else
    rm -rf $Z
    python -m normalization.build --ewis1 "$E" -o $Z --tryb-klas chwasty --skala ewis1=0.47 \
           --proporcje 1 0 0 --ewis1-test-dni $TEST_DNI > /dev/null
    echo "$TEST_DNI" > $Z/test_dni.txt
fi
python - <<PY
import json
from pathlib import Path
i = json.loads(Path("$Z/build_info.json").read_text())
assert not i.get("nierozpoznane_etykiety"), i["nierozpoznane_etykiety"]
for c in ("train", "test"):
    print(f"  {c:<6} kafelkow {len(list(Path('$Z/dataset/images', c).glob('*.jpg'))):>4}")
PY

krok "3. Ocena modeli, ktore nie widzialy EWIS1"
MODELE="wg_sudars wg_sudars_s1 sezon_nadprob sezon_nadprob_s1"
if [ "${TRENUJ:-0}" = "1" ]; then
    krok "3a. Trening: WeedsGalore (nadprobkowanie) + Sudars + EWIS1 bez dnia $TEST_DNI"
    python -m models.cli nadprobkuj zbior_lotnicze/dataset --waga 2023-05-25=3 --waga 2023-05-30=2 \
           --dodaj-trening zbior_naziemne/dataset --dodaj-trening $Z/dataset \
           -o zbior_lotnicze/dataset/data_dane_ewis1_bt.yaml > /dev/null
    cat > plan_dane_ewis1_bt.json <<'PLAN'
{"model_bazowy": "yolov8n.pt", "batch": 16, "cierpliwosc": 30, "seed": 0,
 "etapy": [{"nazwa": "dane-ewis1-bt", "dane": "zbior_lotnicze/dataset/data_dane_ewis1_bt.yaml",
            "epoki": 100, "lr0": 0.01, "wejscie": 1280}]}
PLAN
    [ -f modele_dane_ewis1_bt/model.pt ] || \
        python -m models.cli trenuj --plan plan_dane_ewis1_bt.json -o modele_dane_ewis1_bt > log_dane_ewis1_bt.txt 2>&1
    python -m models.cli ocen-detektor modele_dane_ewis1_bt/model.pt zbior_lotnicze/dataset/data.yaml \
           --prog 0.15 --tta -o "$W/dane_ewis1_bt_wg_tta.json" > /dev/null
    MODELE="$MODELE dane_ewis1_bt"
fi
for M in $MODELE; do
    [ -f modele_$M/model.pt ] || continue
    for T in "" "--tta"; do
        S=$([ -n "$T" ] && echo "_tta" || echo "")
        [ -f "$W/${M}${S}.json" ] || python -m models.cli ocen-detektor modele_$M/model.pt $Z/dataset/data.yaml \
            --split test --prog 0.15 $T -o "$W/${M}${S}.json" > /dev/null
    done
    echo "  $M ocenione"
done

krok "4. Wynik na drugim tescie (EWIS1, dzien $TEST_DNI)"
python - <<'PY'
import json
from pathlib import Path
W = Path("wyniki/ewis1_test")
print(f"\n  {'model':<20}| {'bez TTA':^24} | {'z TTA':^24}")
print(f"  {'':<20}| {'AP50':>7}{'czułość':>9}{'prec.':>8} | {'AP50':>7}{'czułość':>9}{'prec.':>8}")
for m in ("wg_sudars", "wg_sudars_s1", "sezon_nadprob", "sezon_nadprob_s1", "dane_ewis1_bt"):
    a, b = W / f"{m}.json", W / f"{m}_tta.json"
    if not (a.exists() and b.exists()):
        continue
    a, b = json.loads(a.read_text()), json.loads(b.read_text())
    print(f"  {m:<20}| {a['AP50']:>7.3f}{a['przy_progu']['czulosc']:>9.3f}{a['przy_progu']['precyzja']:>8.3f} | "
          f"{b['AP50']:>7.3f}{b['przy_progu']['czulosc']:>9.3f}{b['przy_progu']['precyzja']:>8.3f}")
p = W / "dane_ewis1_bt_wg_tta.json"
if p.exists():
    w = json.loads(p.read_text())
    print(f"\n  dane_ewis1_bt na WeedsGalore (TTA): AP50 {w['AP50']:.3f}, czułość {w['przy_progu']['czulosc']:.3f}, "
          f"25.05 {w['wedlug_dni'].get('2023-05-25', {}).get('czulosc', float('nan')):.3f}")
PY