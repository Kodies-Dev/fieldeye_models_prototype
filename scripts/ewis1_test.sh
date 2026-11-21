#!/usr/bin/env bash
# ewis1_test.sh - drugi, niezalezny test modeli na EWIS1 + (opcjonalnie) trening modelu koncowego.
#
# Po co:
#   WeedsGalore to jedno pole i jedna kamera. EWIS1 to inne pole, inny dron (DJI Mavic 2 Pro)
#   i zwykla kamera RGB podobna do kamery ktora my posiadamy w hardware. Caly jeden dzien
#   lotu EWIS1 idzie do testu, reszta EWIS1 moze isc do treningu.
#
# Jak dziala (po kolei):
#   1. Liczy zdjecia kukurydzy z kazdego dnia lotu EWIS1 (data jest w nazwie pliku)
#      i wybiera dzien testowy - ten, ktory ma okolo 1/3 wszystkich zdjec.
#   2. Buduje zbior zbior_ewis1_podzial/ (normalization/build.py): tylko klasa "chwast",
#      zdjecia zmniejszone x0.47 (1,18 -> 2,5 mm/px, skala WeedsGalore), dzien testowy -> test,
#      reszta -> train. Potem sprawdza, czy wszystkie etykiety zostaly rozpoznane.
#   3a. Tylko z TRENUJ=1:
#      - nadprobkuj (models/season.py): kafelki WeedsGalore z 25.05 x3, z 30.05 x2
#        + Sudars + EWIS1 train -> zbior_lotnicze/dataset/data_dane_ewis1_bt.yaml
#      - zapisuje plan treningu plan_dane_ewis1_bt.json (yolov8n, 100 epok, wejscie 1280)
#      - trenuj (models/plan.py, models/training.py) -> modele_dane_ewis1_bt/model.pt
#      - ocenia ten model na tescie WeedsGalore z TTA
#   3. Ocenia kazdy model (ocen-detektor: mission/detector.py + models/evaluation.py)
#      na dniu testowym EWIS1 - raz bez TTA, raz z TTA, prog pewnosci 0.15.
#   4. Drukuje tabele: AP50, czulosc, precyzja (bez TTA | z TTA)
#      + wynik dane_ewis1_bt na WeedsGalore.
#   Kazdy krok pomija to, co juz jest (zbior, model.pt, plik .json) - mozna przerwac
#   i uruchomic ponownie, skrypt pojdzie dalej.
#
# Argumenty (zmienne srodowiskowe, wszystkie opcjonalne):
#   EWIS1_DIR - folder z EWIS1, domyslnie "dane/EWIS1 Maize, Sorghum & Weed Dataset" (albo dane/ewis1)
#   TEST_DNI  - dzien testowy RRRRMMDD, np. 20220509; domyslnie wybrany w kroku 1
#   TRENUJ    - 1 = tez krok 3a (trening ok. 3-4 h; pomijany, jesli model.pt juz jest)
#
# Zwraca (zapisuje):
#   zbior_ewis1_podzial/          - zbior YOLO: dataset/, build_info.json, test_dni.txt
#   wyniki/ewis1_test/*.json      - <model>.json, <model>_tta.json, dane_ewis1_bt_wg_tta.json
#   modele_dane_ewis1_bt/model.pt - tylko z TRENUJ=1 (+ log_dane_ewis1_bt.txt)
#
# Uzywany w:
#   scripts/final_candidates.sh, color_experiment.sh - potrzebuja zbior_ewis1_podzial z kroku 2
#
#   bash scripts/ewis1_test.sh                         # kilka minut
#   TEST_DNI="20220509" bash scripts/ewis1_test.sh     # wybrany dzien
#   TRENUJ=1 caffeinate -i bash scripts/ewis1_test.sh  # + trening, ok. 4 h (caffeinate: Mac nie usypia)
set -euo pipefail                 # stop przy bledzie, nieustawionej zmiennej i bledzie w potoku
cd "$(dirname "$0")/.."           # glowny folder repo - dziala niezaleznie skad odpalisz
E=${EWIS1_DIR:-"dane/EWIS1 Maize, Sorghum & Weed Dataset"}   # ${X:-domyslna}: X albo domyslna
[ -d "$E" ] || E=dane/ewis1       # nie ma takiego folderu -> krotsza nazwa
Z=zbior_ewis1_podzial             # tu powstaje zbior z odlozonym dniem
W=wyniki/ewis1_test               # tu ida wyniki ocen (.json)
mkdir -p "$W"
krok() { echo; echo "=== $*"; }   # naglowek kroku w terminalu

krok "1. Dni lotu (kukurydza)"
# Python liczy zdjecia 'maize' na dzien (data _RRRRMMDD_ z nazwy), lista dni -> stderr (widac w
# terminalu), na stdout tylko wybrany dzien -> trafia do WYBOR
WYBOR=$(python - "$E" <<'PY'
import collections, os, re, sys
n = collections.Counter(m.group(1) for p in os.listdir(os.path.join(sys.argv[1], "images"))
                        if "maize" in p.lower() and (m := re.search(r"_(20\d{6})_", p)))
for d, k in sorted(n.items()):
    print(f"  {d}: {k} zdjec", file=sys.stderr)
cel = sum(n.values()) / 3                 # okolo 1/3 zdjec do testu
print(min(n, key=lambda d: (abs(n[d] - cel), d)))   # najblizej celu; remis -> wczesniejsza data
PY
)
TEST_DNI=${TEST_DNI:-$WYBOR}      # mozna wymusic dzien z zewnatrz
echo "  dzien testowy: $TEST_DNI"

krok "2. Zbior z odlozonym dniem"
# zbior juz zbudowany dla tego samego dnia -> pomijamy budowanie
if [ -f $Z/test_dni.txt ] && [ "$(cat $Z/test_dni.txt)" = "$TEST_DNI" ]; then
    echo "  $Z juz jest dla $TEST_DNI - pomijam"
else
    rm -rf $Z
    # tylko chwasty, skala x0.47, dzien testowy -> test, reszta -> train (proporcje 1 0 0)
    python -m normalization.build --ewis1 "$E" -o $Z --tryb-klas chwasty --skala ewis1=0.47 \
           --proporcje 1 0 0 --ewis1-test-dni $TEST_DNI > /dev/null
    echo "$TEST_DNI" > $Z/test_dni.txt      # zapamietuje, dla jakiego dnia jest zbior
fi
# kontrola: brak nierozpoznanych etykiet + liczba kafelkow w train i test
python - <<PY
import json
from pathlib import Path
i = json.loads(Path("$Z/build_info.json").read_text())
assert not i.get("nierozpoznane_etykiety"), i["nierozpoznane_etykiety"]
for c in ("train", "test"):
    print(f"  {c:<6} kafelkow {len(list(Path('$Z/dataset/images', c).glob('*.jpg'))):>4}")
PY

krok "3. Ocena modeli, ktore nie widzialy EWIS1"
MODELE="wg_sudars wg_sudars_s1 sezon_nadprob sezon_nadprob_s1"   # _s1 = drugie ziarno (seed)
if [ "${TRENUJ:-0}" = "1" ]; then
    krok "3a. Trening: WeedsGalore (nadprobkowanie) + Sudars + EWIS1 bez dnia $TEST_DNI"
    # kopie kafelkow wczesnych dni (male chwasty) + dodatkowe zbiory treningowe -> nowy data.yaml
    python -m models.cli nadprobkuj zbior_lotnicze/dataset --waga 2023-05-25=3 --waga 2023-05-30=2 \
           --dodaj-trening zbior_naziemne/dataset --dodaj-trening $Z/dataset \
           -o zbior_lotnicze/dataset/data_dane_ewis1_bt.yaml > /dev/null
    # plan treningu: jeden etap, yolov8n od wag startowych, early stopping po 30 epokach
    cat > plan_dane_ewis1_bt.json <<'PLAN'
{"model_bazowy": "yolov8n.pt", "batch": 16, "cierpliwosc": 30, "seed": 0,
 "etapy": [{"nazwa": "dane-ewis1-bt", "dane": "zbior_lotnicze/dataset/data_dane_ewis1_bt.yaml",
            "epoki": 100, "lr0": 0.01, "wejscie": 1280}]}
PLAN
    # trening tylko gdy nie ma jeszcze model.pt; caly wypis do logu
    [ -f modele_dane_ewis1_bt/model.pt ] || \
        python -m models.cli trenuj --plan plan_dane_ewis1_bt.json -o modele_dane_ewis1_bt > log_dane_ewis1_bt.txt 2>&1
    # ocena na glownym tescie (WeedsGalore) z TTA
    python -m models.cli ocen-detektor modele_dane_ewis1_bt/model.pt zbior_lotnicze/dataset/data.yaml \
           --prog 0.15 --tta -o "$W/dane_ewis1_bt_wg_tta.json" > /dev/null
    MODELE="$MODELE dane_ewis1_bt"
fi
# kazdy model: bez TTA ("") i z TTA ("--tta"); brak model.pt -> pomijamy model,
# jest juz .json -> pomijamy ocene
for M in $MODELE; do
    [ -f modele_$M/model.pt ] || continue
    for T in "" "--tta"; do
        S=$([ -n "$T" ] && echo "_tta" || echo "")     # koncowka nazwy pliku: "" albo "_tta"
        [ -f "$W/${M}${S}.json" ] || python -m models.cli ocen-detektor modele_$M/model.pt $Z/dataset/data.yaml \
            --split test --prog 0.15 $T -o "$W/${M}${S}.json" > /dev/null
    done
    echo "  $M ocenione"
done

krok "4. Wynik na drugim tescie (EWIS1, dzien $TEST_DNI)"
# tabela z plikow .json: AP50, czulosc, precyzja bez TTA | z TTA (+ dane_ewis1_bt na WeedsGalore)
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