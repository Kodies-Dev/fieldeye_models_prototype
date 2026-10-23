#!/usr/bin/env bash
# Kandydaci na model koncowy - od razu z wejsciem 640, jak w pliku HEF.
# Wspolny przepis: WeedsGalore (kolory liniowe, nadprobkowanie maja) + Sudars + EWIS1
# kukurydza bez dnia testowego 16.05.2022. Warianty roznia sie tylko dodatkowymi danymi:
#
#   k640_baza    przepis jak dane_ewis1_bt, tylko wejscie 640   -> ile tracimy przy 640
#   k640_sorgo   + EWIS1 sorgo (inne pola i dni, bez dnia testowego)
#   k640_pheno   + PhenoBench (dron, burak, trzy fazy)
#
# Ocena na obu testach (WeedsGalore i odlozony dzien EWIS1), bez TTA i z TTA.
# Punkt odniesienia: dane_ewis1_bt (wejscie 1280) z scripts/ewis1_test.sh.
#
#   caffeinate -i bash scripts/final_candidates.sh                          # trzy, ok. 4-6 h
#   WARIANTY="k640_baza" caffeinate -i bash scripts/final_candidates.sh     # jeden
set -euo pipefail
cd "$(dirname "$0")/.."
WARIANTY=${WARIANTY:-"k640_baza k640_sorgo k640_pheno"}
SEED=${SEED:-0}
SUF=$([ "$SEED" = "0" ] && echo "" || echo "_s$SEED")
EWIS1_DIR=${EWIS1_DIR:-"dane/EWIS1 Maize, Sorghum & Weed Dataset"}
[ -d "$EWIS1_DIR" ] || EWIS1_DIR=dane/ewis1
PHENO_DIR=${PHENO_DIR:-dane/PhenoBench}
[ -d "$PHENO_DIR" ] || PHENO_DIR=dane/phenobench/PhenoBench
MAKS_PHENO=${MAKS_PHENO:-1000}
TEST_DNI=$(cat zbior_ewis1_podzial/test_dni.txt 2>/dev/null || echo 20220516)
E=zbior_ewis1_podzial
W=wyniki/kandydaci
mkdir -p "$W"
krok() { echo; echo "=== $*"; }
sprawdz_etykiety() {
    python - "$1" <<'PY'
import json, sys
nz = json.load(open(sys.argv[1], encoding="utf-8")).get("nierozpoznane_etykiety") or {}
if nz:
    print("  NIEROZPOZNANE ETYKIETY:", nz); sys.exit(1)
PY
}

krok "0. Kontrola"
for d in zbior_lotnicze/dataset zbior_naziemne/dataset/images/train $E/dataset/images/test; do
    [ -e "$d" ] || { echo "brak $d (zbior_ewis1_podzial: bash scripts/ewis1_test.sh)"; exit 1; }
done
echo "  dzien testowy EWIS1: $TEST_DNI"

if [[ " $WARIANTY " == *" k640_sorgo "* ]]; then
    krok "1a. EWIS1 z sorgo, bez dnia testowego"
    if [ -d zbior_ewis1_sorgo_podzial/dataset/images/train ]; then
        echo "  zbior_ewis1_sorgo_podzial juz jest - pomijam"
    else
        python -m normalization.build --ewis1 "$EWIS1_DIR" -o zbior_ewis1_sorgo_podzial --tryb-klas chwasty \
               --skala ewis1=0.47 --proporcje 1 0 0 --ewis1-z-sorgo --ewis1-test-dni $TEST_DNI > /dev/null
    fi
    sprawdz_etykiety zbior_ewis1_sorgo_podzial/build_info.json
    echo "  train: $(ls zbior_ewis1_sorgo_podzial/dataset/images/train | wc -l) kafelkow (dzien $TEST_DNI w czesci test, nieuzywanej)"
fi

if [[ " $WARIANTY " == *" k640_pheno "* ]]; then
    krok "1b. PhenoBench (skala z wielkosci chwastow)"
    if [ -d zbior_phenobench/dataset/images/train ]; then
        echo "  zbior_phenobench juz jest - pomijam"
    else
        [ -d "$PHENO_DIR/train" ] || { echo "brak $PHENO_DIR/train"; exit 1; }
        rm -rf /tmp/pheno_probe
        python -m normalization.build --phenobench "$PHENO_DIR" -o /tmp/pheno_probe --tryb-klas chwasty \
               --maks-zdjec phenobench=150 --proporcje 1 0 0 > /dev/null
        python -m models.cli analiza /tmp/pheno_probe/dataset/data.yaml -o /tmp/pheno_probe/analiza > /dev/null
        SKALA_PHENO=$(python -c "import json; m=json.load(open('/tmp/pheno_probe/analiza/raport.json'))['razem']['bok_px']['mediana']; print(f'{min(1.0, max(0.15, 13.7 / m)):.3f}')")
        echo "  skala PhenoBench: x$SKALA_PHENO"
        python -m normalization.build --phenobench "$PHENO_DIR" -o zbior_phenobench --tryb-klas chwasty \
               --skala phenobench=$SKALA_PHENO --maks-zdjec phenobench=$MAKS_PHENO --proporcje 1 0 0 > /dev/null
    fi
    sprawdz_etykiety zbior_phenobench/build_info.json
    python -m models.cli analiza zbior_phenobench/dataset/data.yaml -o analiza_zbior_phenobench > /dev/null
    python -c "import json; r=json.load(open('analiza_zbior_phenobench/raport.json'))['razem']; print(f\"  PhenoBench: {r['obrazow']} kafelkow, mediana chwastu {r['bok_px']['mediana']} px (WeedsGalore ok. 13,7)\")"
fi

krok "2. Dane treningowe i plany (wejscie 640, ziarno $SEED)"
for M in $WARIANTY; do
    DOD="--dodaj-trening zbior_naziemne/dataset --dodaj-trening $E/dataset"
    case $M in
        k640_sorgo) DOD="--dodaj-trening zbior_naziemne/dataset --dodaj-trening zbior_ewis1_sorgo_podzial/dataset" ;;
        k640_pheno) DOD="$DOD --dodaj-trening zbior_phenobench/dataset" ;;
    esac
    python -m models.cli nadprobkuj zbior_lotnicze/dataset --waga 2023-05-25=3 --waga 2023-05-30=2 \
           $DOD -o zbior_lotnicze/dataset/data_$M.yaml > /dev/null
    echo "  $M: $(python -m models.cli sprawdz zbior_lotnicze/dataset/data_$M.yaml | grep -E '^ +train ' | head -1)"
    cat > plan_$M$SUF.json <<PLAN
{"model_bazowy": "yolov8n.pt", "batch": 16, "cierpliwosc": 30, "seed": $SEED,
 "etapy": [{"nazwa": "$M", "dane": "zbior_lotnicze/dataset/data_$M.yaml", "epoki": 100, "lr0": 0.01, "wejscie": 640}]}
PLAN
done

krok "3. Trening"
for M in $WARIANTY; do
    if [ -f modele_$M$SUF/model.pt ]; then echo "  modele_$M$SUF juz jest - pomijam"; continue; fi
    echo "  $M$SUF ... postep: grep -a \"/100\" log_$M$SUF.txt | tail -n 1"
    python -m models.cli trenuj --plan plan_$M$SUF.json -o modele_$M$SUF > log_$M$SUF.txt 2>&1
done

krok "4. Ocena na obu testach"
ocen() {   # model, wejscie, data.yaml, plik wyniku
    [ -f "$4.json" ] || python -m models.cli ocen-detektor "$1" "$3" --wejscie "$2" --split test --prog 0.15 -o "$4.json" > /dev/null
    [ -f "$4_tta.json" ] || python -m models.cli ocen-detektor "$1" "$3" --wejscie "$2" --split test --prog 0.15 --tta -o "$4_tta.json" > /dev/null
}
for M in dane_ewis1_bt $WARIANTY; do
    MOD=modele_$M$SUF/model.pt
    [ -f "$MOD" ] || continue
    WEJ=$([ "$M" = "dane_ewis1_bt" ] && echo 1280 || echo 640)
    ocen "$MOD" $WEJ zbior_lotnicze/dataset/data.yaml "$W/${M}${SUF}_wg"
    ocen "$MOD" $WEJ $E/dataset/data.yaml "$W/${M}${SUF}_ewis1"
    echo "  $M$SUF ocenione (wejscie $WEJ)"
done

krok "5. Porownanie"
SUF="$SUF" WARIANTY="$WARIANTY" python - <<'PY'
import json, os
from pathlib import Path
W, suf = Path("wyniki/kandydaci"), os.environ.get("SUF", "")
opis = {"dane_ewis1_bt": ("1280", "punkt odniesienia"), "k640_baza": ("640", "przepis bazowy"),
        "k640_sorgo": ("640", "+ sorgo"), "k640_pheno": ("640", "+ PhenoBench")}
print(f"\n  {'model':<16}{'wej.':>6}  {'dane':<18}| {'WeedsGalore (TTA)':^24} | {'EWIS1 16.05 (TTA)':^27}")
print(f"  {'':<16}{'':>6}  {'':<18}| {'AP50':>7}{'25.05':>8}{'czuł.':>8} | {'AP50':>7}{'czułość':>10}{'prec.':>9}")
odn = None
for m in ["dane_ewis1_bt"] + os.environ["WARIANTY"].split():
    a, b = W / f"{m}{suf}_wg_tta.json", W / f"{m}{suf}_ewis1_tta.json"
    if not (a.exists() and b.exists()):
        continue
    a, b = json.loads(a.read_text()), json.loads(b.read_text())
    maj = a["wedlug_dni"].get("2023-05-25", {}).get("czulosc", float("nan"))
    w, d = opis.get(m, ("?", m))
    print(f"  {m:<16}{w:>6}  {d:<18}| {a['AP50']:>7.3f}{maj:>8.3f}{a['przy_progu']['czulosc']:>8.3f} | "
          f"{b['AP50']:>7.3f}{b['przy_progu']['czulosc']:>10.3f}{b['przy_progu']['precyzja']:>9.3f}")
    if m == "k640_baza":
        odn = b["AP50"]
    elif odn is not None and m != "dane_ewis1_bt":
        print(f"  {'':<42}  EWIS1 AP50 {b['AP50'] - odn:+.3f} wzgledem k640_baza")
print("\n  Wybor: najwyzsze AP50 na EWIS1 bez wyraznego spadku na WeedsGalore (zwlaszcza 25.05).")
PY