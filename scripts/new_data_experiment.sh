#!/usr/bin/env bash
# Nowe dane w treningu, przepisem, ktory wygral: nadprobkowanie WeedsGalore
# (25.05 x3, 30.05 x2) + Sudars, a do tego:
#   dane_ewis1         + EWIS1, tylko kukurydza (30 zdjec z "Maize" w nazwie)
#   dane_ewis1_sorgo   + caly EWIS1 z sorgo (88 zdjec) - czy wiecej danych przewaza nad
#                        ryzykiem, ze sorgo jako tlo nauczy model pomijac trawy
#   dane_ewis1_pheno   + EWIS1 (kukurydza) + PhenoBench (dron, burak, trzy fazy wzrostu)
# Walidacja i test bez zmian (tylko WeedsGalore), ocena jak dotad: bez TTA i z TTA.
#
#   caffeinate -i bash scripts/new_data_experiment.sh                          # trzy, ok. 12 h
#   WARIANTY="dane_ewis1" caffeinate -i bash scripts/new_data_experiment.sh    # jeden, ok. 4 h
# Kroki juz zrobione (zbiory, modele) sa pomijane.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=${SEED:-0}
SUF=$([ "$SEED" = "0" ] && echo "" || echo "_s$SEED")
WARIANTY=${WARIANTY:-"dane_ewis1 dane_ewis1_sorgo dane_ewis1_pheno"}
EWIS1_DIR=${EWIS1_DIR:-"dane/EWIS1 Maize, Sorghum & Weed Dataset"}
[ -d "$EWIS1_DIR" ] || EWIS1_DIR=dane/ewis1
PHENO_DIR=${PHENO_DIR:-dane/PhenoBench}
[ -d "$PHENO_DIR" ] || PHENO_DIR=dane/phenobench/PhenoBench
SKALA_EWIS1=${SKALA_EWIS1:-0.47}          # 1,18 mm/px -> 2,5 mm/px
MAKS_PHENO=${MAKS_PHENO:-1000}
DANE=zbior_lotnicze/dataset/data.yaml
W=wyniki/dane
mkdir -p "$W"
krok() { echo; echo "=== $*"; }
nierozpoznane() {
    python - "$1" <<'PY'
import json, sys
nz = json.load(open(sys.argv[1], encoding="utf-8")).get("nierozpoznane_etykiety") or {}
if nz:
    print("  NIEROZPOZNANE ETYKIETY - wklej to i dopisze je do taksonomii:")
    for k, v in nz.items():
        print(f"    {k}: {v}")
    sys.exit(1)
print("  wszystkie etykiety rozpoznane")
PY
}

krok "0. Kontrola"
[ -f modele_sezon_nadprob/model.pt ] || { echo "brak modele_sezon_nadprob (punkt odniesienia)"; exit 1; }
[ -d zbior_naziemne/dataset/images/train ] || { echo "brak zbior_naziemne (Sudars)"; exit 1; }
[ -d "$EWIS1_DIR/bounding_boxes" ] || { echo "brak $EWIS1_DIR/bounding_boxes"; exit 1; }

krok "1. EWIS1 (skala x$SKALA_EWIS1, wszystko do treningu)"
if [ -d zbior_ewis1/dataset/images/train ]; then
    echo "  zbior_ewis1 juz jest - pomijam"
else
    python -m normalization.build --ewis1 "$EWIS1_DIR" -o zbior_ewis1 --tryb-klas chwasty \
           --skala ewis1=$SKALA_EWIS1 --proporcje 1 0 0
fi
nierozpoznane zbior_ewis1/build_info.json
if [[ " $WARIANTY " == *" dane_ewis1_sorgo "* ]]; then
    if [ -d zbior_ewis1_sorgo/dataset/images/train ]; then
        echo "  zbior_ewis1_sorgo juz jest - pomijam"
    else
        python -m normalization.build --ewis1 "$EWIS1_DIR" -o zbior_ewis1_sorgo --tryb-klas chwasty \
               --skala ewis1=$SKALA_EWIS1 --proporcje 1 0 0 --ewis1-z-sorgo
    fi
    nierozpoznane zbior_ewis1_sorgo/build_info.json
fi

if [[ " $WARIANTY " == *" dane_ewis1_pheno "* ]]; then
    krok "2. PhenoBench: skala z wielkosci chwastow, potem zbior"
    if [ -d zbior_phenobench/dataset/images/train ]; then
        echo "  zbior_phenobench juz jest - pomijam"
    else
        [ -d "$PHENO_DIR/train" ] || { echo "brak $PHENO_DIR/train"; exit 1; }
        rm -rf /tmp/pheno_probe
        python -m normalization.build --phenobench "$PHENO_DIR" -o /tmp/pheno_probe --tryb-klas chwasty \
               --maks-zdjec phenobench=150 --proporcje 1 0 0 > /dev/null
        python -m models.cli analiza /tmp/pheno_probe/dataset/data.yaml -o /tmp/pheno_probe/analiza > /dev/null
        SKALA_PHENO=$(python - <<'PY'
import json
m = json.load(open("/tmp/pheno_probe/analiza/raport.json"))["razem"]["bok_px"]["mediana"]
# mediana boku chwastu w WeedsGalore: ok. 13,7 px przy 2,5 mm/px
print(f"{min(1.0, max(0.15, 13.7 / m)):.3f}")
PY
)
        echo "  skala PhenoBench: x$SKALA_PHENO"
        python -m normalization.build --phenobench "$PHENO_DIR" -o zbior_phenobench --tryb-klas chwasty \
               --skala phenobench=$SKALA_PHENO --maks-zdjec phenobench=$MAKS_PHENO --proporcje 1 0 0
    fi
    nierozpoznane zbior_phenobench/build_info.json
fi

krok "3. Wielkosc chwastow po przeskalowaniu (mediana boku [px])"
for Z in zbior_lotnicze zbior_ewis1 zbior_ewis1_sorgo zbior_phenobench; do
    [ -f $Z/dataset/data.yaml ] || continue
    python -m models.cli analiza $Z/dataset/data.yaml -o analiza_$Z > /dev/null
    python -c "import json; r=json.load(open('analiza_$Z/raport.json'))['razem']; print(f\"  {'$Z':<18} mediana {r['bok_px']['mediana']:>6} px   obiektow {r['obiektow']:>6}   obrazow {r['obrazow']:>5}\")"
done
echo "  Mediany powinny byc podobne (rzad 10-20 px). Wyraznie wieksza: zmniejsz SKALA_*, mniejsza: zwieksz."

krok "4. Dane treningowe i plany (ziarno $SEED)"
for M in $WARIANTY; do
    DOD="--dodaj-trening zbior_naziemne/dataset --dodaj-trening zbior_ewis1/dataset"
    [ "$M" = "dane_ewis1_sorgo" ] && DOD="--dodaj-trening zbior_naziemne/dataset --dodaj-trening zbior_ewis1_sorgo/dataset"
    [ "$M" = "dane_ewis1_pheno" ] && DOD="$DOD --dodaj-trening zbior_phenobench/dataset"
    python -m models.cli nadprobkuj zbior_lotnicze/dataset --waga 2023-05-25=3 --waga 2023-05-30=2 \
           $DOD -o zbior_lotnicze/dataset/data_$M.yaml > /dev/null
    python -m models.cli sprawdz zbior_lotnicze/dataset/data_$M.yaml | grep -E "train|Dane|problem" || true
    cat > plan_$M$SUF.json <<PLAN
{"model_bazowy": "yolov8n.pt", "batch": 16, "cierpliwosc": 30, "seed": $SEED,
 "etapy": [{"nazwa": "$M", "dane": "zbior_lotnicze/dataset/data_$M.yaml",
            "epoki": 100, "lr0": 0.01, "wejscie": 1280}]}
PLAN
done

krok "5. Trening"
for M in $WARIANTY; do
    if [ -f modele_$M$SUF/model.pt ]; then echo "  modele_$M$SUF juz jest - pomijam"; continue; fi
    echo "  $M$SUF ... postep: grep -a \"/100\" log_$M$SUF.txt | tail -n 1"
    python -m models.cli trenuj --plan plan_$M$SUF.json -o modele_$M$SUF > log_$M$SUF.txt 2>&1
done

krok "6. Ocena: bez TTA (wedlug dni, mAP50) i z TTA (konfiguracja koncowa)"
for M in sezon_nadprob $WARIANTY; do
    MOD=modele_$M$SUF/model.pt
    [ -f "$MOD" ] || continue
    [ -f "$W/$M$SUF.json" ] || python -m models.cli ocen-sezon "$MOD" "$DANE" --wejscie 1280 --prog 0.15 -o "$W/$M$SUF.json" > /dev/null
    [ -f "$W/${M}${SUF}_map.json" ] || python -m models.cli ocen "$MOD" "$DANE" --wejscie 1280 -o "$W/${M}${SUF}_map.json" > /dev/null
    [ -f "$W/${M}${SUF}_tta.json" ] || python -m models.cli ocen-detektor "$MOD" "$DANE" --wejscie 1280 --prog 0.15 --tta -o "$W/${M}${SUF}_tta.json" > /dev/null
    echo "  $M$SUF ocenione"
done

krok "7. Porownanie z modelem bez nowych danych"
SUF="$SUF" WARIANTY="$WARIANTY" python - <<'PY'
import json, os
from pathlib import Path
W, suf = Path("wyniki/dane"), os.environ.get("SUF", "")
czytaj = lambda p: json.loads(p.read_text()) if p.exists() else None
dni = ["2023-05-25", "2023-05-30", "2023-06-06"]
opis = {"sezon_nadprob": "bez nowych danych", "dane_ewis1": "+ EWIS1 (kukurydza)",
        "dane_ewis1_sorgo": "+ EWIS1 z sorgo", "dane_ewis1_pheno": "+ EWIS1 + PhenoBench"}
print(f"\n  {'':<24}| {'bez TTA, próg 0,15':^41} | {'z TTA, próg 0,15':^33}")
print(f"  {'model':<24}| {'mAP50':>6}{'25.05':>7}{'30.05':>7}{'06.06':>7}{'razem':>7}{'prec.':>7} | "
      f"{'AP50':>6}{'25.05':>7}{'razem':>7}{'prec.':>7}{'fałsz.':>7}")
odn = None
for k in ["sezon_nadprob"] + os.environ["WARIANTY"].split():
    s, m, t = (czytaj(W / f"{k}{suf}{x}.json") for x in ("", "_map", "_tta"))
    if not (s and t):
        continue
    j = s["jeden_prog"]
    cz = [j.get(d, {}).get("czulosc", float("nan")) for d in dni]
    tm = t["wedlug_dni"].get(dni[0], {}).get("czulosc", float("nan"))
    r = t["przy_progu"]
    print(f"  {opis.get(k, k):<24}| {(m or {}).get('mAP50', float('nan')):>6.3f}" + "".join(f"{c:>7.3f}" for c in cz)
          + f"{j['razem']['czulosc']:>7.3f}{j['razem']['precyzja']:>7.3f} | {t['AP50']:>6.3f}{tm:>7.3f}"
          f"{r['czulosc']:>7.3f}{r['precyzja']:>7.3f}{r['falszywe']:>7}")
    if k == "sezon_nadprob":
        odn = (t["AP50"], tm)
    elif odn:
        print(f"  {'':<24}  z TTA: AP50 {t['AP50'] - odn[0]:+.3f}, 25.05 {tm - odn[1]:+.3f} względem modelu bez nowych danych")
print("\n  Poprawa liczy się, gdy AP50 i 25 maja rosną; potwierdzenie drugim ziarnem: SEED=1 ...")
PY