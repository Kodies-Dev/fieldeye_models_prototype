#!/usr/bin/env bash
# Sezonowosc: trzy proby poprawy wyniku na najmlodszych siewkach (25 maja),
# porownane z modelem wg_sudars tym samym kodem oceny.
#
#   A  nadprobkowanie   kafelki 25 maja x3, 30 maja x2 w treningu
#   B  A + mniej zmniejszania w augmentacji (scale 0,2 zamiast 0,5)
#   C  model wczesny    wg_sudars dostrojony na kafelkach z maja (lr0 0,001)
#   +  dla kazdego modelu: prog dobrany na walidacji osobno dla kazdego dnia
#
# Uruchomienie z katalogu stitch-live, z aktywnym .venv (ok. 7 h):
#   caffeinate -i bash scripts/season_experiments.sh
# Powtorzenie z innym ziarnem (do potwierdzenia wyniku):
#   SEED=1 caffeinate -i bash scripts/season_experiments.sh
# Tylko wybrane warianty (np. potwierdzenie samego zwyciezcy, ok. 3 h):
#   SEED=1 WARIANTY="sezon_nadprob" caffeinate -i bash scripts/season_experiments.sh
# Kroki juz zrobione (np. wytrenowany model) sa pomijane.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=${SEED:-0}
WARIANTY=${WARIANTY:-"sezon_nadprob sezon_skala sezon_wczesny"}
SUF=$([ "$SEED" = "0" ] && echo "" || echo "_s$SEED")
DANE=zbior_lotnicze/dataset/data.yaml
W=wyniki/sezon
mkdir -p "$W"
krok() { echo; echo "=== $*"; }

krok "0. Kontrola"
[ -f modele_wg_sudars/model.pt ] || { echo "brak modele_wg_sudars/model.pt"; exit 1; }
[ -d zbior_naziemne/dataset/images/train ] || { echo "brak zbior_naziemne/dataset (Sudars)"; exit 1; }

krok "1. Dane"
python -m models.cli nadprobkuj zbior_lotnicze/dataset --waga 2023-05-25=3 --waga 2023-05-30=2 \
       --dodaj-trening zbior_naziemne/dataset -o zbior_lotnicze/dataset/data_sezon.yaml
python -m models.cli sprawdz zbior_lotnicze/dataset/data_sezon.yaml
python -m models.cli wybierz-dni zbior_lotnicze/dataset --dni 2023-05-25 2023-05-30 -o zbior_maj

krok "2. Plany treningu (ziarno $SEED)"
cat > plan_sezon_nadprob$SUF.json <<PLAN
{"model_bazowy": "yolov8n.pt", "batch": 16, "cierpliwosc": 30, "seed": $SEED,
 "etapy": [{"nazwa": "sezon-nadprob", "dane": "zbior_lotnicze/dataset/data_sezon.yaml",
            "epoki": 100, "lr0": 0.01, "wejscie": 1280}]}
PLAN
cat > plan_sezon_skala$SUF.json <<PLAN
{"model_bazowy": "yolov8n.pt", "batch": 16, "cierpliwosc": 30, "seed": $SEED,
 "etapy": [{"nazwa": "sezon-skala", "dane": "zbior_lotnicze/dataset/data_sezon.yaml",
            "epoki": 100, "lr0": 0.01, "wejscie": 1280, "augmentacje": {"scale": 0.2}}]}
PLAN
cat > plan_sezon_wczesny$SUF.json <<PLAN
{"model_bazowy": "modele_wg_sudars$SUF/model.pt", "batch": 16, "cierpliwosc": 20, "seed": $SEED,
 "etapy": [{"nazwa": "sezon-wczesny", "dane": "zbior_maj/dataset/data.yaml",
            "epoki": 50, "lr0": 0.001, "wejscie": 1280}]}
PLAN

krok "3. Trening"
for M in $WARIANTY; do
    if [ -f modele_$M$SUF/model.pt ]; then
        echo "  modele_$M$SUF juz jest - pomijam"
        continue
    fi
    if [ "$M" = "sezon_wczesny" ] && [ ! -f modele_wg_sudars$SUF/model.pt ]; then
        echo "  brak modele_wg_sudars$SUF - model wczesny pominiety"
        continue
    fi
    echo "  $M$SUF ... postep: grep -a \"/\" log_$M$SUF.txt | tail -n 1"
    python -m models.cli trenuj --plan plan_$M$SUF.json -o modele_$M$SUF > log_$M$SUF.txt 2>&1
done

krok "4. Ocena: prog kazdego dnia na walidacji, test wedlug dni, mAP50"
for M in wg_sudars $WARIANTY; do
    MOD=modele_$M$SUF/model.pt
    [ -f "$MOD" ] || continue
    python -m models.cli prog-dni "$MOD" "$DANE" --wejscie 1280 -o "$W/$M${SUF}_progi.json" > /dev/null
    python -m models.cli ocen-sezon "$MOD" "$DANE" --wejscie 1280 --prog 0.15 \
           --progi "$W/$M${SUF}_progi.json" -o "$W/$M$SUF.json" > /dev/null
    python -m models.cli ocen "$MOD" "$DANE" --wejscie 1280 -o "$W/$M${SUF}_map.json" > /dev/null
    echo "  $M$SUF ocenione"
done

krok "5. Porownanie"
SUF="$SUF" python - <<'PY'
import json, os
from pathlib import Path
W, suf = Path("wyniki/sezon"), os.environ.get("SUF", "")
czytaj = lambda p: json.loads(p.read_text()) if p.exists() else None
nazwy = {"wg_sudars": "wg_sudars (odniesienie)", "sezon_nadprob": "A nadprobkowanie",
         "sezon_skala": "B A + scale 0,2", "sezon_wczesny": "C model wczesny"}
dni = ["2023-05-25", "2023-05-30", "2023-06-06"]
print(f"\n  {'model':<24}{'mAP50':>7} | {'czułość przy 0,15':^26} | {'przy progu dnia':^17} | {'precyzja':>8}")
print(f"  {'':<24}{'':>7} | {'25.05':>8}{'30.05':>9}{'06.06':>9} | {'25.05':>8}{'razem':>9} | {'razem':>8}")
odn = None
for k, opis in nazwy.items():
    s, m = czytaj(W / f"{k}{suf}.json"), czytaj(W / f"{k}{suf}_map.json")
    if not s:
        continue
    j, p = s["jeden_prog"], s.get("progi_dni", {})
    cz = [j.get(d, {}).get("czulosc", float("nan")) for d in dni]
    w = (m or {}).get("mAP50", float("nan"))
    print(f"  {opis:<24}{w:>7.3f} | {cz[0]:>8.3f}{cz[1]:>9.3f}{cz[2]:>9.3f} | "
          f"{p.get(dni[0], {}).get('czulosc', float('nan')):>8.3f}{p.get('razem', {}).get('czulosc', float('nan')):>9.3f} | "
          f"{j['razem']['precyzja']:>8.3f}")
    if k == "wg_sudars":
        odn = cz
    elif odn:
        roz = cz[0] - odn[0]
        werdykt = ("POPRAWA" if roz > 0.04 and cz[2] >= odn[2] - 0.02 else
                   "bez zmian (w granicach szumu)" if abs(roz) <= 0.04 else "gorzej")
        print(f"  {'':<24}  25.05: {roz:+.3f} względem odniesienia, 06.06: {cz[2] - odn[2]:+.3f}  -> {werdykt}")
print("\n  POPRAWA = 25 maja lepiej o ponad 4 pp (rozrzut między ziarnami) i 6 czerwca nie gorzej o ponad 2 pp.")
print("  Wariant z POPRAWĄ potwierdź drugim ziarnem: SEED=1 caffeinate -i bash scripts/season_experiments.sh")
PY