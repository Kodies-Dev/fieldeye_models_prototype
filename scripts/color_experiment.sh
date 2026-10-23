#!/usr/bin/env bash
# Kolory WeedsGalore: czy model uczony na WeedsGalore z korekta gamma sRGB (jak zwykly
# aparat) lepiej przenosi sie na inne pole i kamere RGB (odlozony dzien EWIS1)?
#
#   kolory_srgb         WeedsGalore sRGB (nadprobkowanie) + Sudars
#   kolory_srgb_ewis1   to samo + EWIS1 bez dnia testowego (kandydat na model koncowy)
#
# Punkt odniesienia: sezon_nadprob (WeedsGalore liniowe) i dane_ewis1_bt (liniowe + EWIS1,
# z TRENUJ=1 bash scripts/ewis1_test.sh). Ocena: test WeedsGalore (w kolorach, w jakich
# model sie uczyl) i odlozony dzien EWIS1, bez TTA i z TTA.
#
# Wymaga: zbior_lotnicze (oryginalny), zbior_naziemne (Sudars), zbior_ewis1_podzial (ewis1_test.sh).
#
#   caffeinate -i bash scripts/color_experiment.sh                              # oba, ok. 7-8 h
#   WARIANTY="kolory_srgb_ewis1" caffeinate -i bash scripts/color_experiment.sh # jeden, ok. 4 h
set -euo pipefail
cd "$(dirname "$0")/.."
WARIANTY=${WARIANTY:-"kolory_srgb kolory_srgb_ewis1"}
SEED=${SEED:-0}
SUF=$([ "$SEED" = "0" ] && echo "" || echo "_s$SEED")
Z=zbior_lotnicze_srgb
E=zbior_ewis1_podzial
W=wyniki/kolory
mkdir -p "$W"
krok() { echo; echo "=== $*"; }

krok "0. Kontrola"
for d in zbior_lotnicze zbior_naziemne/dataset/images/train $E/dataset/images/test; do
    [ -e "$d" ] || { echo "brak $d"; exit 1; }
done
[ -f zbior_lotnicze/build_info.json ] || { echo "brak zbior_lotnicze/build_info.json"; exit 1; }

krok "1. WeedsGalore w kolorach sRGB (ten sam podzial i kafelki co zbior_lotnicze)"
if [ -d $Z/dataset/images/test ]; then
    echo "  $Z juz jest - pomijam"
else
    ARGS=$(python - <<'PY'
import json, shlex
i = json.load(open("zbior_lotnicze/build_info.json"))
p, z = i["parametry"], i["zrodla"]
a = ["--weedsgalore", z["weedsgalore"], "--tryb-klas", p["tryb_klas"], "--kafelek", str(p["kafelek"]),
     "--zakladka", str(p["zakladka"]), "--udzial-pustych", str(p["udzial_pustych"]),
     "--prog-dedup", str(p["prog_dedup"]), "--seed", str(p["seed"])]
print(" ".join(shlex.quote(x) for x in a))
PY
)
    echo "  normalizacja: $ARGS --weedsgalore-kolory srgb"
    eval python -m normalization.build $ARGS -o $Z --weedsgalore-kolory srgb > /dev/null
fi
python - <<PY
from pathlib import Path
for c in ("train", "val", "test"):
    a = sorted(p.name for p in Path("zbior_lotnicze/dataset/images", c).glob("*.jpg"))
    b = sorted(p.name for p in Path("$Z/dataset/images", c).glob("*.jpg"))
    assert a == b, f"{c}: inne kafelki niz w zbior_lotnicze ({len(a)} vs {len(b)})"
    print(f"  {c:<6} {len(b):>4} kafelkow - te same co w zbior_lotnicze")
PY

krok "2. Dane treningowe i plany (ziarno $SEED)"
for M in $WARIANTY; do
    DOD="--dodaj-trening zbior_naziemne/dataset"
    [ "$M" = "kolory_srgb_ewis1" ] && DOD="$DOD --dodaj-trening $E/dataset"
    python -m models.cli nadprobkuj $Z/dataset --waga 2023-05-25=3 --waga 2023-05-30=2 \
           $DOD -o $Z/dataset/data_$M.yaml > /dev/null
    python -m models.cli sprawdz $Z/dataset/data_$M.yaml | grep -E "train |Dane|problem" || true
    cat > plan_$M$SUF.json <<PLAN
{"model_bazowy": "yolov8n.pt", "batch": 16, "cierpliwosc": 30, "seed": $SEED,
 "etapy": [{"nazwa": "$M", "dane": "$Z/dataset/data_$M.yaml", "epoki": 100, "lr0": 0.01, "wejscie": 1280}]}
PLAN
done

krok "3. Trening"
for M in $WARIANTY; do
    if [ -f modele_$M$SUF/model.pt ]; then echo "  modele_$M$SUF juz jest - pomijam"; continue; fi
    echo "  $M$SUF ... postep: grep -a \"/100\" log_$M$SUF.txt | tail -n 1"
    python -m models.cli trenuj --plan plan_$M$SUF.json -o modele_$M$SUF > log_$M$SUF.txt 2>&1
done

krok "4. Ocena na obu testach"
ocen() {   # model, data.yaml, plik wyniku
    [ -f "$3.json" ] || python -m models.cli ocen-detektor "$1" "$2" --split test --prog 0.15 -o "$3.json" > /dev/null
    [ -f "$3_tta.json" ] || python -m models.cli ocen-detektor "$1" "$2" --split test --prog 0.15 --tta -o "$3_tta.json" > /dev/null
}
for M in sezon_nadprob dane_ewis1_bt $WARIANTY; do
    MOD=modele_$M$SUF/model.pt
    [ -f "$MOD" ] || continue
    case $M in kolory_*) WG=$Z ;; *) WG=zbior_lotnicze ;; esac
    ocen "$MOD" $WG/dataset/data.yaml "$W/${M}${SUF}_wg"
    ocen "$MOD" $E/dataset/data.yaml "$W/${M}${SUF}_ewis1"
    echo "  $M$SUF ocenione"
done

krok "5. Porownanie"
SUF="$SUF" python - <<'PY'
import json, os
from pathlib import Path
W, suf = Path("wyniki/kolory"), os.environ.get("SUF", "")
opis = {"sezon_nadprob": ("liniowe", "nie"), "dane_ewis1_bt": ("liniowe", "tak"),
        "kolory_srgb": ("sRGB", "nie"), "kolory_srgb_ewis1": ("sRGB", "tak")}
print(f"\n  {'model':<20}{'kolory':>8}{'EWIS1':>7} | {'WeedsGalore (TTA)':^24} | {'inne pole: EWIS1 16.05 (TTA)':^30}")
print(f"  {'':<20}{'':>8}{'':>7} | {'AP50':>7}{'25.05':>8}{'czuł.':>8} | {'AP50':>7}{'czułość':>10}{'prec.':>9}")
for m, (k, e) in opis.items():
    a, b = W / f"{m}{suf}_wg_tta.json", W / f"{m}{suf}_ewis1_tta.json"
    if not (a.exists() and b.exists()):
        continue
    a, b = json.loads(a.read_text()), json.loads(b.read_text())
    maj = a["wedlug_dni"].get("2023-05-25", {}).get("czulosc", float("nan"))
    print(f"  {m:<20}{k:>8}{e:>7} | {a['AP50']:>7.3f}{maj:>8.3f}{a['przy_progu']['czulosc']:>8.3f} | "
          f"{b['AP50']:>7.3f}{b['przy_progu']['czulosc']:>10.3f}{b['przy_progu']['precyzja']:>9.3f}")
print("\n  Najwazniejsza kolumna: AP50 na innym polu. WeedsGalore oceniany w kolorach, w jakich model sie uczyl.")
PY
