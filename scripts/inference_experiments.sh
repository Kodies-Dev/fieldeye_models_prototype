#!/usr/bin/env bash
# Trzy poprawy bez treningu, zmierzone naraz:
#   1. polaczenie dwoch modeli (ensemble, WBF): wg_sudars + wg_sudars_s1
#   2. detekcja na odbiciu poziomym (TTA)
#   3. niski prog z glosowaniem: slabe wykrycie przyjete, gdy widac je na >= 2 kadrach
# Dla kazdej konfiguracji: ocena na kafelkach testowych (AP50, czulosc wedlug dni)
# oraz symulowany lot (scalanie / laczenie widokow / glosowanie / niski prog + glosowanie).
# Kilka-kilkanascie minut; najlepiej, gdy nic innego nie trenuje.
#
#   bash scripts/inference_experiments.sh
#   ZAKLOCENIA=1 bash scripts/inference_experiments.sh      # lot takze z zakloceniami kamery
set -euo pipefail
cd "$(dirname "$0")/.."
DANE=zbior_lotnicze/dataset/data.yaml
W=wyniki/inferencja
mkdir -p "$W"
A=modele_wg_sudars/model.pt
B=modele_wg_sudars_s1/model.pt
N=modele_sezon_nadprob/model.pt
N1=modele_sezon_nadprob_s1/model.pt

# nazwa | modele | tta
KONF=("wg_sudars|$A|" "tta|$A|--tta")
[ -f "$B" ] && KONF+=("ensemble|$A $B|" "ensemble_tta|$A $B|--tta")
[ -f "$N" ] && KONF+=("nadprob|$N|")
[ -f "$N" ] && [ -f "$N1" ] && KONF+=("nadprob_ensemble|$N $N1|")

for k in "${KONF[@]}"; do
    IFS='|' read -r NAZWA MODELE TTA <<< "$k"
    echo "=== $NAZWA"
    python -m models.cli ocen-detektor $MODELE "$DANE" --wejscie 1280 --prog 0.15 $TTA \
           -o "$W/${NAZWA}_kafelki.json" > /dev/null
    python -m mission.cli symulacja --model $MODELE $TTA --prog 0.15 -o "$W/${NAZWA}_lot.json" > /dev/null
    if [ "${ZAKLOCENIA:-0}" = "1" ]; then
        python -m mission.cli symulacja --model $MODELE $TTA --prog 0.15 --zaklocenia \
               -o "$W/${NAZWA}_lot_zaklocenia.json" > /dev/null
    fi
done

python - <<'PY'
import json
from pathlib import Path
W = Path("wyniki/inferencja")
czytaj = lambda p: json.loads(p.read_text()) if p.exists() else None
konf = [p.name[:-len("_kafelki.json")] for p in sorted(W.glob("*_kafelki.json"))]
kolejnosc = ["wg_sudars", "tta", "ensemble", "ensemble_tta", "nadprob", "nadprob_ensemble"]
konf.sort(key=lambda k: kolejnosc.index(k) if k in kolejnosc else 99)
dni = ["2023-05-25", "2023-05-30", "2023-06-06"]

print("\n  1) KAFELKI TESTOWE (jak dotychczasowe oceny modeli)")
print(f"  {'konfiguracja':<18}{'AP50':>7} | {'czułość przy 0,15':^34} | {'precyzja':>8}")
print(f"  {'':<18}{'':>7} | {'25.05':>8}{'30.05':>8}{'06.06':>8}{'razem':>10} | {'razem':>8}")
for k in konf:
    w = czytaj(W / f"{k}_kafelki.json")
    d = w["wedlug_dni"]
    print(f"  {k:<18}{w['AP50']:>7.3f} | " + "".join(f"{d.get(x, {}).get('czulosc', float('nan')):>8.3f}" for x in dni)
          + f"{w['przy_progu']['czulosc']:>10.3f} | {w['przy_progu']['precyzja']:>8.3f}")

sposoby = [("scalanie", "scalanie"), ("laczenie_widokow", "łączenie widoków"),
           ("laczenie_z_glosowaniem", "+ głosowanie"), ("niski_prog_z_glosowaniem", "niski próg + głos.")]
for wariant, opis in (("lot", "bez zakłóceń"), ("lot_zaklocenia", "z zakłóceniami")):
    if not any((W / f"{k}_{wariant}.json").exists() for k in konf):
        continue
    print(f"\n  2) SYMULOWANY LOT, {opis}: czułość na mozaice (fałszywe alarmy)")
    print(f"  {'konfiguracja':<18}{'sposób':<20}" + "".join(f"{x[5:]:>15}" for x in dni) + f"{'precyzja razem':>16}")
    for k in konf:
        w = czytaj(W / f"{k}_{wariant}.json")
        if not w:
            continue
        for i, (s, nazwa) in enumerate(sposoby):
            r = w["dni"]
            kom = "".join(f"{r[x][s]['czulosc']:>8.3f} ({r[x][s]['falszywe']:>4})" if x in r else f"{'—':>15}" for x in dni)
            t = sum(r[x][s]["trafione"] for x in r); f = sum(r[x][s]["falszywe"] for x in r)
            print(f"  {k if i == 0 else '':<18}{nazwa:<20}{kom}{t / max(1, t + f):>16.3f}")
print("\n  Ensemble i TTA zmieniają skalę pewności (średnia z przebiegów), więc porównuj je głównie po AP50;")
print("  czułość przy stałym progu 0,15 może przez to spaść, choć model jest lepszy.")
PY