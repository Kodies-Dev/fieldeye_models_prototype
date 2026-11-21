#!/usr/bin/env bash
# final_model.sh - model koncowy FIELDEYE od zera: normalizacja -> trening -> ocena -> paczka ONNX.
#
# Po co:
#   Jeden skrypt, ktory z surowych danych w dane/ odtwarza najlepszy model
#   (models/weights/fieldeye_final.pt, w eksperymentach: dane_ewis1_bt) i wszystko,
#   czego potrzebuje app.py. Po nim dziala: streamlit run app.py
#
# Jak dziala (po kolei):
#   1. normalizacja 3 zbiorow (normalization/build.py), tylko klasa "chwast":
#      WeedsGalore -> zbior_lotnicze/       (dron; trening, walidacja, test)
#      Sudars      -> zbior_naziemne/       (z ziemi; tylko trening)
#      EWIS1       -> zbior_ewis1_podzial/  (skala x0.47; dzien 20220516 = test, reszta = trening)
#   2. trening: WeedsGalore z nadprobkowaniem (25.05 x3, 30.05 x2) + Sudars + EWIS1 trening;
#      yolov8n, 100 epok, wejscie 1280 -> modele_dane_ewis1_bt/model.pt
#      -> kopia do models/weights/fieldeye_final.pt (tego modelu uzywa app.py)
#   3. ocena tego modelu na 2 testach: EWIS1 (dzien 20220516) i WeedsGalore, bez i z TTA, prog 0.15
#   4. paczka ONNX dla Hailo: ONNX (1280, opset 11) + zdjecia kalibracyjne + compile_hef.py
#   5. tabela wynikow (punkt odniesienia, AP50 z TTA: EWIS1 0.817, WeedsGalore 0.714)
#   Kroki juz zrobione sa pomijane - mozna przerwac i uruchomic ponownie.
#
# Wymaga:
#   aktywne .venv z requirements.txt i surowe dane:
#   dane/weedsgalore-dataset, dane/sudars/dataset, "dane/EWIS1 Maize, Sorghum & Weed Dataset"
#
# Zwraca (zapisuje):
#   zbior_lotnicze/, zbior_naziemne/, zbior_ewis1_podzial/  - zbiory YOLO (app.py: kafelki z .../images/test)
#   models/weights/fieldeye_final.pt                       - model koncowy (+ log_dane_ewis1_bt.txt)
#   wyniki/model_koncowy/*.json                            - ewis1.json, ewis1_tta.json, wg.json, wg_tta.json
#   hailo/hailo_final.zip                                  - do kompilacji HEF na WSL (scripts/HAILO.md krok 2)
#
#   caffeinate -i bash scripts/final_model.sh     # ok. 4 h z treningiem (caffeinate: Mac nie usypia)
set -euo pipefail                    # stop przy bledzie, nieustawionej zmiennej i bledzie w potoku
cd "$(dirname "$0")/.."              # glowny folder repo - dziala niezaleznie, skad odpalisz

WG=dane/weedsgalore-dataset          # surowe WeedsGalore (dron)
SU=dane/sudars/dataset               # surowy Sudars (z ziemi)
E="dane/EWIS1 Maize, Sorghum & Weed Dataset"   # surowy EWIS1 (inny dron, zwykla kamera RGB podobna do zuytej w naszym projekcie hardware)
TEST_DNI=20220516                    # dzien EWIS1 odlozony do testu (ok. 1/3 zdjec kukurydzy)
L=zbior_lotnicze                     # WeedsGalore po normalizacji tez z odlozonymi do testu
N=zbior_naziemne                     # Sudars po normalizacji nie sa ta zdj  z drona brak odlozonych
Z=zbior_ewis1_podzial                # EWIS1 po normalizacji, z odlozonym dniem
M=models/weights/fieldeye_final.pt   # model koncowy - domyslny w app.py
W=wyniki/model_koncowy               # tu ida wyniki oceny (.json)
mkdir -p "$W"
krok() { echo; echo "=== $*"; }    # naglowek kroku w terminalu, np. krok "1. Normalizacja"
# bez surowych danych nie ma czego normalizowac - stop od razu, z nazwa brakujacego folderu
for d in "$WG" "$SU" "$E"; do [ -d "$d" ] || { echo "brak surowych danych: $d"; exit 1; }; done #czy istnieje i jest folderem

krok "1. Normalizacja danych"
# kazdy zbior budowany tylko raz - build_info.json powstaje na koncu budowania
if [ ! -f $L/build_info.json ]; then
    rm -rf $L
    python -m normalization.build --weedsgalore "$WG" -o $L --tryb-klas chwasty > /dev/null
fi
if [ ! -f $N/build_info.json ]; then
    rm -rf $N
    python -m normalization.build --sudars "$SU" -o $N --tryb-klas chwasty > /dev/null
fi
if [ ! -f $Z/build_info.json ]; then
    rm -rf $Z
    # skala x0.47: 1,18 -> 2,5 mm/px jak WeedsGalore; dzien testowy -> test, reszta -> train
    python -m normalization.build --ewis1 "$E" -o $Z --tryb-klas chwasty --skala ewis1=0.47 \
           --proporcje 1 0 0 --ewis1-test-dni $TEST_DNI > /dev/null
fi
# kontrola: brak nierozpoznanych etykiet + liczba kafelkow w kazdej czesci
python - $L $N $Z <<'PY'
import json, sys
from pathlib import Path
for z in sys.argv[1:]:
    i = json.loads(Path(z, "build_info.json").read_text())
    assert not i.get("nierozpoznane_etykiety"), (z, i["nierozpoznane_etykiety"])
    n = {c: len(list(Path(z, "dataset", "images", c).glob("*.jpg"))) for c in ("train", "val", "test")}
    print(f"  {z:<20} train {n['train']:>5}   val {n['val']:>4}   test {n['test']:>4}")
PY

krok "2. Trening"
if [ -f modele_dane_ewis1_bt/model.pt ]; then
    echo "  modele_dane_ewis1_bt/model.pt jest - pomijam"
else
    # kopie kafelkow wczesnych dni (male chwasty) + Sudars + EWIS1 trening -> data.yaml treningu
    python -m models.cli nadprobkuj $L/dataset --waga 2023-05-25=3 --waga 2023-05-30=2 \
           --dodaj-trening $N/dataset --dodaj-trening $Z/dataset \
           -o $L/dataset/data_dane_ewis1_bt.yaml > /dev/null
    # plan: jeden etap od yolov8n.pt, early stopping po 30 epokach bez poprawy
    cat > plan_dane_ewis1_bt.json <<'PLAN'
{"model_bazowy": "yolov8n.pt", "batch": 16, "cierpliwosc": 30, "seed": 0,
 "etapy": [{"nazwa": "dane-ewis1-bt", "dane": "zbior_lotnicze/dataset/data_dane_ewis1_bt.yaml",
            "epoki": 100, "lr0": 0.01, "wejscie": 1280}]}
PLAN
    echo "  trening w toku (ok. 4 h), postep: tail -f log_dane_ewis1_bt.txt"
    python -m models.cli trenuj --plan plan_dane_ewis1_bt.json -o modele_dane_ewis1_bt > log_dane_ewis1_bt.txt 2>&1
fi
mkdir -p models/weights
cmp -s modele_dane_ewis1_bt/model.pt "$M" || cp modele_dane_ewis1_bt/model.pt "$M"   # kopia tylko gdy inny

krok "3. Ocena: EWIS1 (dzien $TEST_DNI) i WeedsGalore, bez i z TTA"
for T in "" "--tta"; do
    S=$([ -n "$T" ] && echo "_tta" || echo "")      # koncowka nazwy pliku: "" albo "_tta"
    python -m models.cli ocen-detektor "$M" $Z/dataset/data.yaml --prog 0.15 $T -o "$W/ewis1$S.json" > /dev/null
    python -m models.cli ocen-detektor "$M" $L/dataset/data.yaml --prog 0.15 $T -o "$W/wg$S.json" > /dev/null
done

krok "4. Paczka ONNX -> hailo/hailo_final.zip"
if [ -f hailo/hailo_final.zip ] && [ hailo/hailo_final.zip -nt "$M" ]; then
    echo "  paczka nowsza niz model - pomijam"
else
    rm -rf hailo/final hailo/hailo_final.zip && mkdir -p hailo/final/_kal
    python -m models.cli onnx "$M" -o hailo/final/fieldeye_final_1280.onnx --wejscie 1280 --opset 11 > /dev/null
    # zdjecia kalibracyjne tylko z treningu (WeedsGalore + EWIS1), nigdy z testu
    cp $L/dataset/images/train/*.jpg $Z/dataset/images/train/*.jpg hailo/final/_kal/
    python -m models.cli kalibracja hailo/final/_kal -o hailo/final/kalibracja_1280 --liczba 128 --rozmiar 1280 > /dev/null
    rm -rf hailo/final/_kal
    cp scripts/compile_hef.py hailo/final/
    (cd hailo/final && zip -qr ../hailo_final.zip .)
fi

krok "5. Wyniki"
python - "$W" "$TEST_DNI" <<'PY'
import json, sys
from pathlib import Path
W = Path(sys.argv[1])
def wiersz(d):
    p = d["przy_progu"]
    return f"{d['AP50']:>7.3f}{p['czulosc']:>9.3f}{p['precyzja']:>8.3f}"
print(f"\n  {'test':<22}| {'bez TTA':^24} | {'z TTA':^24}")
print(f"  {'':<22}| {'AP50':>7}{'czułość':>9}{'prec.':>8} | {'AP50':>7}{'czułość':>9}{'prec.':>8}")
for nazwa, k in ((f"EWIS1 {sys.argv[2]}", "ewis1"), ("WeedsGalore", "wg")):
    a, b = (json.loads((W / f"{k}{s}.json").read_text()) for s in ("", "_tta"))
    print(f"  {nazwa:<22}| {wiersz(a)} | {wiersz(b)}")
print("\n  punkt odniesienia, AP50 z TTA: EWIS1 0.817, WeedsGalore 0.714")
PY

echo
echo "Gotowe. Aplikacja: streamlit run app.py"
