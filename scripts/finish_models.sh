#!/usr/bin/env bash
# Dokonczenie modeli FIELDEYE na Macu, jednym poleceniem:
#   1. trening modelu B (YOLOv8n, wejscie 640, zgodny z gotowa konfiguracja Hailo),
#   2. ocena obu modeli (.pt): mAP50, wynik wedlug dni i rozmiaru, AP50 naszym kodem,
#   3. eksport obu do ONNX pod Hailo (opset 11) i sprawdzenie, czy eksport nic nie zmienil,
#   4. zdjecia kalibracyjne w rozmiarze wejscia kazdego modelu,
#   5. paczka hailo/ do skopiowania na Linuksa (kompilacja) i na Raspberry Pi (sprawdzenie),
#   6. tabela z wynikami.
#
# Uruchomienie z katalogu stitch-live, z aktywnym .venv (ok. 1-1,5 h, glownie trening):
#   caffeinate -i bash scripts/finish_models.sh
# Kroki, ktore juz sa zrobione (np. wytrenowany model B), sa pomijane.
set -euo pipefail
cd "$(dirname "$0")/.."

DANE=zbior_lotnicze/dataset/data.yaml
DANE_S=zbior_lotnicze/dataset/data_z_sudarsem.yaml
mkdir -p wyniki hailo
krok() { echo; echo "=== $*"; }

krok "0. Kontrola"
[ -f modele_wg_sudars/model.pt ] || { echo "brak modele_wg_sudars/model.pt (model A)"; exit 1; }
[ -f "$DANE_S" ] || { echo "brak $DANE_S - utworz go wedlug pipeline'u (krok 4)"; exit 1; }
python -m models.cli sprawdz "$DANE_S"

krok "1. Trening modelu B (wejscie 640)"
if [ -f modele_wg_sudars_640/model.pt ]; then
    echo "  modele_wg_sudars_640/model.pt juz jest - pomijam"
else
    cat > plan_wg_sudars_640.json <<'PLAN'
{"model_bazowy": "yolov8n.pt", "batch": 16, "cierpliwosc": 30,
 "etapy": [{"nazwa": "wg-sudars-640", "dane": "zbior_lotnicze/dataset/data_z_sudarsem.yaml",
            "epoki": 100, "lr0": 0.01, "wejscie": 640}]}
PLAN
    echo "  trening w toku, postep: grep -a \"/100\" log_wg_sudars_640.txt | tail -n 1"
    python -m models.cli trenuj --plan plan_wg_sudars_640.json -o modele_wg_sudars_640 > log_wg_sudars_640.txt 2>&1
fi

krok "2. Ocena obu modeli (.pt)"
for M in "wg_sudars 1280" "wg_sudars_640 640"; do
    set -- $M
    python -m models.cli ocen modele_$1/model.pt "$DANE" --wejscie $2 --wedlug-dni --male-obiekty \
           --prog 0.15 -o wyniki/$1_dni.json > /dev/null
    python -m models.cli ocen-detektor modele_$1/model.pt "$DANE" --wejscie $2 --prog 0.15 \
           -o wyniki/$1_pt_det.json
done

krok "3. Eksport ONNX pod Hailo i kontrola"
python -m models.cli onnx modele_wg_sudars_640/model.pt -o hailo/wg_sudars_640.onnx  --wejscie 640  --opset 11
python -m models.cli onnx modele_wg_sudars/model.pt     -o hailo/wg_sudars_1280.onnx --wejscie 1280 --opset 11
# ten sam kod oceny co dla .pt; ONNX liczony na procesorze
python -m models.cli ocen-detektor hailo/wg_sudars_640.onnx  "$DANE" --wejscie 640  --urzadzenie cpu \
       --prog 0.15 -o wyniki/wg_sudars_640_onnx_det.json
python -m models.cli ocen-detektor hailo/wg_sudars_1280.onnx "$DANE" --wejscie 1280 --urzadzenie cpu \
       --prog 0.15 -o wyniki/wg_sudars_onnx_det.json

krok "4. Zdjecia kalibracyjne (tylko kafelki treningowe)"
python -m models.cli kalibracja zbior_lotnicze/dataset/images/train -o hailo/kalibracja_640  --liczba 104 --rozmiar 640
python -m models.cli kalibracja zbior_lotnicze/dataset/images/train -o hailo/kalibracja_1280 --liczba 104 --rozmiar 1280

krok "5. Paczka hailo/"
# zbior testowy z data.yaml o sciezce wzglednej - dziala po skopiowaniu na Raspberry Pi
rm -rf hailo/test && mkdir -p hailo/test/images hailo/test/labels
cp -R zbior_lotnicze/dataset/images/test hailo/test/images/
cp -R zbior_lotnicze/dataset/labels/test hailo/test/labels/
printf "path: .\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n  0: chwast\n" > hailo/test/data.yaml
mkdir -p hailo/wyniki_mac && cp wyniki/*_det.json hailo/wyniki_mac/
cp scripts/HAILO.md hailo/
du -sh hailo | sed 's/^/  rozmiar paczki: /'

krok "6. Wyniki"
python - <<'PY'
import json
from pathlib import Path

def czytaj(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None

def f(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

print(f"\n  {'model':<24}{'mAP50':>7}{'AP50 .pt':>10}{'AP50 .onnx':>12}{'czułość':>9}{'precyzja':>10}{'25 maja':>9}")
uwagi = []
for opis, k in (("A: wg_sudars (1280)", "wg_sudars"), ("B: wg_sudars_640 (640)", "wg_sudars_640")):
    d, pt, ox = czytaj(f"wyniki/{k}_dni.json"), czytaj(f"wyniki/{k}_pt_det.json"), czytaj(f"wyniki/{k}_onnx_det.json")
    r = (pt or {}).get("przy_progu", {})
    maj = (pt or {}).get("wedlug_dni", {}).get("2023-05-25", {}).get("czulosc")
    print(f"  {opis:<24}{f((d or {}).get('mAP50')):>7}{f((pt or {}).get('AP50')):>10}"
          f"{f((ox or {}).get('AP50')):>12}{f(r.get('czulosc')):>9}{f(r.get('precyzja')):>10}{f(maj):>9}")
    if pt and ox and abs(pt["AP50"] - ox["AP50"]) > 0.01:
        uwagi.append(f"  UWAGA {opis}: ONNX różni się od .pt o {ox['AP50'] - pt['AP50']:+.3f} AP50 - sprawdź eksport")
print("\n".join(uwagi) or "\n  ONNX zgodny z .pt dla obu modeli.")
a, b = czytaj("wyniki/wg_sudars_pt_det.json"), czytaj("wyniki/wg_sudars_640_pt_det.json")
if a and b:
    roz = b["AP50"] - a["AP50"]
    print(f"  Model 640 względem 1280: {roz:+.3f} AP50. Koszt na układzie: 8,6 zamiast 34,6 GFLOPs na zdjęcie.")
PY

echo
echo "Dalej: skopiuj katalog hailo/ na komputer z Linuksem i dockerem Hailo"
echo "i skompiluj modele wedlug hailo/HAILO.md (krok 2), a potem na Raspberry Pi:"
echo "  bash scripts/check_hef.sh"
