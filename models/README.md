# models — trening, ocena i eksport detektora chwastów

Bierze zbiory z `normalization/`, trenuje YOLOv8n z jedną klasą „chwast”,
ocenia go (także osobno dla każdego dnia lotu) i eksportuje do ONNX, z którego
na końcu powstaje plik `.hef` dla Hailo-8 (`scripts/HAILO.md`).

---

## Model docelowy (stan na październik 2026)

| | |
|---|---|
| architektura | YOLOv8n, jedna klasa „chwast”, wejście 1280 |
| dane treningowe | WeedsGalore + Sudars |
| przepis | **nadpróbkowanie wczesnej fazy**: kafelki 25 maja ×3, 30 maja ×2 |
| trening | 100 epok, SGD, lr0 0,01, batch 16, cierpliwość 30 |
| uruchamianie | **TTA**: oryginał + odbicie poziome, łączone przez WBF; próg 0,15 we wszystkich dniach |
| plik | `modele_sezon_nadprob/model.pt` |

Wynik na teście WeedsGalore (26 kafelków, 1628 chwastów), dwa ziarna losowe:

| | punkt wyjścia (`wg_sudars`, bez TTA) | model docelowy |
|---|---|---|
| AP50 | 0,673 | **0,713–0,718** |
| czułość przy 0,15 | 67% | **68–69%** |
| precyzja przy 0,15 | 58% | **65–68%** |
| fałszywe alarmy | 779 | **ok. 540–600** |
| czułość 25 maja | 27% | **ok. 30%** |

AP50 w tej tabeli liczy nasza funkcja `ap50()` (tak samo dla `.pt`, `.onnx`
i `.hef`); mAP50 z walidacji Ultralytics wychodzi o ok. 1,5 pp niżej.

**Zastrzeżenie:** test pochodzi z tego samego pola co trening. Drugi test na
innym polu i kamerą RGB: `scripts/ewis1_test.sh` (odłożony dzień lotu EWIS1).

---

## Historia eksperymentów

Wszystko na teście WeedsGalore, YOLOv8n, wejście 1280, 100 epok, o ile nie podano inaczej.

| Wariant | Wynik | Wniosek |
|---|---|---|
| 30 epok (Windows) | mAP50 0,567 | za krótko |
| `wg100`: tylko WeedsGalore | mAP50 0,651; 25 maja 17–21% | punkt odniesienia |
| YOLOv8s | mAP50 0,645 | +3,5 × obliczeń bez zysku: odrzucony |
| Sudars, potem WeedsGalore (dwa etapy) | mAP50 0,572 / 0,624 | trening po kolei gorszy niż razem |
| `wg_sudars`: WeedsGalore + Sudars razem | mAP50 0,659; 25 maja 27–28% | Sudars daje ok. +8 pp w maju |
| test na całym nieznanym dniu (25 maja) | czułość 23–26% we wszystkich wariantach | model nie przenosi się na nieznaną, najwcześniejszą fazę |
| **nadpróbkowanie maja** | AP50 +1,3–1,7 pp, maj +0,6–3,6 pp (2 ziarna) | **przyjęte** |
| mniejszy `scale` w augmentacji | więcej fałszywych alarmów | odrzucone |
| osobny model na maj (dostrojenie) | bez poprawy | odrzucone: jeden wspólny model |
| próg osobno dla każdego dnia | maj 27 → 40%, ale fałszywe 111 → 499 | odrzucone: zostaje 0,15 |
| **TTA** | AP50 +2,2–2,8 pp, fałszywe ok. −30% (2 ziarna) | **przyjęte** |
| dwa modele naraz (ensemble) | jak TTA, ale 2 pliki HEF | odrzucone |
| łączenie widoków, głosowanie, niski próg + głosowanie | brak zysku netto | odrzucone |
| EWIS1, PhenoBench w treningu | w toku | `scripts/new_data_experiment.sh` |

Różnice do ok. 2 pp AP50 i ok. 4 pp czułości w maju (332 siewki) mieszczą się
w zmienności między uruchomieniami, dlatego każdą poprawę potwierdzamy drugim ziarnem.

---

## Zanim zaczniesz trenować

```bash
python -m models.cli srodowisko                         # biblioteki i urządzenie (CUDA / MPS / CPU)
python -m models.cli sprawdz zbior_lotnicze/dataset/data.yaml
python -m models.cli analiza zbior_lotnicze/dataset/data.yaml -o analiza_lotnicze
```

- `sprawdz`: katalogi, pliki etykiet, przeciek plików między częściami podziału,
- `analiza`: liczba obrazów i obiektów, wielkość obiektów i obiekty na obraz,
  osobno dla części i dni lotu, oraz wykresy (`raport.md`, `*.png`).

---

## Trening

Trening opisuje plan JSON. Wszystkie zbiory w jednym etapie, razem:

```json
{"model_bazowy": "yolov8n.pt", "batch": 16, "cierpliwosc": 30, "seed": 0,
 "etapy": [{"nazwa": "wg-sudars", "dane": "zbior_lotnicze/dataset/data_z_sudarsem.yaml",
            "epoki": 100, "lr0": 0.01, "wejscie": 1280}]}
```

```bash
caffeinate -i python -m models.cli trenuj --plan plan_wg_sudars.json -o modele_wg_sudars
```

| Pole planu | Znaczenie |
|---|---|
| `model_bazowy` | wagi startowe (COCO albo inny model) |
| `batch`, `cierpliwosc`, `seed` | porcja zdjęć, ile epok bez poprawy do zatrzymania, ziarno losowe |
| `etapy[].dane` | `data.yaml`; `train` może być listą katalogów (kilka zbiorów naraz) |
| `etapy[].augmentacje` | dodatkowe ustawienia augmentacji Ultralytics, np. `{"scale": 0.2}` |

Optymalizator SGD jest ustawiony jawnie, bo przy `auto` Ultralytics ignoruje `lr0`.
Walidacja i test zawsze z WeedsGalore; dodatkowe zbiory tylko do treningu.

### Nadpróbkowanie wczesnej fazy

```bash
python -m models.cli nadprobkuj zbior_lotnicze/dataset --waga 2023-05-25=3 --waga 2023-05-30=2 \
       --dodaj-trening zbior_naziemne/dataset -o zbior_lotnicze/dataset/data_sezon.yaml
```

Kopie kafelków treningowych trafiają do `images/train_nadprob` z inną nazwą;
walidacja i test bez zmian.

---

## Ocena

| Komenda | Co liczy |
|---|---|
| `ocen model.pt data.yaml --wejscie 1280 --wedlug-dni --male-obiekty --prog 0.15` | mAP50 Ultralytics, wynik według dni i rozmiaru |
| `ocen-detektor model.pt [model2.pt] data.yaml --prog 0.15 [--tta]` | AP50 i wynik według dni **naszym kodem** – ten sam dla `.pt`, `.onnx`, `.hef`; kilka modeli = ensemble |
| `ocen-sezon model.pt data.yaml --progi progi.json` | test według dni przy jednym progu i przy progu każdego dnia |
| `prog model.pt data.yaml` / `prog-dni ...` | dobór progu na walidacji (F-beta, czułość 3 × ważniejsza) |

Daty w nazwach plików są rozpoznawane w formacie `2023-05-25` (WeedsGalore)
i `20220509` (EWIS1).

---

## Eksport i Hailo

```bash
python -m models.cli onnx modele_X/model.pt -o hailo/model_640.onnx --wejscie 640 --opset 11
python -m models.cli kalibracja zbior_lotnicze/dataset/images/train -o hailo/kalibracja_640 --liczba 104 --rozmiar 640
python -m models.cli ocen-detektor hailo/model_640.onnx zbior_lotnicze/dataset/data.yaml --wejscie 640 --urzadzenie cpu
```

- stałe wejście i opset 11 to ustawienia zalecane przez kompilator Hailo,
- zdjęcia kalibracyjne: tylko kafelki treningowe, wpisane w kwadrat wejścia sieci,
- kompilacja do `.hef` (Linux x86, WSL) i sprawdzenie na Raspberry Pi: `scripts/HAILO.md`.

---

## Struktura

```
models/
├── plan.py         plan treningu, kontrola data.yaml          (bez ultralytics)
├── device.py       wybór CUDA / MPS / CPU
├── training.py     trening etapowy, ablacja
├── evaluation.py   mAP, AP50 naszym kodem, wynik według dni i rozmiaru, dobór progu
├── season.py       nadpróbkowanie, zbiór z wybranych dni, progi według dni
├── analysis.py     statystyki zbioru i wykresy
├── splits.py       podział całymi dniami (test na nieznanym dniu)
├── export.py       ONNX, zbiór kalibracyjny
└── cli.py          wiersz poleceń
```