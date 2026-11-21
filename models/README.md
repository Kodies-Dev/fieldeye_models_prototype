# models — trening, ocena i eksport detektora chwastów

Bierze zbiory z `normalization/`, trenuje YOLOv8n z jedną klasą „chwast”,
ocenia go (także osobno dla każdego dnia lotu) i eksportuje do ONNX, z którego
na końcu powstaje plik `.hef` dla Hailo-8 (`scripts/HAILO.md`).

---

## Model końcowy

| | |
|---|---|
| plik | `models/weights/fieldeye_final.pt` (w eksperymentach: `modele_dane_ewis1_bt/model.pt`) |
| architektura | YOLOv8n, jedna klasa „chwast”, wejście 1280 |
| dane treningowe | WeedsGalore (z nadpróbkowaniem: 25 maja ×3, 30 maja ×2) + Sudars + EWIS1 kukurydza bez dnia 16.05.2022 — 1866 kafelków |
| trening | 100 epok, SGD, lr0 0,01, batch 16, cierpliwość 30 |
| uruchamianie | **TTA**: oryginał + odbicie poziome, łączone przez WBF; próg 0,15 |
| odtworzenie | `bash scripts/final_model.sh` |

| Test (z TTA) | AP50 | Czułość | Precyzja |
|---|---|---|---|
| EWIS1, 16.05.2022 — zwykła kamera RGB, dzień niewidziany w treningu (132 kafelki, 4422 chwasty) | **0,817** | **78,4%** | **84,3%** |
| WeedsGalore, cały sezon (26 kafelków, 1628 chwastów) | 0,714 | 69,2% | — |
| WeedsGalore, 25 maja (najmłodsze siewki) | — | 32,2% | — |

AP50 liczy nasza funkcja `ap50()` (tak samo dla `.pt`, `.onnx` i `.hef`); mAP50 z walidacji
Ultralytics wychodzi o ok. 1,5 pp niżej.

**Najważniejszy wniosek:** model uczony bez EWIS1 (WeedsGalore + Sudars) miał na WeedsGalore
AP50 0,718, ale na zwykłym zdjęciu RGB tylko 0,044 (czułość 0,2%) — nauczył się wyglądu zdjęć
z kamery multispektralnej. Dodanie EWIS1 do treningu rozwiązało problem bez straty na WeedsGalore.

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
| test na zwykłej kamerze RGB (EWIS1) | model bez EWIS1: AP50 0,044, czułość 0,2% | model nie przenosił się na zdjęcia RGB |
| EWIS1 przerobiony na kolory WeedsGalore | AP50 0,224 | przyczyna głównie w kolorach kamery multispektralnej |
| **+ EWIS1 w treningu** | **EWIS1 AP50 0,817**, WeedsGalore bez straty | **przyjęte: model końcowy** |
| korekta kolorów WeedsGalore (sRGB) | EWIS1 +2 pp, 25 maja −5,7 pp | odrzucone |
| wejście 640 zamiast 1280 | EWIS1 −3,7 pp, 25 maja −10,5 pp | zostaje 1280 |
| + PhenoBench (przy 640) | EWIS1 +2,2 pp | niewielki zysk, poza modelem końcowym |
| + sorgo z EWIS1 | bez zmian | odrzucone |

Różnice do ok. 2 pp AP50 i ok. 4 pp czułości w maju (332 siewki) mieszczą się
w zmienności między uruchomieniami, dlatego każdą poprawę potwierdzamy drugim ziarnem.

---

## Zanim zaczniesz trenować

```bash
python -m models.cli srodowisko                         # biblioteki i urządzenie (CUDA / MPS / CPU)
python -m models.cli sprawdz zbior_lotnicze/dataset/data.yaml
```

- `sprawdz`: katalogi, pliki etykiet, przeciek plików między częściami podziału.

---

## Trening

Trening opisuje plan JSON. Wszystkie zbiory w jednym etapie, razem (plan modelu końcowego,
zapisywany przez `scripts/final_model.sh`):

```json
{"model_bazowy": "yolov8n.pt", "batch": 16, "cierpliwosc": 30, "seed": 0,
 "etapy": [{"nazwa": "dane-ewis1-bt", "dane": "zbior_lotnicze/dataset/data_dane_ewis1_bt.yaml",
            "epoki": 100, "lr0": 0.01, "wejscie": 1280}]}
```

```bash
caffeinate -i python -m models.cli trenuj --plan plan_dane_ewis1_bt.json -o modele_dane_ewis1_bt
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
       --dodaj-trening zbior_naziemne/dataset --dodaj-trening zbior_ewis1_podzial/dataset \
       -o zbior_lotnicze/dataset/data_dane_ewis1_bt.yaml
```

Kopie kafelków treningowych trafiają do `images/train_nadprob` z inną nazwą;
walidacja i test bez zmian.

---

## Ocena

```bash
python -m models.cli ocen-detektor models/weights/fieldeye_final.pt zbior_ewis1_podzial/dataset/data.yaml --prog 0.15 --tta
```

AP50, czułość i precyzja przy progu oraz wynik według dni **naszym kodem** – ten sam
dla `.pt`, `.onnx` i `.hef`, więc wyniki z Maca i z Raspberry Pi są porównywalne.

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
├── training.py     trening etap po etapie
├── evaluation.py   AP50 naszym kodem, czułość i precyzja przy progu, wynik według dni
├── season.py       nadpróbkowanie wczesnych dni
├── export.py       ONNX, zbiór kalibracyjny
└── cli.py          wiersz poleceń
```