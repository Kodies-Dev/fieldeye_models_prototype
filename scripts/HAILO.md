# Model końcowy FIELDEYE na Hailo-8 (Raspberry Pi 5 + AI HAT+ 26 TOPS)

| | |
|---|---|
| model | `models/weights/fieldeye_final.pt` (YOLOv8n, 1 klasa „chwast”, wejście 1280) |
| HEF | `fieldeye_final_1280_h8.hef` (5,4 MB, Hailo-8, HailoRT 4.23) — w `model_fin.zip`, nie w gicie |
| droga | `.pt` → ONNX + zdjęcia kalibracyjne (Mac) → HEF (WSL, Ubuntu 22.04) → uruchomienie (Raspberry Pi) |

HEF nie zastępuje `.pt`: trening, ocena i symulator na komputerze dalej używają `.pt`.
Pełny opis z wynikami: dokument „FIELDEYE: model, konwersja do HEF, Raspberry Pi”.

---

## 1. Mac: ONNX i zdjęcia kalibracyjne

```bash
mkdir -p hailo/final
python -m models.cli onnx models/weights/fieldeye_final.pt -o hailo/final/fieldeye_final_1280.onnx --wejscie 1280 --opset 11

rm -rf /tmp/kal_zrodlo && mkdir -p /tmp/kal_zrodlo
cp zbior_lotnicze/dataset/images/train/*.jpg zbior_ewis1_podzial/dataset/images/train/*.jpg /tmp/kal_zrodlo/
python -m models.cli kalibracja /tmp/kal_zrodlo -o hailo/final/kalibracja_1280 --liczba 128 --rozmiar 1280
cp scripts/compile_hef.py hailo/final/
cd hailo/final && zip -r ~/Desktop/hailo_final.zip . && cd ../..
```

- eksport ma wypisać wyjście `(1, 5, 33600)`; stałe wejście i opset 11 zaleca kompilator Hailo,
- zdjęcia kalibracyjne tylko z części treningowej (WeedsGalore + EWIS1), nigdy z testu.

## 2. WSL: kompilacja do HEF

Kompilator Hailo działa tylko na Linuksie x86: Windows + WSL z Ubuntu 22.04, Python 3.10,
co najmniej 16 GB pamięci (RAM + wymiana, `.wslconfig`).

| Składnik | Wersja | Uwagi |
|---|---|---|
| HailoRT na Raspberry Pi | 4.23.0 | kompilator musi być z tego samego wydania |
| Hailo Dataflow Compiler | 3.33.0 | `hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl` |
| Hailo Model Zoo | 2.17.0 | `hailo_model_zoo-2.17.0-py3-none-any.whl` |
| nie brać | 3.34.0 / 5.x | dla HailoRT 4.24 / dla Hailo-10 i Hailo-15 |

```bash
sudo apt install -y python3.10-venv python3-dev python3-tk graphviz graphviz-dev git build-essential
python3 -m venv ~/hailo_venv && source ~/hailo_venv/bin/activate
pip install --upgrade pip wheel
pip install hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl hailo_model_zoo-2.17.0-py3-none-any.whl

# pakiet Model Zoo nie zawiera konfiguracji NMS - dograć z gałęzi v2.17
git clone --depth 1 --branch v2.17 https://github.com/hailo-ai/hailo_model_zoo.git ~/hmz_src
cp -r ~/hmz_src/hailo_model_zoo/cfg/postprocess_config ~/hailo_venv/lib/python3.10/site-packages/hailo_model_zoo/cfg/

# kompilacja (po rozpakowaniu hailo_final.zip do ~/fieldeye/final)
cd ~/fieldeye/final
python compile_hef.py fieldeye_final_1280.onnx kalibracja_1280 -o fieldeye_final_1280_h8.hef --rozmiar 1280 2>&1 | tee kompilacja_final.log
```

**Dlaczego `compile_hef.py`, a nie samo `hailomz compile`:** konfiguracja NMS z Model Zoo ma obraz
640, próg 0,2 i 100 ramek na klasę. Skrypt bierze z Model Zoo skrypt modelu YOLOv8n i wzór NMS,
ustawia obraz 1280, 1 klasę, próg 0,05 i do 300 ramek, sześć końcowych warstw Conv głowicy
(`/model.22/cv2.x`, `cv3.x`) i nazwę sieci `yolov8n` (do niej odwołuje się konfiguracja NMS).

Model końcowy mieści się w jednym kontekście akceleratora (47% obliczeń, 28% pamięci).
Bez karty graficznej i przy mniej niż 1024 zdjęciach kalibracyjnych kompilator schodzi
do poziomu optymalizacji 0 (64 zdjęcia, podstawowa kwantyzacja) — działa, ale może tracić
więcej dokładności niż przy pełnej optymalizacji (karta graficzna w WSL i ok. 1024 zdjęcia).

| Objaw | Rozwiązanie |
|---|---|
| `Temporary failure resolving` | ręczny DNS w `/etc/resolv.conf` (`nameserver 8.8.8.8`) |
| `No such file ... yolov8n_nms_config.json` | dograć `postprocess_config` z gałęzi v2.17 |
| `The layer named yolov8n/conv41 doesn't exist` | nazwa sieci `yolov8n` (jest w `compile_hef.py`) |
| `invalid choice: 'hailo8'` | Model Zoo 2.17 zamiast 5.x |
| `Unsupported hef version` (na Pi) | kompilator z wydania HailoRT na Pi |

## 3. Raspberry Pi 5 z AI HAT+: sprawdzenie

`fieldeye_final_1280_h8.hef` w katalogu repozytorium, opcjonalnie zbiór `zbior_ewis1_podzial`
(z paczki danych) albo paczka testowa `test_ewis1/`:

```bash
bash scripts/check_hef.sh                                                     # domyślnie test_ewis1/data.yaml
bash scripts/check_hef.sh fieldeye_final_1280_h8.hef zbior_ewis1_podzial/dataset/data.yaml
```

Skrypt sprawdza akcelerator (`hailortcli fw-control identify`), plik (`parse-hef`: wejście
1280×1280×3, wyjście z NMS), szybkość (`hailortcli run`) i dokładność na teście EWIS1 bez TTA
i z TTA, z porównaniem z `.pt`:

| | AP50 `.pt` | czułość | precyzja |
|---|---|---|---|
| bez TTA | 0,812 | 79,1% | 81,1% |
| z TTA | 0,817 | 78,4% | 84,3% |

Różnica to strata z kwantyzacji i kompilacji: do ok. 2 pp „dobrze”, do 5 pp „akceptowalnie”,
więcej — sprawdzić kompilację i zdjęcia kalibracyjne. NMS na układzie odrzuca wykrycia poniżej
0,05, więc AP50 HEF-a może wyjść ok. 0,01 niżej tylko z tego powodu.

## 4. Użycie w locie

```bash
./fieldeye-detect ZDJECIA WYNIKI fieldeye_final_1280_h8.hef                    # binarka
python3 -m mission.detect_cli ZDJECIA WYNIKI fieldeye_final_1280_h8.hef        # ze źródeł
```

Domyślnie próg 0,15 i kafelek 950 px (1,5 m pola z 5 m). Przy `.hef` wejście sieci jest
wbudowane w plik. Pełny opis (telemetria, mapa oprysku, paczka na Pi): `docs/fieldeye-detect.md`.
