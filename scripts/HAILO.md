# Modele FIELDEYE na Hailo-8 (Raspberry Pi 5 + AI HAT+ 26 TOPS)

Dwa modele, ta sama architektura YOLOv8n, jedna klasa „chwast”:

| | Model A | Model B |
|---|---|---|
| katalog | `modele_wg_sudars` | `modele_wg_sudars_640` |
| wejście | 1280 | 640 |
| kompilacja | własna konfiguracja NMS dla 1280 | konfiguracja `yolov8n` z Model Zoo |

Droga: `model.pt` → `model.onnx` (Mac) → `model.hef` (Linux x86) → uruchomienie (Raspberry Pi).
HEF nie zastępuje `.pt`: trening, ocena i interfejs na Macu dalej używają `.pt`.

---

## 1. Mac: ONNX i zdjęcia kalibracyjne

**Najprościej jednym skryptem**, który robi wszystko z tego kroku, a do tego
trenuje model B, ocenia oba modele, sprawdza ONNX i pakuje zbiór testowy:

```bash
caffeinate -i bash scripts/finish_models.sh
```

Ręcznie, krok po kroku:

```bash
mkdir -p hailo
python -m models.cli onnx modele_wg_sudars_640/model.pt -o hailo/wg_sudars_640.onnx  --wejscie 640  --opset 11
python -m models.cli onnx modele_wg_sudars/model.pt     -o hailo/wg_sudars_1280.onnx --wejscie 1280 --opset 11

python -m models.cli kalibracja zbior_lotnicze/dataset/images/train -o hailo/kalibracja_640  --liczba 104 --rozmiar 640
python -m models.cli kalibracja zbior_lotnicze/dataset/images/train -o hailo/kalibracja_1280 --liczba 104 --rozmiar 1280
```

- opset 11, stałe wejście i uproszczony graf to ustawienia zalecane dla kompilatora Hailo,
- `--rozmiar` wpisuje zdjęcia w kwadrat wejścia sieci tak samo, jak robi to `HailoDetector`,
- zdjęcia kalibracyjne to tylko kafelki **treningowe** WeedsGalore (104), nigdy test.
  Na wiosnę trzeba je zastąpić zdjęciami z własnego pola i własnej kamery.

Katalog `hailo/` skopiuj na komputer z Linuksem.

## 2. Linux x86: kompilacja do HEF

Potrzebny jest **Hailo AI Software Suite** (docker) w wersji dla **Hailo-8/8L**
(Dataflow Compiler 3.x + Model Zoo 2.x), z hailo.ai → Developer Zone. Wersja dla
Hailo-10/15 nie przyjmie `--hw-arch hailo8`. Kompilator nie działa na Macu.

### 2a. Konfiguracja NMS

Gotowa konfiguracja zakłada wejście 640 i najwyżej 100 wykryć na klasę. Na naszych
kafelkach bywa ponad 100 chwastów, więc limit trzeba podnieść w **obu** modelach,
a dla modelu A zmienić też rozmiar obrazu.

```bash
# w kontenerze; ścieżki mogą się różnić między wersjami Model Zoo
find / -name "yolov8n_nms_config.json" 2>/dev/null
find / -name "yolov8n.alls" 2>/dev/null

cp <...>/yolov8n_nms_config.json nms_640.json
cp <...>/yolov8n_nms_config.json nms_1280.json
# w obu plikach:   "max_proposals_per_class": 300
# w nms_1280.json:  "image_dims": [1280, 1280]

cp <...>/yolov8n.alls yolov8n_640.alls
cp <...>/yolov8n.alls yolov8n_1280.alls
# w każdym .alls w linii nms_postprocess(...) podmień ścieżkę na nms_640.json / nms_1280.json
```

### 2b. Kompilacja

```bash
hailomz compile yolov8n --ckpt hailo/wg_sudars_640.onnx  --hw-arch hailo8 \
        --calib-path hailo/kalibracja_640  --classes 1 --model-script yolov8n_640.alls
hailomz compile yolov8n --ckpt hailo/wg_sudars_1280.onnx --hw-arch hailo8 \
        --calib-path hailo/kalibracja_1280 --classes 1 --model-script yolov8n_1280.alls
```

Wynik: `yolov8n.hef` po każdej kompilacji. Przemianuj od razu na
`wg_sudars_640.hef` i `wg_sudars_1280.hef`, żeby się nie pomyliły.

**Jeśli `parse` zgłosi błąd węzłów:** Ultralytics 8.4 może nazywać warstwy inaczej,
niż spodziewa się konfiguracja. Otwórz ONNX w https://netron.app, znajdź sześć
ostatnich splotów przed łączeniem wyjść (po dwa na każdą z trzech skal) i podaj je
w `--end-node-names`.

## 3. Raspberry Pi 5 z AI HAT+: sprawdzenie

**Najprościej jednym skryptem.** Domyślnie sprawdza model końcowy
`fieldeye_final_1280_h8.hef`: sam wypakowuje go z `model_fin.zip` do `hailo/`, pokazuje
`parse-hef` i przepustowość, a gdy jest zbiór testowy EWIS1 (`zbior_ewis1_podzial/`),
liczy AP50 bez TTA i z TTA i porównuje z wynikami `.pt` z dokumentacji modelu:

```bash
bash scripts/check_hef.sh
bash scripts/check_hef.sh hailo/inny_model.hef zbior_lotnicze/dataset/data.yaml   # inny model lub zbiór
```

Ręcznie:

```bash
sudo apt install hailo-all          # HailoRT i pakiet hailo_platform
hailortcli fw-control identify      # czy akcelerator jest widoczny
hailortcli parse-hef wg_sudars_640.hef
```

`parse-hef` powinien pokazać wejście 640×640×3 (albo 1280×1280×3) i wyjście
z NMS (`yolov8_nms_postprocess`).

### Ocena tym samym kodem co na Macu

Na Raspberry Pi potrzebny jest katalog `stitch-live` (bez ultralytics) i zbiór
testowy `zbior_lotnicze/dataset` (26 kafelków).

```bash
# Raspberry Pi
python -m models.cli ocen-detektor wg_sudars_640.hef  zbior_lotnicze/dataset/data.yaml --prog 0.15 -o hef_640.json
python -m models.cli ocen-detektor wg_sudars_1280.hef zbior_lotnicze/dataset/data.yaml --prog 0.15 -o hef_1280.json

# Mac - te same liczby dla oryginałów
python -m models.cli ocen-detektor modele_wg_sudars_640/model.pt zbior_lotnicze/dataset/data.yaml --wejscie 640  --prog 0.15
python -m models.cli ocen-detektor modele_wg_sudars/model.pt     zbior_lotnicze/dataset/data.yaml --wejscie 1280 --prog 0.15
```

Różnica AP50 i czułości między `.pt` a `.hef` to strata z kwantyzacji i kompilacji.
Kilka punktów procentowych jest normalne. Gdy HEF prawie nic nie wykrywa, a `.pt`
wykrywa, sprawdź po kolei: liczbę klas (`--classes 1`), ścieżkę NMS w `.alls`,
format zdjęć kalibracyjnych.

## 4. Użycie w misji

Gotowa konfiguracja jest w `configs/mission_hailo.yaml`:

```bash
python3 -m mission.cli pelna zdjecia/ -o wynik_misji/ --config configs/mission_hailo.yaml
```

Przy `.hef` wejście sieci jest wbudowane w plik, więc `wejscie_sieci` nie ma znaczenia.
