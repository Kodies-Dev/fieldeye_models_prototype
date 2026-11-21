# mission — detekcja chwastów w locie i mapa oprysku

Na dronie działa jeden program: **`fieldeye-detect`** (binarka na Raspberry Pi 5 + Hailo-8).
Obserwuje folder, do którego dron zapisuje zdjęcia, wykrywa chwasty na każdym nowym zdjęciu
i — z telemetrii GeoJSON leżącej obok zdjęcia — zamienia wykrycia na współrzędne i punkty oprysku.
Pełny opis (uruchamianie, format telemetrii, wyniki, binarka, Raspberry Pi):
[`docs/fieldeye-detect.md`](../docs/fieldeye-detect.md).

```bash
python -m mission.detect_cli ZDJECIA WYNIKI models/weights/fieldeye_final.pt    # ze źródeł
./fieldeye-detect ZDJECIA WYNIKI fieldeye_final_1280_h8.hef                     # binarka na Pi
python -m mission.drone_sim lot/ --ewis1 zbior_ewis1_podzial/dataset/images/test # symulowany dron
```

---

## Mapowanie nie wymaga mozaiki

Żeby nanieść chwast na mapę, **nie trzeba składać zdjęć**. Wystarczy pozycja GPS, kurs
i wysokość tego jednego zdjęcia:

```
piksel w kadrze
   → przesunięcie od środka kadru w pikselach
   → × GSD (metry na piksel, z wysokości i optyki)
   → obrót o kurs drona
   → dodanie do pozycji GPS
   → współrzędne geograficzne
```

Dzięki temu każde zdjęcie jest przetwarzane osobno, zaraz po zrobieniu, ze stałą zajętością
pamięci, a wyniki są dopisywane na bieżąco (awaria zasilania nie kasuje misji).

---

## Detektor

Wymienny — reszta programu nie wie, jakiego modelu używa. Wszystkie klasy są w `detector.py`
i zwracają te same obiekty `Detection`.

| Klasa | Do czego |
|---|---|
| `YoloDetector` | model `.pt` przez Ultralytics (komputer) |
| `OnnxDetector` | model `.onnx` przez ONNX Runtime — bez PyTorcha (lekka binarka) |
| `HailoDetector` | model `.hef` na akceleratorze Hailo-8; sam robi *letterbox* i przelicza ramki z NMS Hailo |
| `TiledDetector` | opakowanie: tnie zdjęcie na kafelki, scala wyniki, usuwa duplikaty z zakładki |
| `FlipTTADetector` | opakowanie: oryginał + odbicie poziome, łączone przez WBF (TTA) |
| `DummyDetector` | **nie jest detektorem chwastów** — plamy zieleni, tylko do sprawdzenia potoku bez modelu (klasa `atrapa`) |

`zbuduj_z_plikow(model, tta=...)` wybiera klasę po rozszerzeniu pliku (`.hef` → Hailo,
`.onnx` → ONNX Runtime, reszta → Ultralytics). `iou_ramek()` to jedyne liczenie IoU
w projekcie (używa go też `models/evaluation.py`).

**Po co kafelkowanie.** Zdjęcie z kamery ma kilka tysięcy pikseli, wejście sieci 1280 px.
Zmniejszenie całego zdjęcia sprawia, że kilkunastopikselowy chwast znika. Kafelek dobiera się
tak, żeby obejmował ok. **1,5 m pola** — na takiej skali uczył się model (Arducam 64MP,
tryb 16 MP, 5 m: ok. 950 px; `stitchlive/camera.py`).

---

## Symulowany lot w interfejsie

`sim_flight.py` — funkcje zakładki 4 w `app.py`: pole z kafelków jednego dnia WeedsGalore
(`pole_z_kafelkow`), prawdziwe ramki w kadrze, scalanie powtórzeń na mozaice i porównanie
z prawdą. Zdjęcie EWIS1 złożone z powrotem z kafelków co do piksela (`zdjecia_z_kafelkow`,
`pole_ze_zdjecia`) wykorzystuje symulowany dron (`drone_sim --ewis1`).

---

## Ograniczenia

- **Zakłada nadir i płaski teren.** Przechył z autopilota służy tylko do odrzucania zdjęć ponad limit.
- **Dokładność ograniczona przez GPS**, nie przez obliczenia. Bez korekcji ok. 1–2 m; RTK dałby centymetry.
- **Wysokość musi być nad terenem**, nie nad poziomem morza.
- **Brak korekty zniekształceń obiektywu.** Przy szerokim kącie błąd rośnie ku brzegom kadru.

---

## Struktura

```
mission/
├── detect_cli.py     fieldeye-detect: nowe zdjęcia → detekcja → mapa oprysku
├── spray_map.py      telemetria GeoJSON → współrzędne → punkty oprysku (bez pyproj)
├── drone_sim.py      symulowany dron: zdjęcia + telemetria jak z prawdziwego drona
├── detector.py       .pt / .onnx / .hef, kafelki, TTA, atrapa, iou_ramek()
└── sim_flight.py     symulowany lot w zakładce 4 app.py
```

Testy: `tests/test_detect_cli.py`, `test_spray_map.py`, `test_binary.py`, `test_hailo.py`,
`test_inference.py`, `test_sim_tiles.py`, `test_gui_flight.py`.
