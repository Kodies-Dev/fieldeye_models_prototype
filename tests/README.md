# tests/ — testy automatyczne

```bash
python -m pytest tests -q                       # wszystkie, ok. 4 min, 253 testy
python -m pytest tests -q --deselect tests/test_gui_flight.py    # bez interfejsu, ok. 2 min
python -m pytest tests/test_hailo.py -v         # jeden plik
```

Uruchamiaj z katalogu głównego projektu. Z zainstalowanym ultralytics jeden test
jest celowo pomijany (sprawdza zachowanie programu bez tej biblioteki).

| Plik | Testów | Co sprawdza |
|---|---|---|
| `test_global_stitcher.py` | 12 | metoda `global`: obrócone kadry, dokładność par, rozmyte kadry, obcy kadr, kotwica przeciw zgnieceniu, kadr → mozaika |
| `test_stitch.py` | 15 | metody `basic` i `detail`, okno przesuwne, CLI |
| `test_flight_sim.py` | 17 | symulator lotu i ocena mozaiki: wycinek zgodny ze źródłem, marginesy, powtarzalność, progi werdyktu |
| `test_camera.py` | 11 | model kamery Arducam z 5 m, kadr 4:3, rozmycie ruchem, prawdziwe ramki w kadrze prostokątnym |
| `test_generate_field.py` | 9 | generator sztucznego pola z etykietami YOLO |
| `test_normalization.py` | 32 | aliasy klas, kody EPPO z fazą, deduplikacja, podział po grupach bez przecieków, kafelkowanie, format YOLO |
| `test_weedmaize.py` | 5 | zmniejszanie do skali drona, limit zdjęć, wszystko do treningu |
| `test_new_datasets.py` | 12 | adaptery EWIS1 (tylko kukurydza, zamienione wymiary w XML, obrócone ramki, odłożony dzień) i PhenoBench (maski 16-bit) |
| `test_models.py` | 58 | plan treningu, kontrola danych, ocena według dni i rozmiaru, podział dniami, analiza zbioru (trening wymaga ultralytics — nie jest uruchamiany) |
| `test_season.py` | 8 | nadpróbkowanie, zbiór z wybranych dni, progi według dni |
| `test_mission.py` | 33 | telemetria, georeferencja (błąd ok. 3 cm na znacznikach), scalanie, mapa |
| `test_hailo.py` | 18 | `HailoDetector` bez sprzętu: letterbox, odczyt NMS Hailo, wybór detektora, AP50, konfiguracje misji, skrypty |
| `test_inference.py` | 9 | WBF, ensemble, TTA z odbiciem, niski próg z głosowaniem |
| `test_fusion.py` | 8 | łączenie wykryć z wielu zdjęć, symulowany lot, komenda `symulacja` |
| `test_gui_flight.py` | 6 | zakładka 4 interfejsu (Streamlit AppTest): lot z detekcją, kafelki według dni, tryb kamery drona |

## Zasada testów

Dane testowe powstają z **znanego** obrazu albo ze znanych ramek, więc wiadomo,
jak powinien wyglądać wynik. Testy sprawdzają nie tylko, czy kod się wykonał,
ale czy wynik zgadza się z prawdą — np. błąd mapowania w centymetrach, PSNR
mozaiki albo dokładne położenie ramki po przeliczeniu.