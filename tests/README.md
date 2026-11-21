# tests/ — testy automatyczne

```bash
python -m pytest tests -q                       # wszystkie, ok. 1-2 min, ok. 210 testów
python -m pytest tests -q --deselect tests/test_gui_flight.py    # bez interfejsu, ok. 1 min
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
| `test_new_datasets.py` | 9 | adapter EWIS1: tylko kukurydza, zamienione wymiary w XML, obrócone ramki, odłożony dzień, budowa ze skalą |
| `test_colors.py` | 5 | kolory WeedsGalore: liniowe bez zmian, sRGB z korektą gamma (16 i 8 bit), opcja `--weedsgalore-kolory` |
| `test_compile_hef.py` | 4 | `compile_hef.py` bez kompilatora: konfiguracja NMS 1280, podmiana ścieżki w skrypcie modelu, zdjęcia kalibracyjne RGB |
| `test_models.py` | 32 | plan treningu, kontrola danych, etykiety, AP50 i miary przy progu, wynik według dni, kalibracja, wybór urządzenia (trening wymaga ultralytics — nie jest uruchamiany) |
| `test_season.py` | 4 | nadpróbkowanie wczesnych dni: liczby kopii, brak przecieku, ponowne uruchomienie, opcje komendy |
| `test_hailo.py` | 17 | `HailoDetector` bez sprzętu: letterbox, odczyt NMS Hailo, wybór detektora po rozszerzeniu, AP50, opcje komend, składnia skryptów |
| `test_inference.py` | 10 | IoU, WBF, TTA z odbiciem, kafelki (przesunięcie i duplikaty z zakładki), atrapa, `zbuduj_z_plikow` |
| `test_detect_cli.py` | 10 | `fieldeye-detect`: nowe zdjęcia w folderze, plik w trakcie zapisu, opcje, wyniki |
| `test_spray_map.py` | 14 | telemetria GeoJSON, piksel → metry, punkty oprysku, symulowany dron → detekcja → porównanie z prawdą |
| `test_binary.py` | 6 | binarka: detektor ONNX, model obok binarki, brak ciężkich bibliotek, polecenia `dron` i `porownaj`, skrypty |
| `test_gui_flight.py` | 6 | zakładka 4 interfejsu (Streamlit AppTest): lot z detekcją, kafelki według dni, tryb kamery drona |
| `test_sim_tiles.py` | 4 | pole do lotu odtworzone z kafelków z położeniem w nazwie: zdjęcie co do piksela, chwast z zakładki jako jedna ramka |

## Zasada testów

Dane testowe powstają z **znanego** obrazu albo ze znanych ramek, więc wiadomo,
jak powinien wyglądać wynik. Testy sprawdzają nie tylko, czy kod się wykonał,
ale czy wynik zgadza się z prawdą — np. błąd mapowania w centymetrach, PSNR
mozaiki albo dokładne położenie ramki po przeliczeniu.