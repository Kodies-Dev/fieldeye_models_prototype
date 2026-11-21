# scripts/ — skrypty pomocnicze

Wszystkie uruchamia się z głównego katalogu projektu.

## Model końcowy — jeden skrypt

```bash
caffeinate -i bash scripts/final_model.sh     # ok. 4 h z treningiem
streamlit run app.py
```

`final_model.sh` z surowych danych w `dane/` odtwarza najlepszy model i wszystko, czego
potrzebuje `app.py`:

1. normalizacja WeedsGalore → `zbior_lotnicze/`, Sudars → `zbior_naziemne/`,
   EWIS1 → `zbior_ewis1_podzial/` (dzień 16.05.2022 jako test),
2. trening → `modele_dane_ewis1_bt/model.pt`, kopia do `models/weights/fieldeye_final.pt`,
3. ocena tego modelu na teście EWIS1 i WeedsGalore, bez i z TTA → `wyniki/model_koncowy/`,
4. paczka ONNX dla Hailo → `hailo/hailo_final.zip`,
5. tabela wyników.

Wymaga aktywnego `.venv` (`pip install -r requirements.txt`). Kroki już zrobione są
pomijane, więc można przerwać i uruchomić ponownie.

## Hailo (inne maszyny)

| Plik | Gdzie | Do czego |
|---|---|---|
| `HAILO.md` | — | kompilacja do `.hef` krok po kroku: Mac → WSL → Raspberry Pi |
| `compile_hef.py` | WSL | ONNX → HEF przez kompilator Hailo, własna konfiguracja NMS (obraz 1280, 1 klasa, próg 0,05, do 300 ramek) |
| `check_hef.sh` | Raspberry Pi | akcelerator, `parse-hef`, szybkość, dokładność na teście EWIS1 i porównanie z `.pt` (AP50 0,812 / 0,817) |

## Składanie mozaiki i symulator

| Skrypt | Do czego |
|---|---|
| `generate_field.py` | generuje sztuczne zdjęcia pola z etykietami chwastów YOLO |
| `make_test_data.py` | starsza, prostsza wersja symulatora (`make dane`) |
| `simulate_flight.sh` | kopiuje zdjęcia do katalogu w odstępach czasu, jak dron w locie |

`simulate_flight.sh` kopiuje przez plik tymczasowy i zmienia nazwę — operacja atomowa, więc
obserwujący program (`watch`) nigdy nie zobaczy pliku w połowie zapisu.

```bash
python -m stitchlive.cli watch naplyw --renderuj-co 4 -o wyniki     # terminal 1
./scripts/simulate_flight.sh dane/test/images naplyw 1              # terminal 2
```

## Binarka fieldeye-detect

| Skrypt | Do czego |
|---|---|
| `fieldeye-detect.sh` | zdjęcia z katalogu → model chwastów → wynik w terminalu (bez budowania binarki) |
| `fieldeye_detect_main.py` | punkt wejścia binarki: detekcja, symulowany dron z GPS, ocena namiarów |
| `build_binary.sh` | buduje binarkę przez PyInstaller — na tym systemie, na którym ma działać |
| `przygotuj_folder.sh` | samowystarczalny folder: binarka + modele + przykłady + `uruchom.sh` |

Opis: `docs/fieldeye-detect.md`.
