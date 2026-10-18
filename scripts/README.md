# scripts/ — skrypty pomocnicze

Wszystkie uruchamia się z katalogu `stitch-live`, z aktywnym `.venv`.
Długie treningi warto puszczać z `caffeinate -i`, żeby Mac nie zasnął.

## Przygotowanie i dane

| Skrypt | Do czego |
|---|---|
| `setup_mac.sh` | tworzy `.venv`, instaluje biblioteki, sprawdza GPU, uruchamia testy |
| `download_datasets.sh` | pobiera prawdziwe zdjęcia z drona z ODMdata (do testów składania) |
| `generate_field.py` | generuje sztuczne zdjęcia pola z etykietami chwastów YOLO |
| `make_test_data.py` | starsza, prostsza wersja symulatora (nowsza: `python -m stitchlive.cli rozloz`) |
| `simulate_flight.sh` | kopiuje zdjęcia do katalogu w odstępach czasu, jak dron w locie |

## Eksperymenty z modelem

| Skrypt | Co sprawdza | Czas |
|---|---|---|
| `season_experiments.sh` | sezonowość: nadpróbkowanie maja, `scale`, model wczesny, progi dnia (`WARIANTY`, `SEED`) | do ok. 7 h |
| `inference_experiments.sh` | poprawy bez treningu: TTA, ensemble, łączenie widoków, niski próg + głosowanie | minuty |
| `fusion_eval.sh` | łączenie wykryć z wielu zdjęć w symulowanym locie | minuty |
| `new_data_experiment.sh` | nowe dane w treningu: EWIS1 (z sorgo i bez), PhenoBench (`WARIANTY`) | 4–12 h |
| `ewis1_test.sh` | drugi test: odłożony dzień lotu EWIS1; z `TRENUJ=1` także trening bez tego dnia | minuty / ok. 4 h |
| `weedmaize_experiment.sh` | WeedMaize w treningu (`PRZEPIS`, `SKALA`, `MAKS`) | ok. 6 h |

Każdy skrypt pomija kroki już zrobione (zbudowane zbiory, wytrenowane modele)
i kończy się tabelą porównania z punktem odniesienia.

## Hailo

| Plik | Do czego |
|---|---|
| `HAILO.md` | kompilacja do `.hef` krok po kroku: Mac (ONNX, kalibracja) → Linux x86 / WSL (`hailomz`) → Raspberry Pi |
| `finish_models.sh` | Mac: trening modelu 640, oceny, eksport ONNX z kontrolą, kalibracja, paczka `hailo/` |
| `check_hef.sh` | Raspberry Pi: akcelerator, `parse-hef`, ocena `.hef` i porównanie z `.pt` |

## simulate_flight.sh

Kopiuje przez plik tymczasowy i zmienia nazwę — operacja atomowa, więc
obserwujący program (`watch`) nigdy nie zobaczy pliku w połowie zapisu.

```bash
python -m stitchlive.cli watch naplyw --renderuj-co 4 -o wyniki     # terminal 1
./scripts/simulate_flight.sh dane/test/images naplyw 1              # terminal 2
```