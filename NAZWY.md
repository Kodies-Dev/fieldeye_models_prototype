# Nazwy: stare → nowe

Pakiety, pliki i klasy mają nazwy angielskie. Opisy, komentarze, nazwy funkcji,
komendy w wierszu poleceń (`trenuj`, `ocen`, `prog`…) i formaty danych (klucze planów JSON,
sekcje `misja.yaml`, nazwy plików wynikowych) zostały bez zmian, więc istniejące zbiory,
modele i plany działają dalej.

Starsze dokumenty (Word, arkusz) używają nazw z lewej kolumny.

## Pakiety

| Było | Jest |
|---|---|
| `misja/` | `mission/` |
| `modele/` | `models/` |
| `normalizacja/` | `normalization/` |
| `konfiguracje/` | `configs/` |

## Moduły i komendy

| Było | Jest |
|---|---|
| `stitchlive/rozklad.py` | `stitchlive/flight_sim.py` |
| `stitchlive/pole.py` | `stitchlive/field.py` |
| `stitchlive/globalny.py` | `stitchlive/global_stitcher.py` |
| `stitchlive/ocena.py` | `stitchlive/evaluation.py` |
| `misja/detektor.py` | `mission/detector.py` |
| `misja/konfiguracja.py` | `mission/config.py` |
| `misja/telemetria.py` | `mission/telemetry.py` |
| `misja/wlocie.py` | `mission/in_flight.py` |
| `misja/polocie.py` | `mission/post_flight.py` |
| `normalizacja/adaptery.py` | `normalization/adapters.py` |
| `normalizacja/taksonomia.py` | `normalization/taxonomy.py` |
| `normalizacja/kroki.py` | `normalization/steps.py` |
| `normalizacja/eksport.py` | `normalization/export.py` |
| `normalizacja/buduj.py` | `normalization/build.py` |
| `normalizacja/przyklady.py` | `normalization/examples.py` |
| `modele/urzadzenie.py` | `models/device.py` |
| `modele/trening.py` | `models/training.py` |
| `modele/ocena.py` | `models/evaluation.py` |
| `modele/podzial.py` | `models/splits.py` |
| `modele/eksport.py` | `models/export.py` |
| `modele/analiza.py` | `models/analysis.py` |

| Komenda — było | Jest |
|---|---|
| `python -m modele.cli …` | `python -m models.cli …` |
| `python -m misja.cli …` | `python -m mission.cli …` |
| `python -m normalizacja.buduj …` | `python -m normalization.build …` |
| `python -m stitchlive.cli …` | bez zmian |

## Inne pliki

| Było | Jest |
|---|---|
| `scripts/generuj_pole.py` | `scripts/generate_field.py` |
| `scripts/przygotuj_mac.sh` | `scripts/setup_mac.sh` |
| `scripts/dokoncz_model.sh` | `scripts/finish_models.sh` |
| `scripts/sprawdz_hef.sh` | `scripts/check_hef.sh` |
| `konfiguracje/misja_wg_sudars.yaml` | `configs/mission_wg_sudars.yaml` |
| `konfiguracje/misja_hailo.yaml` | `configs/mission_hailo.yaml` |
| `normalizacja/taksonomia.yaml` | `normalization/taxonomy.yaml` |
| `normalizacja/JAK_DZIALA.md` | `normalization/HOW_IT_WORKS.md` |
| `tests/test_rozklad.py` | `tests/test_flight_sim.py` |
| `tests/test_misja.py` | `tests/test_mission.py` |
| `tests/test_modele.py` | `tests/test_models.py` |
| `tests/test_normalizacja.py` | `tests/test_normalization.py` |
| `tests/test_globalny.py` | `tests/test_global_stitcher.py` |
| `tests/test_generuj_pole.py` | `tests/test_generate_field.py` |
| `tests/test_gui_lot.py` | `tests/test_gui_flight.py` |

## Klasy

| Było | Jest | Plik |
|---|---|---|
| `SkladaczGlobalny` | `GlobalStitcher` | `stitchlive/global_stitcher.py` |
| `SkladaczPodstawowy` | `BasicStitcher` | `stitchlive/basic.py` |
| `SkladaczDetail` | `DetailStitcher` | `stitchlive/detail.py` |
| `ZrodloKatalog` | `FolderSource` | `stitchlive/source.py` |
| `ZrodloWsadowe` | `BatchSource` | `stitchlive/source.py` |
| `Ustawienia` | `SimSettings` | `stitchlive/flight_sim.py` |
| `PlanTreningu` | `TrainingPlan` | `models/plan.py` |
| `Etap` | `Stage` | `models/plan.py` |
| `DetektorYOLO` | `YoloDetector` | `mission/detector.py` |
| `DetektorKafelki` | `TiledDetector` | `mission/detector.py` |
| `DetektorAtrapa` | `DummyDetector` | `mission/detector.py` |
| `DetektorHailo` | `HailoDetector` | `mission/detector.py` |
| `ZapisJSONL` | `JsonlWriter` | `mission/in_flight.py` |
| `Uklad` | `LocalFrame` | `mission/geo.py` |
| `Kamera` | `CameraConfig` | `mission/config.py` |
| `Lot` | `FlightConfig` | `mission/config.py` |
| `Mapa` | `MapConfig` | `mission/config.py` |
| `Detekcja` (ustawienia detekcji) | `DetectionConfig` | `mission/config.py` |
| `Konfiguracja` | `MissionConfig` | `mission/config.py` |
| `Kadr` | `FrameInfo` | `mission/telemetry.py` |
| `Taksonomia` | `Taxonomy` | `normalization/taxonomy.py` |
| `Obiekt` | `LabeledObject` | `normalization/manifest.py` |
| `Rekord` | `Record` | `normalization/manifest.py` |
| `Warstwa` | `Layer` | `scripts/generate_field.py` |
| `Detekcja` (jedno wykrycie) | `Detection` | `mission/detector.py` |
