# FIELDEYE — weed detection from drone images

Software for the vision layer of the **FIELDEYE** project: a drone (Raspberry Pi 5 with a
Hailo-8 accelerator, Arducam 64MP camera, flight altitude 5 m) photographs a field, a YOLO
model finds weeds on every photo, and the detections are turned into GPS coordinates of
spray points for the farmer.

The repository covers the whole path: public datasets → one training dataset → trained
detector → model compiled for Hailo-8 → program that runs on the drone, plus a flight
simulator, a mosaic stitcher and a graphical interface for testing all of it on a laptop.

---

## Overall pipeline

```
 PUBLIC DATASETS                    TRAINING (Mac / PC)                         DEPLOYMENT
 ───────────────                    ───────────────────                         ──────────
 WeedsGalore (drone) ─┐
 Sudars (ground)     ─┼─► normalization/  ─► models/          ─► models/weights/fieldeye_final.pt
 EWIS1 (drone, RGB)  ─┘   formats → YOLO      train, evaluate          │
                          dedup, split,       per flight day,          ├─► ONNX + calibration images (models/export.py)
                          tiles 896 px        TTA, threshold 0.15      │        │
                                                                       │        ▼  scripts/compile_hef.py (Linux x86 / WSL)
                                                                       │   fieldeye_final_1280_h8.hef (Hailo-8)
                                                                       │        │
 DRONE                                                                 ▼        ▼
 photo + telemetry (.geojson) ─────────────────────────────► mission/ fieldeye-detect (binary on the Pi)
                                                               tiles → model → GPS coordinates
                                                               → spray map (oprysk.geojson / .csv)

 LAPTOP TOOLS
 stitchlive/  flight simulator (cuts one big image into drone frames), mosaic stitching, camera model
 app.py       Streamlit GUI: split an image into frames, stitch them, simulated flight with detection
```

`scripts/final_model.sh` runs the whole training side in one go: normalization of the three
datasets → training → evaluation on two test sets → ONNX package for the Hailo compiler.

---

## Final model

`models/weights/fieldeye_final.pt` — YOLOv8n, one class (`chwast` = weed), input 1280 px,
trained on WeedsGalore (early-season days oversampled) + Sudars + EWIS1 maize (1866 tiles).
Run with test-time augmentation (TTA: original + horizontal flip, merged by WBF) and
confidence threshold 0.15.

| Test set | Tiles / weeds | AP50 | Recall | Precision |
|---|---|---|---|---|
| EWIS1, 16 May 2022 — ordinary RGB camera, flight day never seen in training | 132 / 4422 | **0.817** | **78.4%** | **84.3%** |
| WeedsGalore, whole season | 26 / 1628 | 0.714 | 69.2% | 64.0% |
| WeedsGalore, 25 May (youngest seedlings) | 8 | — | 32.2% | 42.0% |

The weak point is the earliest growth stage: the seedlings are only a few pixels wide.
Results are stored in `wyniki/model_koncowy/*.json`; the full story is in
[`models/README.md`](models/README.md).

The same model compiled for Hailo-8 (`fieldeye_final_1280_h8.hef`, 5.4 MB, HailoRT 4.23) is
in `model_fin.zip`. Measuring its accuracy on the Pi against the `.pt` (`scripts/check_hef.sh`)
is still to be done.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests -q                    # ~215 tests, 1–3 min, no data needed
streamlit run app.py                         # GUI in the browser
caffeinate -i bash scripts/final_model.sh    # rebuild the final model from raw data (~4 h, needs dane/)
```

Detection on a folder of drone photos (with `.geojson` telemetry next to them):

```bash
python -m mission.detect_cli PHOTOS RESULTS models/weights/fieldeye_final.pt
```

On Windows use `py -m ...` instead of `python -m ...`. `caffeinate -i` only keeps a Mac awake.

---

## Repository map

| Path | Part | What it is |
|---|---|---|
| [`normalization/`](normalization/README.md) | data | public datasets (each in its own format) → YOLO datasets: adapters, class mapping, deduplication, leak-free split, tiling |
| [`models/`](models/README.md) | model | training plan, training (Ultralytics), evaluation (AP50, per flight day), ONNX export, calibration images |
| `models/weights/fieldeye_final.pt` | model | the final model (the only weights file kept in git) |
| [`mission/`](mission/README.md) | deployment | `fieldeye-detect` (detection on new photos → spray map), detectors for `.pt` / `.onnx` / `.hef`, simulated drone |
| [`stitchlive/`](stitchlive/README.md) | simulator / mosaic | flight simulator, mosaic stitching (`global`, `detail`, `basic`), drone camera model |
| [`scripts/`](scripts/README.md) | tooling | `final_model.sh` (everything for the model), Hailo compilation and check, binary build, Pi package, test-data generators |
| [`tests/`](tests/README.md) | tests | ~215 automatic tests for every part |
| [`docs/fieldeye-detect.md`](docs/fieldeye-detect.md) | deployment | full manual of the `fieldeye-detect` program: telemetry format, outputs, coordinates, binary, Raspberry Pi |
| `app.py` | GUI | Streamlit interface with four tabs (see below) |
| `wyniki/model_koncowy/` | results | evaluation of the final model: `ewis1.json`, `ewis1_tta.json`, `wg.json`, `wg_tta.json` |
| `model_fin.zip` | deployment | compiled `fieldeye_final_1280_h8.hef` + a Word note on running it on the Pi |
| `requirements.txt` | setup | all Python dependencies (OpenCV contrib, Streamlit, PyYAML, Ultralytics, ONNX) |
| `pyproject.toml` | setup | package metadata; console scripts `stitchlive`, `fieldeye-models`, `fieldeye-detect` after `pip install -e .` |
| `Makefile` | setup | shortcuts: `make test`, `make model`, `make gui`, `make demo`, `make clean` |
| `claudowy_opis_projektu.docx`, `poradnik_jak_rozpakowac_zipa_extreme.txt` | team notes | project description and a guide for unpacking the packages |

Folders created by the scripts and ignored by git: `dane/` (raw datasets), `zbior_lotnicze/`,
`zbior_naziemne/`, `zbior_ewis1_podzial/` (normalized datasets), `modele_*/` (training runs),
`hailo/` (ONNX package for the compiler).

---

## GUI (`app.py`)

```bash
streamlit run app.py
```

| Tab | What it does |
|---|---|
| **1. Rozkład zdjęcia** (split image) | upload an image or generate a synthetic field, choose the flight pattern and disturbances, cut it into drone frames; shows the route on the original |
| **2. Kadry do pobrania** (frames) | gallery of the frames with their true position; download frames (ZIP), frames + `prawda.json` + route preview, single frames |
| **3. Składanie** (stitching) | frames from tab 1 or your own photos; the mosaic grows frame by frame; comparison with the original; download mosaic (JPG), animation (GIF), full report (ZIP) |
| **4. Lot z detekcją** (flight with detection) | simulated flight over test tiles of one day with the weed model (default `models/weights/fieldeye_final.pt`, input 1280): detections on every frame, ground truth boxes, weeds on the mosaic, optional real drone camera geometry, GIF of the flight |

---

## Status

| Part | State |
|---|---|
| normalization of WeedsGalore, Sudars, EWIS1 | done |
| training and evaluation | done — Windows (CUDA) and Mac (MPS) |
| weed detector | final model: AP50 0.817 on an RGB camera, 0.714 on WeedsGalore |
| ONNX export, Hailo-8 `.hef` | compiled; accuracy check on the Pi pending |
| `fieldeye-detect` (detection + spray map) | done, as Python program and single-file binary |
| detection → coordinates | done; error dominated by GPS (≈1 m without RTK, a few cm with RTK) |
| flight simulator, mosaic stitching (`global`) | done; PSNR 22–25 dB on simulated flights |
| geo-referenced mosaic, IMU tilt correction | not done (tilt is only used to reject photos) |
| fine-tuning on photos from our own drone | waiting for data |

---

## Conventions

- Package, file and class names are English; comments, docstrings, many function names
  and command-line options are Polish (e.g. `--prog` = threshold, `-o` = output folder).
- Every function has a docstring in a fixed template: what it does, why, arguments,
  return value, where it is used.
- Frequent Polish words: `chwast` weed, `uprawa` crop, `zbior` dataset, `kafelek` tile,
  `kadr` frame, `mozaika` mosaic, `prog` threshold, `wyniki` results, `dane` data,
  `ocen` evaluate, `trenuj` train, `rozloz` split into frames, `oprysk` spraying, `lot` flight.

---

## Limitations

- Nadir camera and flat field are assumed both in coordinates and in stitching.
- The model has never seen photos from our own drone and field; public datasets only.
- Map accuracy is limited by GPS, not by the computation.
- The mosaic is not geo-referenced (it is for people to look at; spraying uses per-photo coordinates).
