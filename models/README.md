# models — training, evaluation and export of the weed detector

**Part of the pipeline:** model (after `normalization/`, before `mission/` and Hailo).

Takes the YOLO datasets from `normalization/`, trains YOLOv8n with one class "weed"
(`chwast`), evaluates it — also separately for every flight day — and exports it to ONNX,
from which the Hailo-8 `.hef` file is compiled (`scripts/HAILO.md`).

Steps 2–4 of `scripts/final_model.sh` use this package.
How it works inside — flow, technologies, key functions explained: [`PIPELINE.md`](PIPELINE.md).

---

## Final model

| | |
|---|---|
| file | `models/weights/fieldeye_final.pt` (experiment name: `dane_ewis1_bt`) |
| architecture | YOLOv8n, one class "weed", input 1280 px |
| training data | WeedsGalore (oversampled: 25 May ×3, 30 May ×2) + Sudars + EWIS1 maize without 16 May 2022 — 1866 tiles |
| validation / test | WeedsGalore only (validation picks the best epoch, so the model is chosen for the drone view) + held-out EWIS1 day |
| training | 100 epochs, SGD, lr0 0.01, batch 16, patience 30, seed 0 |
| inference | **TTA**: original + horizontal flip merged by weighted box fusion; threshold 0.15 |
| reproduce | `bash scripts/final_model.sh` |

| Test | AP50 | Recall | Precision |
|---|---|---|---|
| EWIS1, 16 May 2022 — ordinary RGB camera, day never seen in training (132 tiles, 4422 weeds), with TTA | **0.817** | **78.4%** | **84.3%** |
| EWIS1, same, without TTA | 0.812 | 79.1% | 81.1% |
| WeedsGalore, whole season (26 tiles, 1628 weeds), with TTA | 0.714 | 69.2% | 64.0% |
| WeedsGalore, 25 May (youngest seedlings), with TTA | — | 32.2% | 42.0% |

AP50 is computed by our own `ap50()` (identical code for `.pt`, `.onnx` and `.hef`);
Ultralytics' own validation mAP50 comes out about 1.5 pp lower.

**Key finding:** a model trained without EWIS1 (WeedsGalore + Sudars) reached AP50 0.718 on
WeedsGalore but only 0.044 on ordinary RGB photos (recall 0.2%) — it had learned the look of the
multispectral camera. Adding EWIS1 to training fixed this without any loss on WeedsGalore.

---

## Experiment history

WeedsGalore test, YOLOv8n, input 1280, 100 epochs unless stated otherwise.

| Variant | Result | Conclusion |
|---|---|---|
| 30 epochs (Windows) | mAP50 0.567 | too short |
| `wg100`: WeedsGalore only | mAP50 0.651; 25 May 17–21% | baseline |
| YOLOv8s | mAP50 0.645 | 3.5× the compute, no gain: rejected |
| Sudars first, then WeedsGalore (two stages) | mAP50 0.572 / 0.624 | sequential training worse than joint |
| `wg_sudars`: WeedsGalore + Sudars together | mAP50 0.659; 25 May 27–28% | Sudars adds about +8 pp in May |
| test on a whole unseen day (25 May) | recall 23–26% in all variants | the model does not transfer to an unseen earliest stage |
| **oversampling May** | AP50 +1.3–1.7 pp, May +0.6–3.6 pp (2 seeds) | **accepted** |
| smaller `scale` augmentation | more false alarms | rejected |
| separate model for May (fine-tuning) | no improvement | rejected: one shared model |
| threshold per day | May 27 → 40%, but false alarms 111 → 499 | rejected: 0.15 everywhere |
| **TTA** | AP50 +2.2–2.8 pp, about −30% false alarms (2 seeds) | **accepted** |
| two models at once (ensemble) | like TTA, but two HEF files | rejected |
| multi-view fusion, voting, low threshold + voting | no net gain | rejected |
| test on an ordinary RGB camera (EWIS1) | model without EWIS1: AP50 0.044, recall 0.2% | did not transfer to RGB photos |
| EWIS1 converted to WeedsGalore colours | AP50 0.224 | the cause is mostly the multispectral colours |
| **+ EWIS1 in training** | **EWIS1 AP50 0.817**, no loss on WeedsGalore | **accepted: final model** |
| WeedsGalore colour correction (sRGB) | EWIS1 +2 pp, 25 May −5.7 pp | rejected |
| input 640 instead of 1280 | EWIS1 −3.7 pp, 25 May −10.5 pp | keep 1280 |
| + PhenoBench (at 640) | EWIS1 +2.2 pp | small gain, not in the final model |
| + sorghum from EWIS1 | no change | rejected |

Differences up to about 2 pp AP50 and 4 pp recall in May (332 seedlings) are within the
run-to-run noise, so every improvement was confirmed with a second seed.

---

## Commands (`python -m models.cli ...`)

| Command | Used by | What it does |
|---|---|---|
| `srodowisko` | you | library versions and the device training would use (CUDA → MPS → CPU) |
| `sprawdz DATA.yaml` | you | checks a dataset before training: folders, missing labels, empty parts, file leaks between parts |
| `nadprobkuj DATASET --waga DAY=N ... -o DATA.yaml` | final_model.sh step 2 | oversampling: copies training tiles of chosen days N times, adds other datasets' training parts |
| `trenuj --plan PLAN.json -o FOLDER` | final_model.sh step 2 | training from a plan file → `FOLDER/model.pt` |
| `ocen-detektor MODEL DATA.yaml [--tta]` | final_model.sh step 3, check_hef.sh | AP50, recall and precision at the threshold, per flight day — same code for `.pt`, `.onnx`, `.hef` |
| `onnx MODEL.pt -o MODEL.onnx --wejscie 1280 --opset 11` | final_model.sh step 4 | export to ONNX with a fixed input (what the Hailo compiler needs) |
| `kalibracja IMAGES -o FOLDER --liczba 128 --rozmiar 1280` | final_model.sh step 4 | calibration images for INT8 quantization, letterboxed to the network input |

### Training plan

```json
{"model_bazowy": "yolov8n.pt", "batch": 16, "cierpliwosc": 30, "seed": 0,
 "etapy": [{"nazwa": "dane-ewis1-bt", "dane": "zbior_lotnicze/dataset/data_dane_ewis1_bt.yaml",
            "epoki": 100, "lr0": 0.01, "wejscie": 1280}]}
```

| Field | Meaning |
|---|---|
| `model_bazowy` | starting weights (COCO or another model) |
| `batch`, `cierpliwosc`, `seed` | batch size, early-stopping patience in epochs, random seed |
| `etapy[]` | stages; each starts from the weights of the previous one (the final model uses one stage) |
| `etapy[].dane` | `data.yaml`; `train` may be a list of folders (several datasets at once) |
| `etapy[].augmentacje` | extra Ultralytics augmentation settings, e.g. `{"degrees": 15}` |

The optimizer is set to SGD explicitly, because with `auto` Ultralytics ignores `lr0`.

### Oversampling the early season

```bash
python -m models.cli nadprobkuj zbior_lotnicze/dataset --waga 2023-05-25=3 --waga 2023-05-30=2 \
       --dodaj-trening zbior_naziemne/dataset --dodaj-trening zbior_ewis1_podzial/dataset \
       -o zbior_lotnicze/dataset/data_dane_ewis1_bt.yaml
```

Copies go to `images/train_nadprob` under different names; validation and test are untouched.
Running it again does not add more copies.

### Evaluation

```bash
python -m models.cli ocen-detektor models/weights/fieldeye_final.pt zbior_ewis1_podzial/dataset/data.yaml --prog 0.15 --tta
```

Dates in file names are recognised as `2023-05-25` (WeedsGalore) and `20220516` (EWIS1),
which gives the per-day table.

---

## Files

| File | Role |
|---|---|
| `cli.py` | command line (table above); every command is a `cmd_<name>()` function |
| `plan.py` | `Stage`, `TrainingPlan` (read/write JSON or YAML), `katalog_etykiet()` (images → labels folder), `sprawdz_dane()` (dataset check) — no Ultralytics needed |
| `training.py` | training stage by stage through Ultralytics; saves `plan.json`, `historia.json`, `model.pt` |
| `evaluation.py` | own metrics: box matching, precision/recall at thresholds, `ap50()` (101-point, like Ultralytics), per-day results, `ocen_detektor()` for any detector |
| `season.py` | `nadprobkuj()`: oversampling of early flight days |
| `export.py` | `do_onnx()`, `przygotuj_kalibracje()` (letterboxed calibration images), `rozmiary()` (file size, INT8 estimate) |
| `device.py` | device choice CUDA → MPS (Mac) → CPU; sets the MPS fallback before PyTorch is imported |
| `weights/fieldeye_final.pt` | the final model |

Training needs `ultralytics` (AGPL-3.0 licence); the plan, the dataset check and the metrics
work without it. Tests: `tests/test_models.py`, `test_season.py`, `test_hailo.py`.
