# Model pipeline — how it works inside

How the YOLO datasets from `normalization/` become the final weed detector
`models/weights/fieldeye_final.pt`, how it is scored, and how it is handed over to the Hailo
compiler. Commands, results and the experiment history are in [`README.md`](README.md); this page
explains the flow and the hardest parts of the code.

**Input:** `zbior_lotnicze/`, `zbior_naziemne/`, `zbior_ewis1_podzial/`. **Output:** `model.pt`,
`wyniki/model_koncowy/*.json`, `hailo/hailo_final.zip`. **Runs on:** the training computer
(CUDA, Apple MPS or CPU), steps 2–4 of `scripts/final_model.sh`.

---

## 1. Flow

```
zbior_lotnicze/dataset (WeedsGalore)   zbior_naziemne/dataset (Sudars)   zbior_ewis1_podzial/dataset (EWIS1)
        │                                      │                                   │
        ▼                                      ▼                                   ▼
models.cli nadprobkuj ──► season.nadprobkuj()
        copies of 25 May tiles ×3, 30 May ×2 → images/train_nadprob, labels/train_nadprob
        → data_dane_ewis1_bt.yaml   train = WG train + WG copies + Sudars train + EWIS1 train
                                    val / test = WeedsGalore only
        │
models.cli trenuj --plan plan_dane_ewis1_bt.json
        plan.TrainingPlan.wczytaj() → training.trenuj() → for every stage: training.trenuj_etap()
            plan.sprawdz_dane()            classes, folders, labels, leaks — stop before training
            ultralytics YOLO(...).train()  yolov8n.pt, 100 epochs, SGD lr0 0.01, batch 16,
                                           patience 30, input 1280, device from device.wybierz()
        → modele_dane_ewis1_bt/model.pt  (+ plan.json, historia.json, trening/<stage>/)
        → final_model.sh copies it to models/weights/fieldeye_final.pt
        │
models.cli ocen-detektor  ×4  (EWIS1 / WeedsGalore × without / with TTA)
        mission.detector.zbuduj_z_plikow(model, 0.001, 1280, tta) → evaluation.ocen_detektor()
            predykcje_detektora()   detector over every test tile
            ap50()                  area under the precision–recall curve, IoU 0.5
            pr_przy_progach([0.15]) recall / precision at the working threshold
            ocen_wedlug_grup()      the same per flight day
        → wyniki/model_koncowy/{ewis1, ewis1_tta, wg, wg_tta}.json
        │
models.cli onnx        → export.do_onnx()              ONNX, input 1280, opset 11, static, simplified
models.cli kalibracja  → export.przygotuj_kalibracje() 128 training tiles, letterboxed to 1280
        → hailo/hailo_final.zip  ──►  Hailo compilation: scripts/HAILO_PIPELINE.md
```

---

## 2. Technologies

| Technology | Used for | Where |
|---|---|---|
| **Ultralytics YOLOv8** (AGPL-3.0) | the network (YOLOv8n), training loop, augmentation, `predict`, ONNX export | `training.py`, `export.py`, `mission/detector.py` |
| **PyTorch** | backend of Ultralytics on CUDA, Apple MPS or CPU | through Ultralytics; `device.py` |
| **NumPy** | cumulative TP / FP, precision envelope, interpolation and integration for AP50 | `evaluation.py` |
| **OpenCV** | image sizes for labels, letterbox of the calibration images | `evaluation.py`, `export.py` |
| **PyYAML** | reading and writing `data.yaml` (also lists of training folders) | `plan.py`, `season.py` |
| **ONNX**, **onnxslim** | exported model and graph simplification | `export.do_onnx()` |
| JSON | training plan, history, evaluation results | `plan.py`, `training.py`, `evaluation.py` |

Ultralytics is imported **only inside** the functions that need it (`_yolo()`, `do_onnx()`), so the
plan, the data check and the evaluation work without the heavy package.

---

## 3. Files in this process

| File | Role in the flow |
|---|---|
| `cli.py` | `python -m models.cli` — commands `srodowisko`, `sprawdz`, `nadprobkuj`, `trenuj`, `ocen-detektor`, `onnx`, `kalibracja` |
| `season.py` | oversampling of the early-season days |
| `plan.py` | `Stage`, `TrainingPlan` (plan file), `sprawdz_dane()` (data check), `katalog_etykiet()` |
| `training.py` | runs the plan stage by stage through Ultralytics |
| `device.py` | picks CUDA → MPS → CPU; enables the MPS fallback |
| `evaluation.py` | AP50, recall / precision at a threshold, results per flight day — the same code for `.pt`, `.onnx` and `.hef` |
| `export.py` | ONNX export and calibration images for INT8 quantization |
| `mission/detector.py` | the detectors used by the evaluation (`zbuduj_z_plikow`, TTA) — see [`mission/PIPELINE.md`](../mission/PIPELINE.md) |

---

## 4. Key classes and functions

### `nadprobkuj()` — more weight for the youngest seedlings (`season.py`)

The weak point of every model was the earliest flight day (25 May: tiny seedlings, about 41
weeds per tile against about 90 in June), and those tiles are the rarest in training.
`nadprobkuj(dataset, {"2023-05-25": 3, "2023-05-30": 2}, ...)`:

1. reads the day from every training tile name (`\d{4}-\d{2}-\d{2}`),
2. copies a tile with weight `w` another `w − 1` times into `images/train_nadprob/` (label to
   `labels/train_nadprob/`) under the name `<name>__kopiaN` — a missing label becomes an empty
   file, so an empty tile stays a negative example,
3. deletes the old copies first, so running it twice adds nothing,
4. writes a `data.yaml` whose `train` is a **list**: WeedsGalore originals, the copies, and the
   `images/train` folders of Sudars and EWIS1 (absolute paths); `val` and `test` stay WeedsGalore.

Copies instead of a weighted sampler: it works with every Ultralytics version, and validation /
test are untouched, so results stay comparable. Validation only on WeedsGalore means the best
epoch is chosen for the drone view, although Sudars (ground photos) is most of the training set
(1438 of 1866 tiles).

### `TrainingPlan` and `Stage` — the plan file (`plan.py`)

A plan is a list of stages; each stage starts from the best weights of the previous one. The
final model uses one stage:

```json
{"model_bazowy": "yolov8n.pt", "batch": 16, "cierpliwosc": 30, "seed": 0,
 "etapy": [{"nazwa": "dane-ewis1-bt", "dane": "zbior_lotnicze/dataset/data_dane_ewis1_bt.yaml",
            "epoki": 100, "lr0": 0.01, "wejscie": 1280}]}
```

`TrainingPlan.optymalizator` defaults to `"SGD"` on purpose: with Ultralytics' `"auto"` the
`lr0` from the plan is **ignored** and the library picks its own learning rate. The plan is saved
next to the result (`plan.json`), so a training run can be repeated exactly.

### `sprawdz_dane()` and `katalog_etykiet()` — checking before hours of training (`plan.py`)

`sprawdz_dane(data.yaml)` returns `(ok, problems, info)` and `trenuj_etap()` stops when `ok` is
false. It checks: class names present; every `train` / `val` folder exists and has images (a
`train` *list* is supported); every image has a label file; and **no file name appears in two
parts** (a direct leak). The oversampled copies are named `__kopiaN`, so they never collide with
validation or test names.

`katalog_etykiet(images/train)` → `labels/train` replaces the **last path component** named
`images`. An earlier text replacement of `"/images/"` failed on Windows (backslashes), and every
image looked unlabelled.

### `trenuj_etap()` and `trenuj()` — the Ultralytics call (`training.py`)

`trenuj_etap()` maps the plan onto `YOLO(weights).train(...)`: `data`, `epochs`, `imgsz`, `lr0`,
`optimizer`, `batch`, `patience` (early stopping), `device`, `seed`, optional `freeze` and extra
augmentations. Two details that cost time before they were fixed:

- `project` is passed as an **absolute** path — newer Ultralytics versions prefix a relative path
  with `runs/detect/`, and the weights ended up somewhere else;
- the output folder is read back from `model.trainer.save_dir` instead of being guessed; the
  stage result is `weights/best.pt` (best epoch on validation).

`trenuj()` runs the stages in a chain, writes `historia.json` after every stage and copies the
last `best.pt` to `<output>/model.pt`.

### `wybierz()` — CUDA → MPS → CPU (`device.py`)

Ultralytics picks only NVIDIA GPUs by itself; on an Apple Silicon Mac it would train on the CPU.
`wybierz()` returns `"0"`, `"mps"` or `"cpu"`. Importing `device.py` sets
`PYTORCH_ENABLE_MPS_FALLBACK=1` **before** PyTorch is loaded, so operations that MPS does not
support run on the CPU instead of stopping the training.

### `_dopasuj()` — matching predictions to ground truth (`evaluation.py`)

Both AP50 and recall / precision use one matching, so the numbers are always consistent. Per
image, predictions are taken **from the most confident**; each takes the not-yet-used true box of
the same class with the highest IoU (`mission.detector.iou_ramek`); IoU ≥ 0.5 = hit (TP),
otherwise false alarm (FP). The result is a list of `(confidence, hit?)` and the number of true
boxes.

Because the order is fixed by confidence, whether a prediction is a hit does not depend on any
less confident prediction. So one matching serves **every** threshold: `pr_przy_progach()` only
counts the pairs above the threshold, and recall can never grow with a higher threshold.

### `ap50()` — area under the precision–recall curve (`evaluation.py`)

Our own implementation of the Ultralytics method, because a `.hef` model has no Ultralytics
validation — this way `.pt`, `.onnx` and `.hef` are scored by identical code:

```python
oceny.sort(key=lambda r: -r[0])                     # by confidence, descending
tp = cumsum(hit);  fp = cumsum(not hit)
recall    = tp / n_true
precision = tp / (tp + fp)
mrec = [0, *recall, 1];  mpre = [1, *precision, 0]
mpre = flip(maximum.accumulate(flip(mpre)))         # envelope: precision never rises to the right
AP50 = trapezoid(interp(linspace(0, 1, 101), mrec, mpre), linspace(0, 1, 101))
```

AP50 does not depend on the confidence threshold, which is why the evaluation builds the detector
with threshold 0.001: every prediction must be on the curve. With one class AP50 = mAP50. Our
value comes out about 1.5 pp higher than Ultralytics' own validation mAP50.

### `ocen_detektor()`, `predykcje_detektora()`, `ocen_wedlug_grup()` (`evaluation.py`)

`ocen_detektor(detector, data.yaml, split, prog=0.15)` takes **any** object with
`wykryj(image) → [Detection]` — Ultralytics, ONNX Runtime, Hailo, with or without TTA:

1. `predykcje_detektora()` resolves the image and label folders from `data.yaml`, reads the YOLO
   labels into pixel boxes (`wczytaj_etykiety()`), runs the detector on every image and maps every
   class to 0 (one-class model);
2. computes `AP50`, the result at the threshold (`przy_progu`) and `wedlug_dni`.

`ocen_wedlug_grup()` reads the flight day from the file name with
`\d{4}-\d{2}-\d{2}|(?<!\d)20\d{6}(?!\d)` — WeedsGalore writes `2023-05-25`, EWIS1 `20220516`
(normalized to `2022-05-16`) — and computes recall and precision per day plus `razem` (total).
A season average hides a weak day; this is how the 25 May problem was found.

### `do_onnx()` and `przygotuj_kalibracje()` — handing over to Hailo (`export.py`)

`do_onnx()` calls Ultralytics' export with a **fixed** input (`dynamic=False`; an accelerator
needs every tensor size at compile time), graph simplification, and `--opset 11` from
`final_model.sh` (what the Hailo compiler recommends). The output shape must be
`(1, 5, 33600)`: 4 box values + 1 class score for each of `160² + 80² + 40² = 33 600` anchor
points (strides 8, 16, 32 on a 1280 input).

`przygotuj_kalibracje(folder, output, liczba=128, rozmiar=1280)` picks images with a seeded
shuffle and writes them **letterboxed** exactly as at run time: scaled to fit 1280 × 1280 with
proportions kept, the rest filled with grey 114. INT8 quantization measures value ranges on these
images, so they must look like real network input and come from the target distribution — only
**training** tiles (WeedsGalore + EWIS1), never test tiles. It also writes `kalibracja.txt` with
the list of files.

---

Tests: `tests/test_models.py`, `tests/test_season.py`, `tests/test_hailo.py` (AP50 of any
detector). Previous step: [`normalization/PIPELINE.md`](../normalization/PIPELINE.md).
Next: [`scripts/HAILO_PIPELINE.md`](../scripts/HAILO_PIPELINE.md) and
[`mission/PIPELINE.md`](../mission/PIPELINE.md).
