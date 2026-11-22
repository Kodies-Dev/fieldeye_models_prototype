# Hailo-8 pipeline — how it works inside

How the ONNX model becomes `fieldeye_final_1280_h8.hef` and how that file runs on the
Raspberry Pi 5 with the AI HAT+ (Hailo-8, 26 TOPS). The step-by-step commands, versions and
fixes are in [`HAILO.md`](HAILO.md); this page explains what happens inside each step.

**Input:** `hailo/hailo_final.zip` from `scripts/final_model.sh` (ONNX 1280, 128 calibration
images, `compile_hef.py`). **Output:** `.hef` (5.4 MB) + its accuracy on the Pi.
**Runs on:** Mac → Linux x86 / WSL (compiler) → Raspberry Pi (HailoRT).

---

## 1. Flow

```
Mac       models.cli onnx + kalibracja  →  hailo_final.zip
          (fieldeye_final_1280.onnx, kalibracja_1280/*.jpg, compile_hef.py)
            │
WSL       compile_hef.py  (Dataflow Compiler 3.33 + Model Zoo 2.17)
            konfiguracja_nms()       Model Zoo yolov8n_nms_config.json → our NMS settings
            skrypt_modelu()          Model Zoo yolov8n.alls with the path to our NMS file
            ClientRunner(hw_arch="hailo8")
              1. translate_onnx_model()   ONNX → Hailo graph, cut at the 6 last Conv layers  → .parsed.har
              2. load_model_script()      /255 normalization, output activations, NMS
                 optimize(calibration)    FP32 → INT8 on the calibration images             → .quantized.har
              3. compile()                allocation on the chip                            → .hef + .compiled.har
            │
Pi        check_hef.sh
            hailortcli fw-control identify | parse-hef | run        accelerator, file, speed
            models.cli ocen-detektor X.hef  →  HailoDetector  →  the same AP50 code as for .pt
            comparison with the .pt (AP50 0.812 / 0.817)
            │
Pi        fieldeye-detect PHOTOS RESULTS X.hef  →  TiledDetector(HailoDetector)   in flight
            HailoDetector.wykryj():  letterbox() → infer on Hailo-8 → nms_hailo_na_detekcje()
```

---

## 2. Technologies

| Technology | Version | Used for | Where |
|---|---|---|---|
| **Hailo Dataflow Compiler** (`hailo_sdk_client.ClientRunner`) | 3.33.0 | parse, INT8 quantization, compilation to HEF | `compile_hef.py` (Linux x86 only) |
| **Hailo Model Zoo** | 2.17.0 | ready model script `yolov8n.alls` and NMS template `yolov8n_nms_config.json` | `compile_hef.py` |
| **HailoRT** (`hailo_platform`, `hailortcli`) | 4.23.0 on the Pi | loading the HEF, streams to and from the chip, CLI checks | `mission/detector.py`, `check_hef.sh` |
| **ONNX** | opset 11 | exchange format between Ultralytics and the compiler | `models/export.py` |
| **OpenCV, NumPy** | — | calibration images as a tensor, letterbox, box conversion | `compile_hef.py`, `mission/detector.py` |

The compiler must come from the same release as the HailoRT on the Pi (3.33 ↔ 4.23); a newer
compiler produces a HEF the Pi rejects (`Unsupported hef version`).

---

## 3. Files in this process

| File | Role in the flow |
|---|---|
| `models/export.py` | `do_onnx()`, `przygotuj_kalibracje()` — the input package |
| `scripts/compile_hef.py` | the compilation (WSL) |
| `scripts/check_hef.sh` | the check on the Pi |
| `mission/detector.py` | `HailoDetector`, `letterbox()`, `nms_hailo_na_detekcje()` — running the HEF |
| `models/evaluation.py` | the same `ocen_detektor()` for `.pt` and `.hef` |
| `scripts/HAILO.md` | commands, versions, error fixes |

---

## 4. Key functions and classes

### Why `compile_hef.py` and not `hailomz compile`

Model Zoo's ready YOLOv8n recipe is for 640 px COCO: NMS for a 640 image, 80 classes, score
threshold 0.2, at most 100 boxes per class. For our model that is wrong three times — boxes would
be scaled for the wrong image size, every weed below 0.2 would disappear on the chip (our working
threshold is 0.15, the TTA pass uses 0.05), and 100 boxes are too few on a dense tile. The script
reuses Model Zoo's files and changes only what has to change.

### `konfiguracja_nms()` and `skrypt_modelu()` — adapting Model Zoo's files

`konfiguracja_nms(template, 1280)` loads `yolov8n_nms_config.json` and sets:

| Field | Model Zoo | Ours |
|---|---|---|
| `image_dims` | 640 × 640 | 1280 × 1280 |
| `classes` | 80 | 1 |
| `nms_scores_th` | 0.2 | 0.05 |
| `nms_iou_th` | template value | 0.6 |
| `max_proposals_per_class` | 100 | 300 |

The result is saved as `<name>_nms.json`. `skrypt_modelu()` takes `yolov8n.alls` (the model
script: input normalization `/255`, output activations, `nms_postprocess(...)`) and replaces the
path inside `nms_postprocess("...")` with ours — the regex must match **exactly once**, otherwise
the script stops instead of compiling a model with the wrong NMS.

### Parse: `translate_onnx_model()` cut at six layers

```python
KONCOWE = ["/model.22/cv2.0/cv2.0.2/Conv", "/model.22/cv3.0/cv3.0.2/Conv",
           "/model.22/cv2.1/cv2.1.2/Conv", "/model.22/cv3.1/cv3.1.2/Conv",
           "/model.22/cv2.2/cv2.2.2/Conv", "/model.22/cv3.2/cv3.2.2/Conv"]
```

`/model.22` is the YOLOv8 detection head. It has three scales (strides 8, 16, 32 → 160², 80²,
40² cells on a 1280 input); at every scale `cv2` predicts the box (4 sides × 16 distribution bins
= 64 channels) and `cv3` the class score (1 channel). The ONNX graph after these convolutions —
box decoding (DFL), sigmoid, concatenation — is **cut off**: the compiler builds the same decoding
plus NMS itself, on the chip, from `nms_postprocess` in the model script.

The network is named **`yolov8n`** because the NMS config refers to layer names such as
`yolov8n/conv41`; a different name gives `The layer named yolov8n/conv41 doesn't exist`.

### Optimize: `zdjecia_kalibracyjne()` and INT8 quantization

`zdjecia_kalibracyjne(folder, 1280, 128)` loads the calibration images into one array
`(N, 1280, 1280, 3)`, float32, **RGB, 0–255** (OpenCV reads BGR, so it converts) — the `/255` is
inside the model script, so the images must not be normalized twice. An image of another size
stops the script with a hint (`models.cli kalibracja --rozmiar 1280`).

`runner.optimize(images)` runs the images through the network, records the value range of every
layer and converts weights and activations from 32-bit floats to 8-bit integers. Without a GPU
and with fewer than 1024 images the compiler uses optimization level 0 (64 images, basic
quantization) — it works, but may lose more accuracy than full optimization.

### Compile

`runner.compile()` allocates the layers on the Hailo-8 and returns the bytes of the `.hef`. The
final model fits in one context (47% compute, 28% memory). A `.har` file is saved after each of
the three stages, so a failed compilation can be inspected without repeating the whole run.

### `HailoDetector` — running the HEF (`mission/detector.py`)

Opening the accelerator is slow (hundreds of ms), so everything is done **once** in `__init__`:

```python
hef  = HEF(path);  dev = VDevice()
ng   = dev.configure(hef, ConfigureParams.create_from_hef(hef, interface=PCIe))[0]
stream = InferVStreams(ng, InputVStreamParams.make(ng, format_type=UINT8),
                       OutputVStreamParams.make(ng, format_type=FLOAT32))
stream.__enter__();  ng.activate(...).__enter__()
```

The input size (1280 × 1280) is read from the HEF itself. `wykryj(image)`:

1. `letterbox(image, 1280, 1280)` — scale `s = min(1280/w, 1280/h)`, resize, centre on a grey
   (114) canvas, BGR → RGB; returns the image, `s` and the padding `px, py`. Same as Ultralytics
   during training and as the calibration images;
2. `stream.infer({input: image[None]})` — one tile through the chip, **uint8** input;
3. `nms_hailo_na_detekcje()` — conversion of the chip's output.

`zamknij()` exits the two contexts. Only one process can hold the Hailo device, so without it the
next program would not get the accelerator; `fieldeye-detect` and `ocen-detektor` call it at the
end.

### `nms_hailo_na_detekcje()` — reading the chip's NMS output

The HEF output is "NMS by class": a list with one array per class, each row
`[y0, x0, y1, x1, score]` as **fractions of the network input** — note **y before x**, the
opposite of Ultralytics. Back to pixels of the original photo, undoing the letterbox:

```python
X0 = (x0 * 1280 - px) / s        Y0 = (y0 * 1280 - py) / s       # same for X1, Y1
```

then clipped to the photo; boxes lying entirely in the grey padding disappear.

### Threshold layers

| Where | Threshold | Effect |
|---|---|---|
| NMS on the chip (`compile_hef.py`) | 0.05 | anything below never leaves the HEF |
| `HailoDetector(prog=...)` | 0.15 in flight, 0.001 in the evaluation | filters what the chip returned |

Because the chip already drops everything below 0.05, the AP50 of the HEF is computed from a
slightly shorter list than for the `.pt` (threshold 0.001) — a difference of about 0.01 can come
from this alone.

### `check_hef.sh` — is the compiled model still as good?

Five steps: `hailortcli fw-control identify` (accelerator and firmware answer), `parse-hef`
(input 1280 × 1280 × 3, output with NMS, file accepted by this HailoRT), `hailortcli run` (tiles
per second), then `models.cli ocen-detektor` on the EWIS1 test (132 tiles, threshold 0.15) without
and with TTA → `wyniki_hef/hef.json`, `hef_tta.json`, and a table against the `.pt`:

| AP50 loss vs `.pt` | Verdict |
|---|---|
| ≤ 0.02 | good |
| ≤ 0.05 | acceptable (quantization at level 0) |
| more | check the compilation and the calibration images |

The comparison is fair because it is the same evaluation code, the same test tiles and the same
threshold. Status: the first run on the Pi gave 4090 detections on the 132 tiles (4313 with the
`.pt` on a Mac), about 0.03 s per tile; the full AP50 comparison has not been run yet.

---

Tests without the compiler or the hardware: `tests/test_compile_hef.py` (NMS config, model
script, calibration array), `tests/test_hailo.py` (letterbox, NMS decoding, detector choice).
Previous step: [`models/PIPELINE.md`](../models/PIPELINE.md). Use in flight:
[`mission/PIPELINE.md`](../mission/PIPELINE.md).
