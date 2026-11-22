# mission — weed detection during the flight and the weed map

**Part of the pipeline:** deployment (runs on the drone; uses the model from `models/`).

On the drone a single program runs: **`fieldeye-detect`** (a binary on Raspberry Pi 5 +
Hailo-8). It watches the folder where the drone saves photos, detects weeds on every new
photo and — using the GeoJSON telemetry saved next to the photo — turns the detections into
coordinates (one point per weed). Full manual (options, telemetry format, outputs, binary,
Raspberry Pi): [`docs/fieldeye-detect.md`](../docs/fieldeye-detect.md).
How it works inside — flow, technologies, key functions explained: [`PIPELINE.md`](PIPELINE.md).

```bash
python -m mission.detect_cli PHOTOS RESULTS models/weights/fieldeye_final.pt      # from source
./fieldeye-detect PHOTOS RESULTS fieldeye_final_1280_h8.hef                        # binary on the Pi
python -m mission.drone_sim flight/ --ewis1 zbior_ewis1_podzial/dataset/images/test  # simulated drone
```

---

## Mapping does not need a mosaic

To put a weed on the map **no stitching is needed** — the GPS position, heading and altitude
of that one photo are enough:

```
pixel in the photo
   → offset from the photo centre in pixels
   → × GSD (metres per pixel, from altitude and camera optics)
   → rotation by the drone heading
   → added to the GPS position
   → geographic coordinates
```

So every photo is processed on its own, right after it is taken, with constant memory use,
and results are written after every photo (a power loss does not erase the mission).

---

## Detectors

Interchangeable — the rest of the program does not know which model it uses. All classes are
in `detector.py` and return the same `Detection` objects.

| Class | Used for |
|---|---|
| `YoloDetector` | `.pt` model through Ultralytics (laptop) |
| `OnnxDetector` | `.onnx` model through ONNX Runtime — no PyTorch (light binary) |
| `HailoDetector` | `.hef` model on the Hailo-8 accelerator; does its own letterbox and decodes Hailo's NMS output |
| `TiledDetector` | wrapper: cuts the photo into tiles, maps boxes back, removes duplicates from tile overlaps (NMS) |
| `FlipTTADetector` | wrapper: original + horizontal flip, merged by weighted box fusion (TTA) |
| `DummyDetector` | **not a weed detector** — finds green blobs; only to test the pipeline without a model (class `atrapa`) |

`zbuduj_z_plikow(model, tta=...)` picks the class from the file extension (`.hef` → Hailo,
`.onnx` → ONNX Runtime, otherwise Ultralytics). `iou_ramek()` is the single IoU function of the
project (also used by `models/evaluation.py`).

**Why tiles.** A camera photo has several thousand pixels, the network input is 1280 px.
Shrinking the whole photo makes a seedling of a dozen pixels disappear. A tile is chosen to
cover about **1.5 m of field** — the scale the model was trained on (Arducam 64MP, 16 MP mode,
5 m: about 950 px; `stitchlive/camera.py`).

---

## Files

| File | Role |
|---|---|
| `detect_cli.py` | the `fieldeye-detect` program: waits for complete new photos (FIFO by write time), runs the detector, reads telemetry, updates the map, writes photos with boxes and a terminal line per photo |
| `spray_map.py` | telemetry: reading and sanity checks, GSD from altitude, pixel → metres → degrees in a local tangent plane (no `pyproj`), `MapaOprysku` merges detections into one point per weed and writes `oprysk.geojson`, `oprysk.csv`, `chwasty.geojson`, `trasa.geojson`; `porownaj()` scores the weed points against ground truth |
| `drone_sim.py` | simulated drone: flies over a field image (EWIS1 photo rebuilt from tiles, any image or a synthetic field) and writes photos + telemetry in exactly the format of the real drone, plus the true weed positions |
| `detector.py` | the detectors above, `letterbox()`, Hailo NMS decoding, WBF, `iou_ramek()` |
| `sim_flight.py` | helpers for tab 4 of `app.py`: field from tiles of one day, true boxes in a frame, merging repeated boxes on the mosaic, comparison with ground truth; EWIS1 photo rebuilt from its tiles (used by `drone_sim --ewis1`) |

---

## Limitations

- **Nadir camera and flat field.** Roll/pitch from the autopilot are only used to reject photos above a limit.
- **Accuracy is limited by GPS**, not by the computation: about 1 m without RTK, a few cm with RTK.
- **Altitude must be above the field**, not above sea level.
- **No lens distortion correction**; with a wide lens the error grows towards the photo edges.

Tests: `tests/test_detect_cli.py`, `test_spray_map.py`, `test_binary.py`, `test_hailo.py`,
`test_inference.py`, `test_sim_tiles.py`, `test_gui_flight.py`.
