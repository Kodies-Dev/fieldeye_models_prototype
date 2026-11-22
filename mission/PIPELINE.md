# Detection and spray-map pipeline — how it works inside

How `fieldeye-detect` turns photos arriving from the drone into weed coordinates and spray
points, and how the simulated drone feeds it on a laptop. Usage, options, telemetry format and
outputs are in [`docs/fieldeye-detect.md`](../docs/fieldeye-detect.md) and
[`README.md`](README.md); this page explains the flow and the hardest parts of the code.

**Input:** a folder the camera writes to — photo + `.geojson` telemetry with the same name.
**Output:** photos with boxes, `oprysk.geojson`, `oprysk.csv`, `chwasty.geojson`,
`trasa.geojson`. **Runs on:** Raspberry Pi 5 + Hailo-8 (binary with `.hef`), or any computer
(`.pt` / `.onnx`).

---

## 1. Flow

```
detect_cli.main()
│  zbuduj()  →  TiledDetector( [FlipTTADetector(] YoloDetector | OnnxDetector | HailoDetector [)] )
│  MapaOprysku(promien_scalania_m=0.10)
│
└─ loop every --interwal 0.5 s
     kandydaci()            new photos, FIFO by write time
     gotowy() / kompletny() size stable ≥ 1 s and the file ends correctly
     plik_telemetrii()      IMG.geojson next to IMG.jpg (wait up to --czekaj-telemetrie 3 s)
     czytaj_telemetrie()    → lat, lon, agl, heading, roll / pitch, fix ...
     blad_telemetrii()      sanity checks; a bad file → the photo goes "bez GPS"
     gsd_m()                metres per pixel (telemetry, or altitude + camera model)
     tile = 1.5 m / GSD     the model sees weeds at its training scale
     przetworz()            imread → det.wykryj() → boxes ≥ threshold, sorted by confidence
     przechyl_st()          tilt > --maks-przechyl 5° → photo left off the map
     MapaOprysku.dodaj_zdjecie()   box centre → metres → degrees → merge into spray points
     MapaOprysku.zapisz()   4 output files, atomically, after EVERY photo
     linia()                one line in the terminal; photo with boxes → RESULTS/ (atomic)
   zamknij()                releases the Hailo device at the end

laptop only
  drone_sim.main()   field image → stitchlive.flight_sim.rozloz() → photo + telemetry every 0.7 s
                     + *_prawda.geojson (true weed positions)
  spray_map.porownaj(oprysk.geojson, prawda.geojson)  → hits, misses, distances
```

No mosaic is needed: one photo's GPS position, heading and altitude are enough to place its
weeds, so every photo is processed on its own with constant memory, and a power loss does not
erase what was already written.

---

## 2. Technologies

| Technology | Used for | Where |
|---|---|---|
| **HailoRT** (`hailo_platform`) | the `.hef` model on the Hailo-8 | `HailoDetector` — see [`scripts/HAILO_PIPELINE.md`](../scripts/HAILO_PIPELINE.md) |
| **ONNX Runtime** | the `.onnx` model without PyTorch (light binary) | `OnnxDetector` |
| **Ultralytics** | the `.pt` model on a laptop | `YoloDetector` |
| **OpenCV** | reading / writing photos, letterbox, `cv2.dnn.NMSBoxes`, drawing boxes, green-blob dummy detector, affine maths of the simulator | `detector.py`, `detect_cli.py`, `drone_sim.py`, `sim_flight.py` |
| **NumPy** | box arrays, decoding network output | `detector.py` |
| Python `math`, `json`, `pathlib` | WGS84 local plane (no `pyproj`), GeoJSON, atomic writes (temporary file + rename) | `spray_map.py` |
| **GeoJSON** | telemetry input and all map outputs | `spray_map.py`, `drone_sim.py` |
| **PyInstaller** | single-file binary `fieldeye-detect` | `scripts/build_binary.sh`, `scripts/fieldeye_detect_main.py` |
| `stitchlive` | Arducam camera model (GSD from altitude), flight simulator for `drone_sim` | `stitchlive/camera.py`, `stitchlive/flight_sim.py` |

Heavy libraries are imported only inside the class that needs them, so the ONNX binary contains
neither PyTorch nor Ultralytics (`tests/test_binary.py` checks this).

---

## 3. Files in this process

| File | Role in the flow |
|---|---|
| `detect_cli.py` | the program: options, the watch loop, telemetry, map updates, outputs |
| `detector.py` | all detectors and wrappers, `zbuduj_z_plikow()`, `letterbox()`, Hailo NMS decoding, WBF, `iou_ramek()` |
| `spray_map.py` | telemetry reading and checks, GSD, pixel → metres → degrees, `MapaOprysku`, `porownaj()` |
| `drone_sim.py` | simulated drone writing photos + telemetry like the real one |
| `sim_flight.py` | helpers for GUI tab 4 and `drone_sim --ewis1`: fields from tiles, ground truth in a frame, box merging |
| `scripts/fieldeye_detect_main.py` | binary entry point: detection, `dron`, `porownaj` |

---

## 4. Key classes and functions

### `kandydaci()`, `gotowy()`, `kompletny()` — never read half a photo (`detect_cli.py`)

The camera may still be writing a file when the program sees it. Three guards:

- `kandydaci()` lists photos not yet processed, sorted by **modification time** (then name), so
  the order does not depend on how the camera names files; hidden and temporary names
  (`.img.jpg`, `~img.jpg`) are skipped.
- `gotowy(p, stan)` remembers `(size, since)` per file and returns `True` only when the size has
  not changed for `--stabilnosc` (1 s) — also when the writer pauses.
- `kompletny(p)` reads the last 64 bytes: a JPEG must end with the marker `FF D9` (trailing
  `00` padding ignored), a PNG must contain `IEND` in its last 12 bytes.

Results are written the same way the other side should write: to a hidden temporary file, then
`replace()` — a reader never sees half a file.

### `zbuduj()` and `zbuduj_z_plikow()` — the detector chain (`detect_cli.py`, `detector.py`)

Every detector has one method, `wykryj(image_bgr) → list[Detection]` (box in photo pixels, class,
confidence), so wrappers can be stacked:

```
TiledDetector  ─►  FlipTTADetector (--tta)  ─►  HailoDetector  (.hef)
                                               OnnxDetector   (.onnx with onnxruntime installed)
                                               YoloDetector   (anything else, e.g. .pt)
```

`_detektor_z_pliku()` chooses by file extension. With `--tta` each single pass runs at
`min(threshold, 0.05)`, because the fusion lowers scores and filters again afterwards. The tile
size starts at 950 px (1.5 m of field for the Arducam 16 MP at 5 m) and is set from the telemetry
for every photo: `tile = max(64, round(1.5 / GSD))`.

### `TiledDetector` — full resolution, then duplicates removed (`detector.py`)

Shrinking a 4624 × 3472 photo to the 1280 network input would make a seedling of a dozen pixels
disappear. The photo is cut into tiles with 15% overlap — corners computed like in
`normalization.steps.kafelkuj()`, the last tile pushed back to the edge. Each tile's boxes are
shifted back by the tile corner, then `_scal()` removes duplicates from the overlaps with a
greedy NMS: from the most confident, a box is dropped if it overlaps an already kept box of the
same class with IoU ≥ 0.45.

### `FlipTTADetector` and `polacz_wbf()` — test-time augmentation (`detector.py`)

Seen from above a weed looks the same mirrored, but the network sees different pixels and
sometimes finds it in only one version. The detector runs on the original and on the horizontally
flipped image, and flips the boxes back: `x0' = w − x1`, `x1' = w − x0`.

`polacz_wbf(lists, iou=0.55)` — **weighted box fusion** — merges the passes:

1. every detection joins the first existing group (groups taken by their highest confidence) of
   the same class whose current average box has IoU ≥ 0.55 with it; otherwise it starts a group;
2. a group keeps the confidence-weighted sum of coordinates and, per pass, the highest confidence;
3. result per group: box = weighted average; confidence = sum of the per-pass maxima / number of
   passes.

A weed found in only one of two passes gets half its confidence — that is why TTA is judged
mainly by AP50 (threshold-independent). Effect on the final model: AP50 +2.2–2.8 pp and about
30% fewer false alarms.

### `OnnxDetector` — the model without PyTorch (`detector.py`)

The same network exported by `models.cli onnx`. Output `(1, 5, N)` is transposed to `N` rows
`[cx, cy, w, h, score]` in network-input pixels. Steps: `letterbox()` → `/255`, CHW → session
(CUDA provider if available, else CPU) → threshold → `cv2.dnn.NMSBoxes` with the same settings as
Ultralytics at prediction (IoU 0.7, at most 300 boxes) → undo the letterbox
(`X = (x − px) / s`) and clip to the photo.

`HailoDetector` does the same with the chip doing the NMS — see
[`scripts/HAILO_PIPELINE.md`](../scripts/HAILO_PIPELINE.md).

### `DummyDetector` — not a weed detector (`detector.py`)

Exists only to test the whole chain before a model is available. Excess-green index
`ExG = 2G − R − B`, pixels above the 97th percentile, morphological opening 3 × 3, connected
components between 120 px and 2% of the image. Class `atrapa` and a pseudo-confidence 0.5–0.9
from blob size, so it can never be mistaken for a weed.

### `czytaj_telemetrie()` and `blad_telemetrii()` — trusting the drone's data (`spray_map.py`)

`czytaj_telemetrie()` takes the GeoJSON feature whose `properties.image` matches the photo (or the
first one): `[lon, lat]`, `survey.agl_m`, `survey.gsd_mm_per_px`, `heading_deg`, `roll_deg`,
`pitch_deg`, `fix`, `fix_quality`, `time_utc`. **Heading** comes from `heading_deg` (compass /
autopilot). Only if it is missing, `course_deg` (direction of movement from GPS) is used, and only
at ≥ 1 m/s — in hover the GPS course is noise.

`blad_telemetrii()` returns a reason or `None`: coordinates out of range or `0, 0`, no GPS fix,
no heading, altitude outside 0.3–150 m (altitude above sea level instead of above the field?),
GSD outside 0.1–100 mm/px (wrong units?). With a reason the photo is still detected, but goes to
the map "bez GPS" (without GPS).

### `gsd_m()` and `piksel_na_metry()` — from a pixel to metres (`spray_map.py`)

Scale: `survey.gsd_mm_per_px / 1000` if given, otherwise from the camera model
(`stitchlive/camera.py`, Arducam 64MP in 16 MP mode: sensor 7.4 mm, focal length 5.1 mm):

```
GSD = agl · sensor_width / focal_length / image_width_px       5 m → 1.57 mm/px
```

`piksel_na_metry(px, py, W, H, gsd, heading)` — the camera points straight down, the top of the
image is the front of the drone, heading θ is clockwise from north:

```python
dx = (px - W / 2) * gsd                     # right of the photo centre
dy = (py - H / 2) * gsd                     # down from the photo centre
east  =  dx * cos(θ) - dy * sin(θ)
north = -dx * sin(θ) - dy * cos(θ)
```

Check: θ = 0 → east = dx, north = −dy (image down = south); θ = 90° (flying east) → the top of
the image points east, so `dy < 0` (above the centre) gives a positive east.

### `UkladLokalny` — metres ↔ degrees without `pyproj` (`spray_map.py`)

A plane tangent to the WGS84 ellipsoid at the first photo. With `a = 6 378 137 m`,
`e² = 0.006 694 38` and latitude φ₀:

```
M = a (1 − e²) / (1 − e² sin² φ₀)^(3/2)        radius of curvature north–south
N = a / √(1 − e² sin² φ₀)                       radius of curvature east–west
metres per degree:   latitude  M · π/180       longitude  N · cos φ₀ · π/180
```

At 50° N that is about 111 235 m and 71 260 m per degree. On a field of a few hundred metres
the error of this approximation is a fraction of a millimetre — far below the GPS error — and
the binary needs no projection library.

### `MapaOprysku` — from detections to spray points (`spray_map.py`)

The same weed is seen on several overlapping photos. `dodaj_zdjecie(tele, name, W, H, boxes, gsd)`:

1. the first photo sets the local plane; the drone position goes to the route;
2. every box centre → `piksel_na_metry()` → added to the drone position → degrees; the detection
   is stored with its size (`max(w, h) · gsd`) and, if roll / pitch are known, a position
   uncertainty `agl · tan(tilt)`;
3. `_scal()` looks for the **nearest** spray point within `promien_scalania_m` (0.10 m). Found →
   the detection joins it; the point keeps `Σw`, `Σw·east`, `Σw·north` (w = confidence), so its
   position is the confidence-weighted mean, plus max confidence, max size, count and photo set.
   Not found → a new point.

`punkty_oprysku()` gives each point a spray radius = half of its largest weed + 5 cm margin.
`zapisz()` writes `oprysk.geojson` (points), `oprysk.csv` (the same as a table), `chwasty.geojson`
(every detection) and `trasa.geojson` (route line + photo positions), all atomically.

### `porownaj()` — how accurate are the spray points (`spray_map.py`)

For a simulated flight, where the true weed positions are known. Both point sets go into one local
plane; all pairs closer than the radius (0.10 m) are sorted by distance and matched greedily from
the closest, each weed and each point at most once. Result: hits, misses, false points, recall,
precision, median and 90th percentile of the distance in cm.

### `drone_sim.main()` — a drone without a drone (`drone_sim.py`)

1. Field: an EWIS1 test photo rebuilt from its tiles (`sim_flight.zdjecia_z_kafelkow()` +
   `pole_ze_zdjecia()`, with the true boxes), any image (`--pole`), or a synthetic field with
   green blobs at known places (`--sztuczne`).
2. The image is laid on the map: top-left corner at `--lat / --lon`, north up, `--gsd-mm` per
   pixel. `stitchlive.flight_sim.rozloz()` plans a lawn-mower route and cuts the frames.
3. For every frame: centre pixel → `east = x · g`, `north = −y · g` (image y points south) → plus
   optional GPS noise → lat / lon; heading = the frame's rotation; GSD = source GSD ·
   frame footprint / frame width.
4. Every `--interwal` s it writes the photo and then the telemetry — both atomically, with the
   real drone's names (`20261109T142859.998Z.jpg` + `.geojson`) and fields; roll and pitch 0.
5. The true weeds visible on at least one photo go to `FOLDER_prawda.geojson` for `porownaj()`.

### `sim_flight.py` — fields and ground truth for GUI tab 4

| Function | What it does |
|---|---|
| `zdjecia_z_kafelkow()` | groups tiles named `..._k{x}_{y}` (tile corner in the name) by source photo |
| `pole_ze_zdjecia()` | puts every tile back where it was cut — the photo rebuilt pixel-exact; boxes from both sides of an overlap are merged by `_scal_powtorzone()` (the smaller box goes when ≥ 60% of it lies inside the larger) |
| `pole_z_kafelkow()` | glues up to 25 WeedsGalore tiles of **one** day into a grid (days must not be mixed: the same place twice confuses the stitcher) |
| `prawda_w_kadrze()` | true boxes from the field into a frame: inverse of the frame's affine matrix (`cv2.invertAffineTransform`), kept when ≥ 40% visible |
| `scal_ramki()` | the same weed from several frames on the mosaic → one box (IoU ≥ 0.3, most confident kept) |
| `porownaj()` | hits / misses / false boxes of one image through `models.evaluation.pr_przy_progach()` |

---

Tests: `tests/test_detect_cli.py`, `test_spray_map.py`, `test_binary.py`, `test_inference.py`,
`test_hailo.py`, `test_sim_tiles.py`, `test_gui_flight.py`.
Previous steps: [`models/PIPELINE.md`](../models/PIPELINE.md),
[`scripts/HAILO_PIPELINE.md`](../scripts/HAILO_PIPELINE.md).
