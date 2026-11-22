# Simulator and mosaic pipeline — how it works inside

How `stitchlive` cuts one large field image into drone frames with a known ground truth, stitches
frames back into a mosaic, and measures how good the mosaic is. Commands, options and measured
results are in [`README.md`](README.md); this page explains the flow and the hardest parts of the
code.

**Input:** one large image of a field (or a synthetic field), or real drone frames.
**Output:** `images/IMG_*.jpg` + `prawda.json`, the mosaic, the evaluation (`ocena.json`).
**Runs on:** a laptop — `python -m stitchlive.cli`, GUI tabs 1, 3 and 4, and the simulated
drone in `mission/drone_sim.py`.

---

## 1. Flow

```
field image  (or field.zbuduj_pole(): synthetic soil, crop rows, weeds, stones)
   │
   │  camera.opis_lotu()           (drone camera mode) footprint, frame size in source px,
   │                               1.5 m tile, motion blur, photo interval
   ▼
flight_sim.rozloz(image, out, SimSettings)
   plan_trasa() | plan_losowy()    frame centres and headings (lawn-mower route or random walk)
   for every frame:
     macierz_kadru()               2×3 matrix: frame pixel → field pixel (rotation, scale, shift)
     wytnij()                      cv2.warpAffine(..., WARP_INVERSE_MAP)
     zaklocenia()                  vignetting → exposure → motion blur → random blur → noise
     narozniki_kadru()             4 frame corners on the field → prawda.json
   → images/IMG_0001.jpg ..., prawda.json, oryginal.jpg, podglad_trasy.jpg
   │
source.FolderSource | BatchSource  (background thread) → queue → cli.petla()
   ▼
GlobalStitcher.dodaj(path)           for every frame
   _przeskaluj()                   to ~0.6 Mpx
   CLAHE → detectAndCompute()      ORB or SIFT features (second try with a lower threshold)
   _dopasuj_pare(j, new)           kNN + ratio test + RANSAC + checks → self.pary
   every renderuj_co frames:
     _rozwiaz()                    _skladowa() → [_kotwica() → _uklad() → lstsq → _blad_pary()
                                   → drop bad pairs] × ≤ 8 → self.S (a, b, tx, ty per frame)
     _renderuj()                   warpAffine per frame → GainCompensator → [GraphCut seams]
                                   → MultiBandBlender → self.mozaika
GlobalStitcher.zakoncz()           final solve + render
   ▼
evaluation.ocen(mosaic, original, prawda)
   _obszar_lotu() → _szary() → SIFT / ORB + _pary_wzajemne() → RANSAC similarity
   → warp the mosaic onto the original → coverage, mean error, PSNR → werdykt()
```

Alternative stitchers with the same `dodaj()` interface: `DetailStitcher` (`cv2.detail`
pipeline) and `BasicStitcher` (`cv2.Stitcher`) — kept for comparison.

---

## 2. Technologies

| Technology | Used for | Where |
|---|---|---|
| **OpenCV** features | `ORB_create`, `SIFT_create`, `BFMatcher.knnMatch` / `match`, CLAHE | `global_stitcher.py`, `evaluation.py` |
| **OpenCV** geometry | `estimateAffinePartial2D` with RANSAC (similarity: rotation + uniform scale + shift), `warpAffine`, `invertAffineTransform` | everywhere |
| **OpenCV** `cv2.detail` | `GainCompensator` / `BlocksGainCompensator` (exposure), `GraphCutSeamFinder` (seams), `MultiBandBlender` (blending); the whole detail pipeline in `detail.py` | `global_stitcher.py`, `detail.py` |
| **OpenCV** `cv2.Stitcher` | SCANS mode, the reference method | `basic.py` |
| **NumPy** | the linear system and `np.linalg.lstsq`, masks, PSNR, random generator | `global_stitcher.py`, `flight_sim.py`, `evaluation.py` |
| Python `threading`, `queue` | reading a folder without blocking the stitching | `source.py`, `cli.py` |
| `dataclasses`, JSON | `SimSettings`, `CameraModel`; `prawda.json`, `raport.json`, `ocena.json` | `flight_sim.py`, `camera.py` |

`opencv-contrib-python` is required: the plain `opencv-python` has no `cv2.detail`.

---

## 3. Files in this process

| File | Role in the flow |
|---|---|
| `flight_sim.py` | the simulator: `SimSettings`, route planning, frame cutting, disturbances, `rozloz()` |
| `camera.py` | drone camera model (Arducam 64MP, Raspberry Pi HQ): footprint, GSD, frame and tile size, blur |
| `field.py` | `zbuduj_pole()` — synthetic field with enough texture to stitch |
| `source.py` | `FolderSource` (watches a folder in a thread), `BatchSource` (existing folder) |
| `global_stitcher.py` | `GlobalStitcher` — the default stitching method |
| `detail.py`, `basic.py` | `DetailStitcher`, `BasicStitcher` — comparison methods |
| `evaluation.py` | `ocen()`, `werdykt()` — mosaic vs. original |
| `cli.py` | `batch`, `watch`, `info`, `rozloz`, `ocen`, `demo`; `petla()` — the queue loop |

---

## 4. Key classes and functions

### `macierz_kadru()` — one matrix describes a drone photo (`flight_sim.py`)

A frame is the piece of field under the drone: centre `(cx, cy)`, heading `kat_st`, footprint
`zasieg` (field pixels across the frame — larger = flying higher). Simple slicing
`image[y0:y1, x0:x1]` cannot rotate or scale, so the frame is described by a 2 × 3 affine matrix
that maps **frame pixel → field pixel**:

```
s = zasieg / kadr          a = s · cos(kat)          b = s · sin(kat)

x_field = a·x − b·y + tx       tx = cx − a·w/2 + b·h/2
y_field = b·x + a·y + ty       ty = cy − b·w/2 − a·h/2
```

`tx, ty` are chosen so that the frame centre `(w/2, h/2)` lands on `(cx, cy)`. `wytnij()` passes
the matrix to `cv2.warpAffine` with `WARP_INVERSE_MAP` — for every output pixel OpenCV looks up
where to sample in the field — and `INTER_AREA`. `narozniki_kadru()` multiplies the four frame
corners by the same matrix: those are the ground-truth corners in `prawda.json`. The same
`(a, b, tx, ty)` form is what the stitcher solves for, so the truth and the result are directly
comparable.

### `marginesy()` and `plan_trasa()` — planning the flight (`flight_sim.py`)

A rotated frame, or one taken higher, reaches further than half its side, so frame centres must
stay away from the image edge. `marginesy()` takes the worst realistic case (3 standard
deviations): angle `3·obrot` (at most 45°), footprint `zasieg · (1 + 3·skala)` (at most ×1.33), and
the half-size of the box around the rotated rectangle (`w|cos| + h|sin|`, `w|sin| + h|cos|`).

`plan_trasa()` — lawn-mower route:

```
step along the line  = zasieg · (1 − overlap)               e.g. 480 · 0.35 = 168 px
step between lines   = frame height on the field · (1 − sidelap)
```

Lines go from margin to margin, every other line in the opposite direction (with `obrot_linii` the
heading turns 180°, as a real drone does), and every point gets random jitter. Example: a
2400 × 1600 field, frame 480, overlap 0.65, sidelap 0.5 → 11 frames × 5 lines = 55 frames.
`plan_losowy()` is a random walk with the overlap kept between consecutive frames.

### `zaklocenia()` and `rozloz()` — realistic frames with a known truth (`flight_sim.py`)

`zaklocenia()` applies the disturbances in the order of a camera's image path, each only when its
setting is above 0: **vignetting** (`× (1 − strength · r²)`), **exposure** (one random gain per
frame), **motion blur** from flight physics (`speed · exposure time / GSD` px, every frame),
**random blur** (5–12 px streak with a given probability — vibration, gust), **sensor noise**
(Gaussian per pixel and channel). Calculations go through float32 and `clip(0, 255)`, so values
never wrap around in uint8.

`rozloz()` ties it together: a seeded generator (same seed = same frames), the plan, per frame a
heading `kurs + N(0, obrot)` and a footprint `zasieg · (1 + N(0, skala))` clipped to the same
±3σ as the margins, then `macierz_kadru → wytnij → zaklocenia → IMG_xxxx.jpg`. Every frame's
entry in `prawda.json`: file, centre, angle, footprint, frame size, line number and corners.

### `CameraModel` — what the real camera sees (`camera.py`)

For a camera pointing straight down, the footprint follows from similar triangles:

```
footprint = sensor_size / focal_length · altitude      Arducam 64MP, 5 m: 7.4 / 5.1 · 5 = 7.25 m
GSD       = footprint / image_width_px                 7.25 m / 4624 px = 1.57 mm/px
```

`kadr_w_zrodle()` converts the footprint into pixels of the source image (WeedsGalore 2.5 mm/px →
2902 × 2177 px); `kafelek_px()` gives the tile covering 1.5 m (600 source px), the size of a
training tile, so weeds reach the network at the trained scale. `rozmycie_px()` is motion blur
`v · t / GSD`, `odstep_zdjec_s()` the photo interval for a given overlap
(`along_track · (1 − overlap) / speed`). `opis_lotu()` returns all of these for the GUI and the CLI.

### `GlobalStitcher.dodaj()` and `_dopasuj_pare()` — reliable frame pairs (`global_stitcher.py`)

`dodaj(path)`: shrink to `work_megapix` (scale `√(0.6 Mpx / pixels)`), CLAHE on the grey image
(local contrast in an 8 × 8 grid, clip limit 2 — lifts the dark vignetted corners where frames
overlap), features + descriptors. Fewer than 200 points → a second, more sensitive detector
(ORB `fastThreshold=5` / SIFT `contrastThreshold=0.01`) for blurred frames; fewer than 15 → the
frame is skipped. The new frame is then matched with every earlier frame (or the last
`zakres_par`).

`_dopasuj_pare(i, j)` — one wrong pair can tear the whole mosaic, so four sieves:

1. `knnMatch(k=2)` — the two most similar points in frame j for every point of frame i
   (Hamming distance for ORB, L2 for SIFT);
2. **ratio test** — keep only if the best is clearly better than the second (`< 0.75 ×`);
   repetitive crop rows produce ambiguous matches;
3. **RANSAC** `estimateAffinePartial2D` — the rotation + scale + shift that most pairs agree with
   within 3 px;
4. **checks** — at least 15 agreeing pairs, at least 20% of the ratio-test pairs, and scale between
   0.7 and 1.4 (a drone does not change altitude by 40% between two photos).

At most 40 evenly spaced agreeing points per pair are stored — more does not improve the solution
and only slows it.

### `_rozwiaz()` — all frame positions from one linear system (`global_stitcher.py`)

The heart of the method. Every frame `k` gets four unknowns — a similarity transform onto the
mosaic:

```
S_k(p) = [a −b] p + [tx]        a = s·cos(angle),  b = s·sin(angle)
         [b  a]     [ty]
```

Written with `a, b` instead of angle and scale, **the transform is linear in its unknowns**. Every
point pair (p on frame i, q on frame j, the same spot on the field) says "both land in the same
place on the mosaic", `S_i(p) − S_j(q) = 0` — two linear equations:

```
x:  a_i·x1 − b_i·y1 + tx_i − (a_j·x2 − b_j·y2 + tx_j) = 0
y:  b_i·x1 + a_i·y1 + ty_i − (b_j·x2 + a_j·y2 + ty_j) = 0
```

All pairs together give a system `A · x = b` with many more equations than unknowns (55 frames,
128 pairs → 8222 × 216), solved at once by least squares (`np.linalg.lstsq`). Because every frame
is placed against all its neighbours at the same time, **the error does not accumulate** along
the route — the reason this method beats chaining frames one by one, and the `cv2.detail`
pipeline on rotated frames.

Steps of `_rozwiaz()`:

1. `_skladowa()` — frames are graph nodes, pairs are edges; a depth-first search finds the
   **largest connected group**. Only those frames can be placed; the first of them is the reference
   `(1, 0, 0, 0)` — known, so its terms go to the right-hand side `b`, which also fixes the
   mosaic's origin and scale (without it, "everything = 0" would solve the system).
2. Up to 8 passes: `_kotwica()` → `_uklad()` → `lstsq` → `_blad_pary()` for every pair. If the
   worst pair is within 3 px — done. Otherwise pairs with an error above `max(3, worst / 2)` are
   dropped (worst first: good pairs next to a bad one also look worse until it is gone) and the
   system is solved again. If dropping pairs splits the graph (`_skladowa_z()`), the largest part
   is kept.
3. Frames with a resulting scale outside 0.7–1.4 are left out. `blad_px` = median pair error
   (0.66 px in the 55-frame example).

### `_kotwica()` and `_uklad()` — anchor and matrix (`global_stitcher.py`)

Without help, contradictory pairs can "agree" by shrinking frames towards zero size.
`_kotwica()` gives each frame a rough scale and rotation along the strongest connections —
Prim's algorithm with a heap ordered by the number of points in a pair. Scale and rotation are
stored as a complex number `c = a + b·i`; chaining two transforms is just multiplication
(`c_j = c_i · c_T`: scales multiply, angles add). `_uklad()` adds two soft equations per frame,
`w·a_k = w·a0` and `w·b_k = w·b0`, with weight `w = 2 × median frame side` (≈ 960 for 480 px
frames): strong enough to stop the collapse, weak enough not to outvote the point equations.

`_uklad()` builds the matrix block by block with NumPy (all points of a pair at once): for each
pair a block of `2m` rows, the coefficients `(x1, −y1, 1, 0)` / `(y1, x1, 0, 1)` in the columns
of frame i and the negated ones for frame j.

`_blad_pary()` maps p and q onto the mosaic with their frames' solutions and returns the
**median** distance — one bad point does not spoil the measure, a bad pair (a neighbouring crop
row) stands out with tens of pixels.

### `_renderuj()` and `na_mozaike()` — the picture (`global_stitcher.py`)

For every placed frame: the 2 × 3 matrix `[[a, −b, tx], [b, a, ty]]`, the bounding box of its
corners, and a `warpAffine` **only into that box** (not the whole canvas — memory) plus a mask of
valid pixels. Then `GainCompensator` (one gain per frame, no exposure steps; `BlocksGainCompensator`
optionally), optional `GraphCutSeamFinder` seams, and `MultiBandBlender`, which blends coarse and
fine detail separately: smooth seams, sharp details.

`na_mozaike(path, points, original_size)` maps points from an original full-size frame (e.g. the
corners of a detected weed) onto mosaic pixels: scale to the working resolution → the frame's
transform → minus `_poczatek_mozaiki()` (frames left of or above the reference have negative
coordinates, the image starts at pixel 0). GUI tab 4 uses it to draw the weeds on the mosaic.

### `DetailStitcher` and `BasicStitcher` — for comparison

`DetailStitcher` runs the `cv2.detail` blocks with an **affine** model
(`AffineBestOf2NearestMatcher`, `AffineBasedEstimator`, `BundleAdjusterAffinePartial`, warper
`affine`) — the default rotational panorama model would explain the drone's translation by
rotation and produce absurd focal lengths. It keeps only the largest component
(`leaveBiggestComponent`) and can use a sliding window of the last N frames (constant time per
frame). `BasicStitcher` calls `cv2.Stitcher` in SCANS mode on all frames after every new one — the
simplest reference, its time grows with every frame.

### `ocen()` and `werdykt()` — how good is the mosaic (`evaluation.py`)

The mosaic has its own coordinate frame, so it must first be aligned with the original:

1. `_obszar_lotu()` — only the flown part of the original (bounding box of all frame corners
   from `prawda.json` + 5%);
2. `_szary()` — both images grey and shrunk to at most 1600 px;
3. SIFT (8000 features) and, if fewer than 40 pairs, ORB with a low threshold; the set with more
   pairs wins. `_pary_wzajemne()` keeps a pair only if it passes the ratio test (0.8) **and** is
   mutual (A's best match is B and B's best match is A) — repetitive rows otherwise make many
   mosaic points hit one original point and RANSAC picks a degenerate "shrink everything" model;
4. RANSAC similarity mosaic → original (scale < 0.2 = degenerate → not OK);
5. the mosaic warped onto the original; masks of mosaic data and of the flown area (filled frame
   polygons);
6. on the common part: **coverage** (share of the flown area present), **mean brightness error**
   (0–255, computed in int16 to avoid wrap-around) and **PSNR** = `10 · log10(255² / MSE)`.

`werdykt()`: `ZLE` (bad) if fewer than 8 pairs or scale outside 0.85–1.15; `CZESCIOWO` (partial)
if coverage < 0.7; `SLABO` (weak) if PSNR < 20 dB; `DOBRZE` (good) if < 25 dB; otherwise
`BARDZO DOBRZE` (very good). Fewer than 20 pairs adds "low confidence of the evaluation".

### `FolderSource` — live mode (`source.py`)

A background thread lists the folder every `interwal_s` seconds (1 s) and puts new files into a
`queue.Queue`; `cli.petla()` takes them out and calls `dodaj()`, so reading the folder never
blocks stitching. A file is taken only when its size is the same in two reads 0.4 s apart
(`stabilizacja_s`) — a frame being copied is not read half-written.

**Open point:** `GlobalStitcher(detektor=...)` defaults to `"SIFT"` in the class, while
`stitchlive.cli` passes `--detektor ORB` by default; the GUI uses the class default.

---

Tests: `tests/test_global_stitcher.py`, `test_stitch.py`, `test_flight_sim.py`, `test_camera.py`.
The simulator is also used by [`mission/PIPELINE.md`](../mission/PIPELINE.md) (`drone_sim`) and
the GUI ([`docs/gui.md`](../docs/gui.md)).
