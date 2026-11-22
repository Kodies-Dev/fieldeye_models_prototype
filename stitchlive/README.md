# stitchlive — flight simulator, mosaic stitching and drone camera model

**Part of the pipeline:** simulator / mosaic (laptop tools; used by `app.py` tabs 1, 3 and 4
and by the simulated drone in `mission/`).

The package does three things:

1. **Flight simulator** — takes one large image (or a synthetic field) and cuts it into
   frames the way a drone camera would take them, with realistic disturbances, and saves the
   true position of every frame.
2. **Mosaic stitching** — joins drone frames into one mosaic, also live while new frames arrive.
3. **Evaluation** — aligns the mosaic with the original and measures how good it is (PSNR, coverage, scale).

How it works inside — flow, technologies, key functions explained: [`PIPELINE.md`](PIPELINE.md).

This is "Program 1" of the project plan: checking whether the material can be stitched at all.
Geo-referencing the mosaic is out of scope (weed coordinates come from the telemetry of each photo, see `mission/`).

---

## Files

| File | Role | Main elements |
|---|---|---|
| `global_stitcher.py` | **default stitching method** | `GlobalStitcher` — frame pairs from OpenCV (ORB or SIFT + RANSAC), positions of all frames from one linear system |
| `detail.py` | method `detail` | `DetailStitcher` — `cv2.detail` pipeline, affine model, sliding window |
| `basic.py` | method `basic` | `BasicStitcher` — wrapper around `cv2.Stitcher` (SCANS mode) |
| `flight_sim.py` | flight simulator | `SimSettings`, `rozloz()` — cut one image into frames (square or 4:3), lawn-mower route or random, with disturbances and motion blur |
| `camera.py` | drone camera model | `CameraModel`, `opis_lotu()` — ground footprint from altitude, frame size in source pixels, 1.5 m tile, motion blur; Arducam 64MP (FIELDEYE) and Raspberry Pi HQ |
| `evaluation.py` | result evaluation | `ocen()` — aligns the mosaic with the original, computes PSNR, coverage, scale; `werdykt()` |
| `field.py` | test data | `zbuduj_pole()` — synthetic field with crop rows, weeds and stones |
| `source.py` | image sources | `FolderSource` (watches a folder in a background thread, takes a file only when its size stops changing), `BatchSource` |
| `cli.py` | command line | commands `batch`, `watch`, `info`, `rozloz`, `ocen`, `demo` |

All stitchers share one interface, so the GUI and the CLI can swap them:

```python
s = GlobalStitcher(work_megapix=0.6, renderuj_co=3)
for path in photos:
    if s.dodaj(path):        # True = mosaic updated
        preview(s.mozaika)
s.zakoncz()                  # global only: final render
s.raport_czasu()             # [{"zdjec": n, "czas_s": t, "ok": bool}, ...]
```

Data flow:

```
camera.opis_lotu()   ->  frame size, tile, blur                 (drone camera mode)
flight_sim.rozloz()  ->  images/*.jpg + prawda.json (ground truth)
                             |
source.*Source       ->  queue  ->  *Stitcher.dodaj()  ->  mosaic
                                                            |
                                evaluation.ocen(mosaic, original, ground truth)
```

---

## Quick start

Run `python -m stitchlive.cli ...` from the project folder (after `pip install -e .` also as `stitchlive ...`).

```bash
python -m stitchlive.cli demo                                    # split -> stitch -> evaluate on a synthetic field
python -m stitchlive.cli demo photo.jpg --max-wymiar 2400 \
    --tryb-ciecia losowy --liczba 30 --obrot 6 --rozmycie 0.2 --szum 3
```

Output in `demo/`: `symulacja/podglad_trasy.jpg` (route on the original), `symulacja/images/`
(frames), `symulacja/prawda.json` (true position, angle and footprint of each frame),
`mozaika/mozaika_biezaca.jpg` (mosaic), `ocena/ocena_nalozenie.jpg` (mosaic over the original).

### Commands

| Command | What it does |
|---|---|
| `rozloz [IMAGE] -o OUT` | flight simulation: cut one image into frames (no image = synthetic field) |
| `batch FOLDER -o OUT` | stitch a whole folder at once |
| `watch FOLDER -o OUT` | watch a folder during the flight and stitch live |
| `ocen MOSAIC ORIGINAL --prawda prawda.json` | compare a mosaic with the original |
| `demo [IMAGE]` | `rozloz` → stitch → `ocen` in one go |
| `info FOLDER` | information about a set of images |

Live flight in two terminals:

```bash
python -m stitchlive.cli watch inflow -m global --renderuj-co 4 -o results          # terminal 1
python -m stitchlive.cli rozloz photo.jpg --tryb-ciecia losowy --na-zywo inflow --odstep 0.5   # terminal 2
```

### Simulator options

| Option | Simulates | Example |
|---|---|---|
| `--tryb-ciecia` | `trasa` (default): lawn-mower lines with small jitter; `losowy`: random positions, each frame keeps a minimum overlap with the previous one | `--tryb-ciecia losowy` |
| `--obrot` | heading deviation (std. dev. in degrees) | `--obrot 5` |
| `--obrot-linii` | every second line rotated by 180° (drone turns around) | |
| `--skala` | altitude change (fraction) | `--skala 0.05` |
| `--ekspozycja` | brightness change between frames | `--ekspozycja 0.06` |
| `--rozmycie` | probability of motion blur | `--rozmycie 0.2` |
| `--szum` | sensor noise (brightness levels) | `--szum 3` |
| `--winieta` | darker frame corners, 0–1 | `--winieta 0.3` |
| `--jitter` | random position deviation (route mode) | `--jitter 0.1` |

Others: `--kadr` (frame size in px), `--zasieg` (how many source px a frame covers),
`--overlap`, `--sidelap`, `--liczba` (random mode), `--seed`.

### Stitching options

| Option | Meaning |
|---|---|
| `-m, --metoda` | `global` (default), `detail`, `basic` |
| `--megapix` | working resolution (default 0.6 Mpx) |
| `--zakres-par N` | match a frame only with the N previous ones; 0 = with all |
| `--renderuj-co N` | refresh the mosaic every N frames |
| `--cechy` | features per frame (default 1500) |
| `--szwy` | seam finding — removes visible borders, slower |
| `--bloki-ekspozycji` / `--bez-ekspozycji` | block exposure compensation (slow) / none |
| `--detektor` | `ORB` (default) or `SIFT` |

For a lawn-mower flight set `--zakres-par` to at least two line lengths — the neighbouring
frame of the next line is far away in the order.

---

## Drone camera model

The simulator can cut frames like the FIELDEYE camera: **Arducam 64MP Hawkeye** (sensor
7.4 × 5.55 mm, 5.1 mm lens), pointing straight down. From altitude and optics it computes
the ground footprint of a photo and cuts exactly that much from the source.

| Flight at 5 m, 16 MP mode | |
|---|---|
| photo covers | 7.25 × 5.44 m of field |
| camera GSD | 1.57 mm/px |
| frame in a WeedsGalore source (2.5 mm/px) | 2902 × 2176 px |
| tile before the model | 600 source px = 1.5 m of field = a training tile (on the drone: about 956 photo px) |
| motion blur at 3 m/s and 1/1000 s | 1.2 px |
| photo every | 0.73 s at 70% overlap |

```bash
python -m stitchlive.cli rozloz --wysokosc 5 --pole-w 8000 --pole-h 4400 -o sim_5m
python -m stitchlive.cli rozloz photo.jpg --wysokosc 5 --gsd-zrodla 2.5 --predkosc 3 --naswietlanie 0.001
```

The frame keeps the source resolution (upscaling the source would add no detail). Before the
model it is split into 1.5 m tiles exactly as on the drone, so weeds have the same size at the
network input as in training. In the GUI: tab 4, checkbox "Kamera drona". The field must be
larger than one photo (at 5 m: more than 7.3 × 5.4 m).

---

## Stitching methods

### `global` — default

Frame pairs are found with OpenCV (features, ratio test, RANSAC). The positions of **all**
frames come from one linear system: a similarity transform (rotation + scale + shift) is
linear in its parameters, so every point pair gives two equations. The first frame is the
reference. The error does not accumulate along the route.

**Why a separate method:** the `cv2.detail` pipeline — and `cv2.Stitcher` in SCANS mode —
produce a blurred mosaic when frames are rotated by a few degrees, even though single pairs
match within 0.1–0.8 px. Without rotation there is no problem.

Built-in safeguards:

- pairs with too few RANSAC inliers or an absurd scale are rejected,
- pairs that do not fit the solution (error > 3 px) are removed and the system is solved again,
- a frame not connected to the rest is left out instead of spoiling the mosaic; the largest
  connected group is stitched even if the first frame dropped out,
- a blurred frame on which the detector finds nothing gets a second try with a lower threshold,
- local contrast equalisation (CLAHE) before feature detection — with vignetting 0.3 the number
  of matched pairs used to drop from 147 to 35.

### `detail` — `cv2.detail` pipeline

Affine model (what `cv2.Stitcher` SCANS does, split into blocks). Works well without
rotation; has a sliding window (`--okno N`) for constant time per added frame.

### `basic` — `cv2.Stitcher`

Recomputes everything after every photo. Reference point.

---

## Measured results

PSNR of the mosaic against the original (above ~25 dB = very good agreement).

| Case | `detail` / `cv2.Stitcher` | `global` |
|---|---|---|
| synthetic field, no rotation | 27.8 dB | 27.8 dB |
| synthetic field, rotation 8° | 19.7 dB (blurred) | 26.7 dB |
| real drone photo, route, 48 frames | — | 25.2 dB, 48/48 frames |
| real photo, random cut + rotation 6° + altitude change + blur + noise, 30 frames | — | 23.3 dB, 30/30 frames |

Real photo: `DJI_0018.JPG` from the ODMdata *brighton_beach* set (forest, road, lawn — trees
are a hard, repetitive texture).

Time for 48 frames of a real photo: 117 s with block exposure compensation and 3000 features,
10.5 s with per-frame compensation and 1500 features (default); quality practically the same
(24.6 dB). Live mode (16 frames every 0.5 s, render every 4 frames): 0.1–0.45 s per refresh.

---

## Limitations

- **No geo-referencing** — the mosaic is in the coordinate frame of the first frame.
- **Flat field, nadir photos** — the similarity model ignores perspective and terrain height.
- **Memory** — all working frames are kept in memory; for very long missions lower `--megapix`.
- **Matching time grows with the number of frames** when `--zakres-par` is 0.
- **Brightness patches** on uniform surfaces (asphalt) with per-frame exposure — `--szwy` or `--bloki-ekspozycji` help.

Tests: `tests/test_global_stitcher.py`, `test_stitch.py`, `test_flight_sim.py`, `test_camera.py`.
