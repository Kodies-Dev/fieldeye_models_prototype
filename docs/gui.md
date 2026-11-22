# GUI (`app.py`) — how it works inside

How the Streamlit app ties the simulator, the stitching and the weed detector together in a
browser. A short tab overview is in the root [`README.md`](../README.md#gui-apppy); this page
explains the flow of every tab and the parts of the code that are not obvious.

```bash
streamlit run app.py          # from the project folder; opens http://localhost:8501
```

**Input:** an uploaded field image, uploaded frames / ZIP, or the test tiles of
`zbior_lotnicze/`. **Output:** frames, mosaics, evaluations, GIFs and ZIPs to download.
**Runs on:** a laptop (the `.hef` model only on a device with Hailo).

---

## 1. Flow

```
app.py  (the whole file runs top to bottom on EVERY click or slider move)
│
├─ inicjuj_suwaki()                         disturbance sliders start at "Realistyczny dron"
├─ st.tabs([...])
│
├─ tab 1  Rozkład zdjęcia (split an image)
│    upload → wczytaj_z_uploadu() → zmniejsz(≤ 2400 px)
│    SimSettings(route, frame, overlap, disturbances, seed) → stitchlive.flight_sim.rozloz()
│    → session_state.symulacja = {folder, prawda, oryginal}    route preview + first frames
│
├─ tab 2  Kadry do pobrania (frames)
│    gallery from session_state.symulacja; frame details from prawda.json
│    spakuj() → kadry.zip | symulacja_pelna.zip (+ prawda.json, route preview, original)
│
├─ tab 3  Składanie (stitching)
│    frames from tab 1, or uploaded photos / ZIP → rozpakuj_kadry()
│    GlobalStitcher | DetailStitcher | BasicStitcher
│    for each frame: dodaj() → live preview in st.empty(), states kept for the GIF
│    zakoncz() → stitchlive.evaluation.ocen() (only when the frames come from tab 1)
│    → session_state.wynik; downloads: mosaic JPG, GIF (gif_z_klatek), ZIP with steps + report
│
└─ tab 4  Lot z detekcją (flight with detection)
     field: WeedsGalore tiles of one day (pole_z_kafelkow, with ground truth)
            | uploaded image | synthetic field (zbuduj_pole)
     detektor_lotu()  (cached)  →  [TiledDetector 1.5 m tiles in drone camera mode]
     rozloz() → for each frame: detect, prawda_w_kadrze(), porownaj(), draw, GlobalStitcher.dodaj()
     zakoncz() → na_mozaike_wszystkie() for detections and for ground truth → scal_ramki()
     → session_state.lot_wynik; mosaic with boxes, counts, JPG and GIF (gif_lotu)
```

---

## 2. Technologies

| Technology | Used for |
|---|---|
| **Streamlit** | the whole interface: `st.tabs`, `st.columns`, sliders / radios / uploaders, `st.session_state` (state between reruns), `st.cache_resource` (model loaded once), `st.empty()` (one place on the page overwritten frame by frame), `st.progress`, `st.download_button`, `st.metric` |
| **OpenCV, NumPy** | decoding uploads (`cv2.imdecode`), resizing, drawing boxes, side-by-side frames, JPEG encoding in memory (`cv2.imencode`) |
| **Pillow** | animated GIFs |
| `io`, `zipfile`, `tempfile` | ZIPs built in memory; one temporary working folder per browser session |
| `stitchlive` | `flight_sim` (frames), `camera` (drone camera mode), `field` (synthetic field), the three stitchers, `evaluation` |
| `mission` | `detector` (`YoloDetector`, `HailoDetector`, `TiledDetector`, `DummyDetector`), `sim_flight` (fields from tiles, ground truth in a frame, box merging, comparison) |

---

## 3. Files in this process

| File | Role |
|---|---|
| `app.py` | the app: helpers at the top, then one `with tabN:` block per tab |
| `stitchlive/flight_sim.py`, `camera.py`, `field.py` | frames and ground truth — [`stitchlive/PIPELINE.md`](../stitchlive/PIPELINE.md) |
| `stitchlive/global_stitcher.py`, `detail.py`, `basic.py`, `evaluation.py` | stitching and evaluation |
| `mission/detector.py`, `mission/sim_flight.py` | the detector and the tab 4 helpers — [`mission/PIPELINE.md`](../mission/PIPELINE.md) |
| `models/weights/fieldeye_final.pt` | default model of tab 4 |
| `tests/test_gui_flight.py` | tab 4 run headless with Streamlit `AppTest` |

---

## 4. Key mechanisms and functions

### The rerun model and `st.session_state`

Streamlit executes the **whole script again** after every interaction. A result computed when a
button was clicked would disappear on the next slider move, so everything that must survive is
stored in `st.session_state`: `symulacja` (tab 1 → used by tabs 2 and 3), `wynik` (tab 3),
`lot_wynik` (tab 4) and `workdir`. A new split in tab 1 removes the old stitching result,
because it no longer matches the frames.

### `katalog_roboczy()` — one folder per browser session

Creates a folder with `tempfile.mkdtemp(prefix="fieldeye_")` once per session and keeps its path
in `session_state`. Two people using the app at the same time (or two browser tabs) never
overwrite each other's frames. Subfolders: `symulacja/` (tab 1), `wgrane/` and `ocena/` (tab 3),
`lot/` (tab 4). The app does not delete it; F5 starts a new session with a new folder.

### `ustaw_preset()` and `inicjuj_suwaki()` — presets for seven sliders

The disturbance sliders (heading deviation, altitude change, exposure, blur chance, noise,
vignetting, position jitter) have only a `key`, no default value. `inicjuj_suwaki()` runs at the
start of every rerun and uses `setdefault`, so it sets the "Realistyczny dron" values only the
first time and never undoes the user's changes. Choosing a preset calls `ustaw_preset()` as an
`on_change` callback — callbacks run **before** the rerun, when no slider exists yet, which is the
only moment Streamlit allows writing their values in `session_state`.

| Preset | Heading ° | Altitude | Exposure | Blur | Noise | Vignetting | Jitter |
|---|---|---|---|---|---|---|---|
| Czyste ciecie (clean cut) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Realistyczny dron (realistic drone) | 3 | 0.03 | 0.05 | 0.1 | 2 | 0.3 | 0.1 |
| Trudne warunki (hard conditions) | 6 | 0.05 | 0.07 | 0.25 | 4 | 0.35 | 0.12 |

### Small helpers

| Function | What it does |
|---|---|
| `bgr_na_rgb()` | OpenCV stores BGR, the browser expects RGB — without it red and blue swap |
| `wczytaj_z_uploadu()` | uploaded bytes → `np.frombuffer` → `cv2.imdecode`, no file on disk |
| `zmniejsz()` | longer side ≤ N px with `INTER_AREA`; never enlarges |
| `jpg_bajty()` | `cv2.imencode` → bytes for a download button |
| `spakuj()` | ZIP in an `io.BytesIO`; files from disk under `images/`, extra data under their names |
| `rozpakuj_kadry()` | uploaded photos and ZIPs → one folder, sorted by name (= flight order with zero-padded names); folders inside ZIPs flattened, `__MACOSX` skipped |
| `gif_z_klatek()` | growing mosaic states pasted on one black canvas of the largest size (the animation does not jump), scaled, saved by Pillow |

### Tab 3 — one loop for three stitchers

All stitchers have the same `dodaj(path) → bool`. The loop adds frames one by one, writes the
current mosaic into the same `st.empty()` placeholder (the mosaic grows in place instead of
appending images below each other) and keeps every refreshed state for the GIF. Optional delay
between frames imitates a flight. After `zakoncz()` (global only) the mosaic is evaluated with
`ocen()` against the original and `prawda.json` — only possible for frames from tab 1, where the
truth is known. Uploaded photos are stitched without evaluation.

### `detektor_lotu()` — the model loaded once (tab 4)

Decorated with `@st.cache_resource`: for the same `(model, threshold, input)` the detector is
created once per server and reused by every flight. Choice by file: `.hef` → `HailoDetector`,
another existing file → `YoloDetector`; when the file is missing or the library is not installed
→ `DummyDetector` (green blobs, **not** a weed detector) plus a visible warning.

### Tab 4 — the flight with detection

1. **Field.** "Kafelki WeedsGalore (test)": up to 25 tiles of **one** day glued into a grid with
   their true boxes (`pole_z_kafelkow`) — days must not be mixed, because the same place twice
   breaks the stitching. "Wgraj zdjęcie": any image, no ground truth. "Sztuczne pole": synthetic,
   no ground truth — only to check the run.
2. **Frames.** `SimSettings` with a lawn-mower route, overlaps from the sliders and light
   disturbances (rotation 2°, exposure 0.03, jitter 0.08; off with the checkbox). With
   **"Kamera drona"** `opis_lotu()` sets the real camera geometry: frame size in source pixels from
   altitude and optics, motion blur from speed and exposure time, and the detector is wrapped in
   `TiledDetector` with 1.5 m tiles — exactly as on the drone. Without it a frame is 600 px, the
   size of a training tile.
3. **Per frame.** Detection (boxes ≥ threshold), true boxes in that frame (`prawda_w_kadrze()`),
   per-frame hits / misses / false alarms (`porownaj()`), green (truth) and pink (detected) boxes
   drawn, side-by-side preview, `GlobalStitcher.dodaj()`. The stitcher is created with
   `renderuj_co=10**6` — it only collects pairs during the flight and renders once at the end; with
   more than 40 frames it matches each frame only with its own and the neighbouring route line
   (`zakres_par`).
4. **Mosaic.** `zakoncz()`, then the inner `na_mozaike_wszystkie()`: the four corners of every box
   go through `GlobalStitcher.na_mozaike()` (working scale → frame transform → mosaic origin), the
   bounding box of the result is the box on the mosaic, and `scal_ramki()` merges the same weed
   seen on several frames (IoU ≥ 0.3). The same is done for the true boxes, and the merged sets are
   compared once more for the mosaic totals.
5. **Result.** Counts (frames, frames in the mosaic, detections, weeds after merging; with ground
   truth: true weeds, hits, misses, false alarms), a warning when fewer than 80% of the frames made
   it into the mosaic, a view switch (detected / true / both / none) and downloads: mosaic JPG of
   the current view and the flight GIF (`gif_lotu()`: frame | detections pairs, the mosaic as the
   last frame for 4 s).

Mosaic numbers are approximate: merging boxes from neighbouring frames can join two weeds that
grow very close together.

---

Tests: `tests/test_gui_flight.py` (tab 4: flight with detection, tiles grouped by day, drone
camera mode).
