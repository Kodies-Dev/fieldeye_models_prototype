# fieldeye-detect: weed detection on drone photos and the spray map

`fieldeye-detect` watches the folder where the drone saves photos, runs every new photo
through the weed model and — when GPS telemetry is saved next to the photo — turns the
detections into coordinates and builds a spray map. It runs from source (Python) and as a
binary, on macOS, Ubuntu and Raspberry Pi 5 with the Hailo-8 accelerator.

```
camera + autopilot              folder PHOTOS                      fieldeye-detect                  folder RESULTS
(drone, Raspberry Pi)   ──►   20261109T142859.998Z.jpg     ──►   tiles → model → geolocation   ──►   oprysk.geojson / oprysk.csv
                              20261109T142859.998Z.geojson       → merging into spray points        chwasty.geojson, trasa.geojson
                                                                                                     photos with boxes
```

## Contents
1. [Running](#1-running)
2. [Drone telemetry (format)](#2-drone-telemetry-format)
3. [Outputs](#3-outputs)
4. [How coordinates are computed](#4-how-coordinates-are-computed)
5. [Safeguards](#5-safeguards)
6. [Binary](#6-binary)
7. [Raspberry Pi](#7-raspberry-pi)
8. [Simulated drone and scoring](#8-simulated-drone-and-scoring)
9. [Code map](#9-code-map)
10. [Common problems](#10-common-problems)

---

## 1. Running

```bash
fieldeye-detect PHOTOS RESULTS MODEL [options]                       # binary
python -m mission.detect_cli PHOTOS RESULTS MODEL [options]          # from source (project folder)
bash scripts/fieldeye-detect.sh PHOTOS RESULTS MODEL [options]       # from source, activates .venv itself
```

| Argument | Meaning |
|---|---|
| `PHOTOS` | folder with photos; in watch mode it is created if missing |
| `RESULTS` | folder for results (created if missing); without it, results only in the terminal |
| `MODEL` | path with file name: `.hef` (Hailo), `.onnx`, `.pt`; omitted → the model next to the binary, or the final model of the project |

Paths may start with `~` (also in quotes).

| Option | Default | Meaning |
|---|---|---|
| `--raz` | off | process what is in the folder and exit (otherwise: watch until Ctrl+C) |
| `--tylko-nowe` | off | skip photos that were already in the folder at start |
| `--timeout S` | none | exit after S seconds without new photos |
| `--prog P` | 0.15 | confidence threshold (deliberately low: a missed weed is worse than a false alarm) |
| `--kafelek PX` | from telemetry (1.5 m of field); without it 950 | tile size; `0` = whole photo at once |
| `--zakladka F` | 0.15 | overlap of neighbouring tiles |
| `--tta` | off | also a mirrored pass (about 2× slower, fewer false alarms) |
| `--wejscie PX` | 1280 | network input for `.pt` / `.onnx` (built into `.hef`) |
| `--urzadzenie` | auto | `cpu`, `mps`, `0` (CUDA) |
| `--kurs-domyslny DEG` | none | heading used when telemetry has none (tests while hovering) |
| `--maks-przechyl DEG` | 5 | photos tilted more do not go onto the map |
| `--promien-scalania M` | 0.10 | detections closer than this are the same weed |
| `--czekaj-telemetrie S` | 3 | how long to wait for the `.geojson` of a finished photo before going without GPS |
| `--stabilnosc S` | 1 | how many seconds a photo's size must stay constant before it is read |
| `--interwal S` | 0.5 | how often the folder is checked |
| `--jsonl` | off | also `RESULTS/wyniki.jsonl` (boxes in pixels, GPS, coordinates) |
| `--json` | off | output only JSON, one line per photo (for other programs) |
| `--szczegoly N` | 0 | print the N most confident detections of each photo |
| `--bez-telemetrii` | off | do not look for `.geojson` files (detections only) |

Terminal line for every photo:

```
[14:29:01] 20261109T142859.998Z.jpg   4624x3472   chwasty:   17   max 0.91   0.84 s   GPS 50.290049, 18.731013   oprysk: 15 pkt (+15)   (w kolejce: 1)
```

`chwasty` = weeds found, `max` = highest confidence on the photo, `oprysk` = spray points on
the map, `(w kolejce: N)` = photos waiting (a growing number means the model cannot keep up
with the camera).

Photos are processed in the order they were written (FIFO). A photo is read only when it is
complete (size constant for `--stabilnosc` s and a complete JPEG/PNG). Hidden (`.`, `~`) and
temporary (`.part`) files are skipped; damaged files do not stop the watching.

## 2. Drone telemetry (format)

Next to every photo a file with **the same name** and the extension **`.geojson`**, in the same
folder, written **atomically** (temporary file, then rename):

```
20261109T142859.998Z.jpg   →   20261109T142859.998Z.geojson
```

Letter case must match; the extension is lower-case `.geojson`. Telemetry may arrive before the
photo or at most `--czekaj-telemetrie` s after it.

```json
{"type": "FeatureCollection",
 "features": [{"type": "Feature",
   "geometry": {"type": "Point", "coordinates": [18.73101322, 50.29004902, 239.73]},
   "properties": {
     "image": "20261109T142859.998Z.jpg", "seq": 1, "flight_id": "20261109T142856Z",
     "time_utc": "2026-11-09T14:28:59.998Z", "fix": true, "fix_quality": 2,
     "speed_mps": 0.02, "course_deg": 7.9,
     "heading_deg": 7.5, "roll_deg": 0.4, "pitch_deg": -1.2,
     "survey": {"agl_m": 2.0, "gsd_mm_per_px": 0.623},
     "camera": {"width": 4624, "height": 3472}}}]}
```

| Field | Required | Used for |
|---|---|---|
| `coordinates` `[longitude, latitude]` (WGS84) | **yes** | drone position = photo centre on the field (camera points straight down) |
| `survey.agl_m` | **yes** (or `gsd_mm_per_px`) | altitude **above the field** [m] → photo scale from the Arducam camera model |
| `survey.gsd_mm_per_px` | no | scale directly [mm of field per pixel]; takes precedence over altitude |
| `heading_deg` | **yes**\* | drone heading: 0 = north, clockwise; top of the photo = front of the drone |
| `roll_deg`, `pitch_deg` | recommended | tilt check (above `--maks-przechyl` the photo is left off the map) |
| `fix`, `fix_quality` | recommended | `false` / `0` → photo without GPS |
| `time_utc`, `image` | no | time stored on the map; matching a point to its photo |

\* Without `heading_deg`, heading = `course_deg` (direction of motion from GPS), but only at
`speed_mps ≥ 1`: while hovering the direction of motion is noise, and in flight the drone may
fly sideways. Without a heading the photo goes without GPS (for tests: `--kurs-domyslny 0`).
Other fields are ignored.

Reference file: [`tests/dane/telemetria_dron.geojson`](../tests/dane/telemetria_dron.geojson).

## 3. Outputs

In the `RESULTS` folder, refreshed after every photo, written atomically:

| File | When | Content |
|---|---|---|
| `<photo name>.jpg` | always | the photo with pink boxes around weeds |
| `oprysk.geojson` | with telemetry | **spray points**: one point per weed; `id`, `promien_m` (radius), `pewnosc` (confidence), `liczba_wykryc` (detections), `liczba_zdjec` (photos) |
| `oprysk.csv` | with telemetry | the same as a table: `id,szerokosc,dlugosc,promien_m,pewnosc,liczba_zdjec` (latitude, longitude, ...) |
| `chwasty.geojson` | with telemetry | every detection on every photo; `pewnosc`, `rozmiar_m`, `zdjecie`, `czas`, `niepewnosc_przechylu_m` |
| `trasa.geojson` | with telemetry | drone route (line) and photo positions |
| `wyniki.jsonl` | with `--jsonl` | one line per photo: boxes in pixels, GPS, detection coordinates |

Drop the `.geojson` files onto [geojson.io](https://geojson.io) or open them in QGIS.
Without telemetry there are only photos with boxes (no coordinates).

## 4. How coordinates are computed

Code: [`mission/spray_map.py`](../mission/spray_map.py).

1. **Scale** (metres of field per pixel, GSD): `survey.gsd_mm_per_px / 1000` or
   `agl · sensor width / (focal length · photo width in px)` — Arducam 16 MP: sensor 7.4 mm,
   focal length 5.1 mm (2 m → 0.623 mm/px; 5 m → 1.57 mm/px).
2. **Tile** = 1.5 m / scale — the model sees weeds at the size it was trained on.
3. **Weed offset from the photo centre** (box centre `(px, py)`, photo `W × H`):
   `dx = (px − W/2) · GSD` (right), `dy = (py − H/2) · GSD` (down).
4. **Rotation by heading θ** (front of the drone = top of the photo):
   `east = dx·cos θ − dy·sin θ`, `north = −dx·sin θ − dy·cos θ`.
5. **Metres → degrees** in a local plane tangent to the WGS84 ellipsoid (reference: the first
   photo; at 50° latitude about 111 235 m per 1° of latitude and 71 260 m per 1° of longitude).
   Approximation error on a field of a few hundred metres: fractions of a millimetre.
6. **Merging**: a detection closer than `--promien-scalania` to an existing point joins it
   (position = confidence-weighted mean), otherwise it starts a new point. Spray radius = half of
   the largest box + 5 cm.

Example (telemetry from [`tests/dane`](../tests/dane/telemetria_dron.geojson), heading 90°): a weed
at pixel (3100, 900) of a 4624 × 3472 photo at 0.623 mm/px → 0.521 m east and 0.491 m south of the
drone → 50.29004461° N, 18.73102053° E.

**Assumptions** (deliberate simplifications): the camera points straight down (no tilt
correction), flat field, position from the moment of the shutter, camera under the GPS antenna.
Map accuracy is decided mainly by GPS: without RTK (`fix_quality` 1–2) about 1 m, with RTK
(`fix_quality` 4) a few cm.

## 5. Safeguards

| Situation | Reaction |
|---|---|
| no `.geojson` after `--czekaj-telemetrie` s | photo with boxes, no coordinates (`bez GPS`) |
| unreadable file / not a `FeatureCollection` | as above, with a message |
| coordinates out of range or 0, 0 | no GPS: "swapped order?" / "no GPS position" |
| `fix = false` or `fix_quality = 0` | no GPS: "no GPS fix" |
| no heading | no GPS: "add heading_deg…" |
| `agl_m` outside 0.3–150 m | no GPS: "altitude above sea level instead of above the field?" |
| `gsd_mm_per_px` outside 0.1–100 | no GPS: "wrong units?" |
| tilt > `--maks-przechyl` | photo with boxes, left off the map (`PRZECHYL … - poza mapa oprysku`) |
| tilt within the limit | on the map, with `niepewnosc_przechylu_m = agl · tan(angle)` |
| photo still being written | wait for constant size and a complete file |
| damaged photo | skipped (`POMINIETE`), watching continues |
| results folder = photos folder | refuses to start |

The program prints its messages in Polish; they are translated in the table above.

A swapped coordinate order cannot be detected automatically in Poland (50 and 18 are valid in
both ranges) — it is obvious at once on `trasa.geojson`.

## 6. Binary

One file with Python and the libraries inside (PyInstaller); the model lies next to it or is
given as the third argument. **Build on the system it will run on** (Mac → Mac, Ubuntu → Ubuntu,
Pi → Pi).

```bash
bash scripts/build_binary.sh onnx            # Mac / Ubuntu: ONNX Runtime, no PyTorch (~100 MB)
bash scripts/build_binary.sh hailo           # Raspberry Pi with hailo-all: .hef model (~170 MB)
bash scripts/build_binary.sh pt --katalog    # PyTorch; large, better as a folder
```

Result: `dist/fieldeye-detect`. The same binary has helper commands:

```bash
./fieldeye-detect PHOTOS RESULTS MODEL
./fieldeye-detect dron FOLDER --ewis1 TILES          # simulated drone (section 8)
./fieldeye-detect porownaj OPRYSK TRUTH              # score spray points
```

| Variant | `.onnx` | `.hef` | `.pt` |
|---|---|---|---|
| `onnx` | ✅ | ❌ | ❌ |
| `hailo` | ❌ | ✅ | ❌ |
| `pt` | ✅ | ❌ | ✅ |

ONNX from the final model:
`python -m models.cli onnx models/weights/fieldeye_final.pt -o fieldeye_final.onnx --wejscie 1280 --opset 11`.
Self-contained folder (binary, models, examples): `bash scripts/przygotuj_folder.sh TARGET`.

## 7. Raspberry Pi

Step-by-step guide (install, build, start on boot, inputs and outputs): [`raspberry-pi.md`](raspberry-pi.md).

On the Mac: `bash scripts/spakuj_na_pi.sh --hef PATH/fieldeye_final_1280_h8.hef` →
`~/Downloads/fieldeye_pi.zip` (code, HEF, 132 EWIS1 test tiles, `na_pi.sh`, `README_PI.txt`).
Without `--hef` it looks for a `.hef` on disk and then inside `model_fin.zip`.

On the Pi (Raspberry Pi OS 64-bit, AI HAT+, HailoRT 4.23):

```bash
sudo apt install -y hailo-all python3-venv python3-pip python3-opencv python3-numpy unzip
unzip -oq fieldeye_pi.zip && cd fieldeye_pi
bash na_pi.sh sprawdz      # Hailo, Python, OpenCV, numpy, model, data
bash na_pi.sh test         # 132 tiles through the .hef (no GPS)
bash na_pi.sh gps          # simulated flight with telemetry + map + scoring
bash na_pi.sh na-zywo      # a photo every 0.7 s, live detection: does Hailo keep up
bash na_pi.sh binarka      # dist/fieldeye-detect
bash na_pi.sh wszystko     # everything in a row
```

Results in `/tmp/fieldeye_*`, logs in `/tmp/fieldeye_*.log` (`/tmp` is cleared on reboot).
`requirements.txt` is **not needed** on the Pi (PyTorch/Ultralytics are only for training).
Accuracy of the `.hef` against the `.pt`: `scripts/check_hef.sh` ([`scripts/HAILO.md`](../scripts/HAILO.md), section 3).

First results on the Pi (HEF, 132 test tiles): 4090 detections (4313 with the `.pt` on a Mac),
about 0.03 s per tile.

## 8. Simulated drone and scoring

```bash
python -m mission.drone_sim flight --ewis1 zbior_ewis1_podzial/dataset/images/test [--interwal 0.7]
python -m mission.detect_cli flight results fieldeye_final.onnx --raz
python -m mission.spray_map results/oprysk.geojson flight_prawda.geojson
```

The simulated drone rebuilds an EWIS1 photo from its tiles (`--ewis1`; other sources:
`--pole IMAGE`, `--sztuczne`), places it on the map (top-left corner at `--lat`/`--lon`, north
up, 2.5 mm/px), flies over it with the `stitchlive` simulator and every `--interwal` s writes a
photo and its telemetry **in the drone's format** (camera-like names, `heading_deg`,
`survey.agl_m`, `survey.gsd_mm_per_px`, `roll_deg = pitch_deg = 0`). The true weed positions go
to `flight_prawda.geojson`.

Options: `--zaklocenia` (rotation, jitter, brightness), `--szum-gps-m 1.5` (GPS without RTK),
`--szum-kursu-st`, `--numer N` (another EWIS1 photo), `--lat`/`--lon`, `--wysokosc`, `--seed`.

`spray_map` compares spray points with the truth (hit: closer than 10 cm). Trial results
(synthetic field):

| Telemetry | Hits | Position error |
|---|---|---|
| perfect (also with rotation, jitter, altitude change) | 100% | about 0.1–0.2 cm |
| GPS with RTK (±3 cm) | 100% | about 3 cm |
| GPS without RTK (±1.5 m) | 29% | the same weed appears many times on the map |

With the real model (`fieldeye_final.pt`, threshold 0.15) on an EWIS1 test photo and perfect
telemetry: 69 of 83 weeds hit, 20 extra spray points. Picture: [README](../README.md#what-it-looks-like);
map: [`docs/example/spray_map_simulated.geojson`](example/spray_map_simulated.geojson)
(made-up coordinates, only distances are real).

## 9. Code map

| File | Contents |
|---|---|
| [`mission/detect_cli.py`](../mission/detect_cli.py) | the program: `main` (options, loop), `kandydaci` (new photos, FIFO), `kompletny` / `gotowy` (file fully written), `zbuduj` (detector + tiles), `przetworz` (one photo), `linia` (terminal line), `domyslny_model`, `zamknij` |
| [`mission/spray_map.py`](../mission/spray_map.py) | `czytaj_telemetrie`, `blad_telemetrii`, `przechyl_st`, `plik_telemetrii`, `gsd_m`, `piksel_na_metry`, `UkladLokalny`, `MapaOprysku` (merging, writing), `porownaj` |
| [`mission/detector.py`](../mission/detector.py) | `OnnxDetector`, `HailoDetector`, `YoloDetector`, `FlipTTADetector`, `TiledDetector`, `letterbox`, `zbuduj_z_plikow` |
| [`mission/drone_sim.py`](../mission/drone_sim.py) | simulated drone |
| [`stitchlive/camera.py`](../stitchlive/camera.py) | Arducam camera model (footprint and scale from altitude) |
| [`scripts/fieldeye_detect_main.py`](../scripts/fieldeye_detect_main.py) | binary entry point (detection, `dron`, `porownaj`) |
| [`scripts/build_binary.sh`](../scripts/build_binary.sh) | binary build |
| [`scripts/spakuj_na_pi.sh`](../scripts/spakuj_na_pi.sh), [`scripts/na_pi.sh`](../scripts/na_pi.sh) | Pi package and tests on the Pi |
| [`scripts/przygotuj_folder.sh`](../scripts/przygotuj_folder.sh) | self-contained folder with the binary |
| `tests/test_detect_cli.py`, `test_spray_map.py`, `test_binary.py` | tests (`python -m pytest tests -q`) |

## 10. Common problems

| Symptom | Cause / fix |
|---|---|
| every photo `bez GPS`, message about missing heading | no `heading_deg` in the telemetry (for tests `--kurs-domyslny 0`) |
| `bez GPS` without a message | no `.geojson` with the same name (letter case, extension, same folder) |
| the program waits and nothing happens | it watches a different folder than the camera writes to — check the path in the header |
| `(w kolejce: N)` grows | the model cannot keep up (fewer photos, larger tile, no `--tta`) |
| false detections around 0.15 indoors | the image is nothing like the training data; for tests indoors `--prog 0.4` |
| binary: `Exec format error` | built on another system — build on the one it runs on |
| binary blocked on a Mac | `xattr -dr com.apple.quarantine fieldeye-detect` |
| on the Pi "Unsupported hef version" | HEF from a different compiler version than the Pi's HailoRT (4.23 ↔ DFC 3.33) |
| blurred photos in flight | exposure too long — in flight about 1 ms (`exposure_us ≈ 1000`) |
