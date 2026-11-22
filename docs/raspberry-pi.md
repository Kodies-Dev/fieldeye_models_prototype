# FIELDEYE on the Raspberry Pi — step by step

How to put the weed detector on the Raspberry Pi 5 that will be mounted on the drone: what the
Pi needs, how to build the single-file program `fieldeye-detect`, how to run it during the flight,
what it takes in and what it gives back. Written for the person who sets up the Pi.

Full reference of the program (every option, telemetry fields, how coordinates are computed):
[`fieldeye-detect.md`](fieldeye-detect.md).

```
ONCE, ON THE PI                                         EVERY FLIGHT
git clone the repository                                 camera + autopilot write into PHOTOS/:
unzip the model from model_fin.zip                         IMG.jpg + IMG.geojson (telemetry)
bash scripts/build_binary.sh hailo                                     │
        │                                                              ▼
        ▼                                                ./fieldeye-detect PHOTOS RESULTS
~/fieldeye-bin/fieldeye-detect                                         │
~/fieldeye-bin/fieldeye_final_1280_h8.hef                              ▼
(the repository is no longer needed)                     RESULTS/: photos with boxes,
                                                           oprysk.geojson / oprysk.csv (weed map)
```

At run time the program needs **no Python, no libraries and no project folder** — only the
binary, the `.hef` model and the Hailo driver on the Pi.

---

## 1. What you need

| | |
|---|---|
| hardware | Raspberry Pi 5 + **AI HAT+ with Hailo-8** (26 TOPS) |
| system | Raspberry Pi OS 64-bit (Bookworm) |
| Hailo software | package `hailo-all` with **HailoRT 4.23** — the model was compiled for this version |
| for building only | `python3-venv`, `python3-pip`, `python3-opencv`, `python3-numpy`, `git`, `unzip`, internet (PyInstaller is downloaded once), about 1 GB of free disk |
| model | `fieldeye_final_1280_h8.hef` (5.4 MB) — inside `model_fin.zip` in the repository |

**Version check.** The `.hef` file only loads on the HailoRT version it was compiled for
(compiler 3.33 ↔ HailoRT 4.23). If `hailortcli --version` shows another version, the program stops
with `Unsupported hef version`; then either install HailoRT 4.23 or recompile the model with the
matching compiler ([`scripts/HAILO.md`](../scripts/HAILO.md)).

---

## 2. Install the Hailo software (once)

```bash
sudo apt update
sudo apt install -y hailo-all python3-venv python3-pip python3-opencv python3-numpy git unzip
sudo reboot

hailortcli fw-control identify      # must show the Hailo-8 board and its firmware
hailortcli --version                # must be 4.23.x
```

## 3. Get the code and the model

```bash
git clone https://github.com/Kodies-Dev/fieldeye_models_prototype.git ~/fieldeye
cd ~/fieldeye
unzip -j model_fin.zip '*.hef'      # → ~/fieldeye/fieldeye_final_1280_h8.hef

hailortcli parse-hef fieldeye_final_1280_h8.hef   # input 1280×1280×3, output with NMS
hailortcli run fieldeye_final_1280_h8.hef         # speed: tiles per second
```

Alternative with test data: on the Mac, `bash scripts/spakuj_na_pi.sh` builds
`fieldeye_pi.zip` (code, model, 132 labelled EWIS1 test tiles, `na_pi.sh`). On the Pi,
`bash na_pi.sh wszystko` then checks everything in one go: setup, detection on the test tiles,
a simulated flight with GPS, live speed, and the binary build
([`fieldeye-detect.md`, section 7](fieldeye-detect.md#7-raspberry-pi)).

## 4. Quick test from source (optional, before building)

No data needed — a simulated drone writes photos and telemetry exactly like the real one:

```bash
cd ~/fieldeye
python3 -m mission.drone_sim /tmp/lot --sztuczne --interwal 0
python3 -m mission.detect_cli /tmp/lot /tmp/wyniki fieldeye_final_1280_h8.hef --raz
python3 -m mission.spray_map /tmp/wyniki/oprysk.geojson /tmp/lot_prawda.geojson
```

This checks the whole chain — Hailo, telemetry, coordinates, weed map. The synthetic field has
no real weeds, so the counts say nothing about the model's accuracy (that is
`bash scripts/check_hef.sh` with the EWIS1 test tiles, [`scripts/HAILO.md`](../scripts/HAILO.md)).

## 5. Build the binary

```bash
cd ~/fieldeye
bash scripts/build_binary.sh hailo
```

What the script does:

1. creates `.venv-build-hailo` **with access to the system packages** — `hailo_platform`, OpenCV
   and NumPy come from `apt`, only PyInstaller is installed from pip;
2. runs PyInstaller on `scripts/fieldeye_detect_main.py`: Python, the FIELDEYE code and the
   libraries are packed into **one file**; PyTorch, Ultralytics, ONNX Runtime, Streamlit and other
   training-only packages are left out; full log in `build-hailo.log`;
3. runs the result with `--help` and prints its size.

Result: **`dist/fieldeye-detect`** (about 170 MB). With `bash scripts/build_binary.sh hailo --katalog`
you get a folder (`dist/fieldeye-detect/fieldeye-detect` + `_internal/`) instead of one file — it
starts 1–2 s faster, because a single file first unpacks itself into a temporary folder.

**Build on the Pi itself.** A binary contains machine code for one system: one built on a Mac or
a PC will not run on the Pi (`Exec format error`), and the `hailo` variant needs the Hailo
libraries that only the Pi has.

## 6. Install it

```bash
mkdir -p ~/fieldeye-bin
cp ~/fieldeye/dist/fieldeye-detect ~/fieldeye/fieldeye_final_1280_h8.hef ~/fieldeye-bin/
```

That is all the drone needs; `~/fieldeye` can be deleted. A model file named
`fieldeye_final_1280_h8.hef` lying **next to the binary** is found automatically, so the model
argument can be left out. To replace the model, replace the file (or give another path as the
third argument).

---

## 7. Run

```
fieldeye-detect PHOTOS [RESULTS] [MODEL] [options]     detection (the normal mode)
fieldeye-detect dron FOLDER [options]                  simulated drone (tests without a drone)
fieldeye-detect porownaj OPRYSK PRAWDA                 score detected weed points against true weeds
```

During the flight:

```bash
~/fieldeye-bin/fieldeye-detect ~/loty/2026-11-22 ~/wyniki/2026-11-22
```

The program watches `PHOTOS` until Ctrl+C. Every photo is processed as soon as it is completely
written, in the order of writing, and all results are rewritten after every photo — stopping or a
power loss never loses what was already processed. Use a **new folder for every flight**
(or `--tylko-nowe`), otherwise photos of the previous flight are processed again.

| Argument | Meaning |
|---|---|
| `PHOTOS` | folder the camera writes to; created if missing (the camera may start later) |
| `RESULTS` | folder for results; created if missing; must differ from `PHOTOS`; without it results only appear in the terminal |
| `MODEL` | path to the `.hef` file; omitted → `fieldeye_final_1280_h8.hef` next to the binary |

Most useful options (all of them: `fieldeye-detect --help`, or
[`fieldeye-detect.md`, section 1](fieldeye-detect.md#1-running)):

| Option | Default | Meaning |
|---|---|---|
| `--prog P` | 0.15 | confidence threshold; deliberately low — a missed weed is worse than a false alarm |
| `--tta` | off | second pass on a mirrored image: about 30% fewer false alarms, about 2× slower |
| `--kafelek PX` | from telemetry | tile size; by default each tile covers 1.5 m of field (the training scale); 950 px without telemetry |
| `--raz` | off | process what is in the folder and exit (e.g. after landing) |
| `--tylko-nowe` | off | ignore photos that were already in the folder at start |
| `--timeout S` | — | exit after S seconds without a new photo |
| `--maks-przechyl DEG` | 5 | photos tilted more than this are left off the map |
| `--promien-scalania M` | 0.10 | detections closer than this are one weed (one point on the map) |
| `--czekaj-telemetrie S` | 3 | how long to wait for a photo's `.geojson` before processing it without GPS |
| `--kurs-domyslny DEG` | — | heading to assume when telemetry has none (tests only) |
| `--json` | off | standard output: only one JSON line per photo, for another program |
| `--jsonl` | off | also `RESULTS/wyniki.jsonl` |

Exit code: `0` = normal end (Ctrl+C, `--raz`, `--timeout`); `2` = wrong arguments (model missing
or not `.pt` / `.onnx` / `.hef`, `RESULTS` equal to `PHOTOS`, `--jsonl` without `RESULTS`).

---

## 8. What it needs from the drone

The program does not control the camera and does not read GPS — the drone-side software does that
and leaves files in `PHOTOS`. This folder is the whole interface.

**Photos.** `.jpg` (also `.png`, `.tif`, `.bmp`), straight down. Write each file under a temporary
name (starting with `.` or `~`, or ending with `.part`) and rename it when complete — the program
skips such names, and in any case waits until the size has not changed for 1 s and the JPEG end
marker is present.

**Telemetry.** Next to every photo, a GeoJSON file with **the same name** and the extension
`.geojson` (`20261109T142859.998Z.jpg` → `20261109T142859.998Z.geojson`), written the same atomic
way, at most 3 s after the photo:

```json
{"type": "FeatureCollection",
 "features": [{"type": "Feature",
   "geometry": {"type": "Point", "coordinates": [18.73101322, 50.29004902]},
   "properties": {"image": "20261109T142859.998Z.jpg", "time_utc": "2026-11-09T14:28:59.998Z",
                  "heading_deg": 7.5, "roll_deg": 0.4, "pitch_deg": -1.2,
                  "fix": true, "fix_quality": 2,
                  "survey": {"agl_m": 5.0}}}]}
```

| Field | Required | Why |
|---|---|---|
| `coordinates` = `[longitude, latitude]` | yes | drone position = centre of the photo on the field |
| `survey.agl_m` | yes, or `survey.gsd_mm_per_px` | altitude **above the field** (not above sea level) → photo scale |
| `heading_deg` | yes | which way the top of the photo points (0 = north, clockwise) |
| `roll_deg`, `pitch_deg` | recommended | photos tilted over 5° are left off the map |
| `fix`, `fix_quality` | recommended | without a GPS fix the photo goes without coordinates |

**Note for the drone side:** the example telemetry received so far
([`tests/dane/telemetria_dron.geojson`](../tests/dane/telemetria_dron.geojson)) has **no
`heading_deg`**. Without it every photo goes without GPS (the GPS course is used only when the
drone moves at least 1 m/s; in hover it is noise). The autopilot's heading has to be added. For
tests without it: `--kurs-domyslny 0`.

**Camera.** The scale is computed from `agl_m` for the Arducam 64MP in 16 MP mode (sensor 7.4 mm,
lens 5.1 mm, 4624 × 3472). With another camera or resolution, send `survey.gsd_mm_per_px`
(mm of field per pixel) instead. Recommended settings: full 16 MP field of view, fixed focus,
exposure at most 1/1000 s (the example telemetry has 60 ms — in flight that blurs the image),
camera pointing straight down.

---

## 9. What it gives back

**Terminal**, one line per photo:

```
[14:29:01] 20261109T142859.998Z.jpg   4624x3472   chwasty:   17   max 0.91   0.84 s   GPS 50.290049, 18.731013   oprysk: 15 pkt (+15)   (w kolejce: 1)
```

`chwasty` = weeds found, `max` = highest confidence, the time per photo, the drone position,
`oprysk` = detected weeds (points) so far, `w kolejce` = photos waiting (if it keeps growing, the Pi does not
keep up with the camera). Messages are in Polish; the table in
[`fieldeye-detect.md`, section 5](fieldeye-detect.md#5-safeguards) translates them.

**Folder `RESULTS`**, refreshed after every photo:

| File | Content |
|---|---|
| `<photo name>.jpg` | the photo with pink boxes around the weeds |
| `oprysk.geojson` | **detected weeds** — one point per weed even if it was on several photos; `promien_m` (radius: half the weed + 5 cm), `pewnosc` (confidence), `liczba_zdjec` (photos) |
| `oprysk.csv` | the same as a table: `id,szerokosc,dlugosc,promien_m,pewnosc,liczba_zdjec` (latitude, longitude, ...) |
| `chwasty.geojson` | every single detection on every photo |
| `trasa.geojson` | the drone's route and the photo positions |
| `wyniki.jsonl` | with `--jsonl`: one line per photo with boxes in pixels and coordinates |

Without telemetry only the photos with boxes are written. The `.geojson` files open on
[geojson.io](https://geojson.io) or in QGIS; `oprysk.csv` is the same list as a table.
The name `oprysk` (Polish: spraying) is kept from the code: these files hold the detected weeds.
What to do with them (e.g. spraying) is decided on the drone / frontend side.

**For another program** (`--json`): standard output carries one JSON object per photo —
`plik`, `szer`, `wys`, `chwasty`, `sekundy`, `wykrycia` (`[x0, y0, x1, y1, confidence]` in
pixels) and, with valid telemetry, `gps` and `wykrycia_geo` (`[lat, lon]` of every detection).
All other messages then go to standard error.

---

## 10. Start with the Pi (optional)

A `systemd` service starts the program on boot and restarts it after an error. Replace `USER`
and the folders:

```ini
# /etc/systemd/system/fieldeye.service
[Unit]
Description=FIELDEYE weed detection
After=multi-user.target

[Service]
User=USER
ExecStart=/home/USER/fieldeye-bin/fieldeye-detect /home/USER/zdjecia /home/USER/wyniki --tylko-nowe
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now fieldeye     # start now and on every boot
journalctl -u fieldeye -f                # the terminal lines live
sudo systemctl stop fieldeye             # stop
```

Only one program at a time can use the Hailo-8: stop the service before running
`check_hef.sh`, `na_pi.sh` or a second copy by hand.

## 11. Speed

The first measurement on the Pi: about 0.03 s per tile. A 16 MP photo from 5 m is cut into
30 tiles of 950 px (1.5 m of field, 15% overlap), so one photo takes roughly 1 s, plus reading
the 16 MP JPEG; with `--tta` about twice as long. At 3 m/s and 70% overlap the camera takes a
photo about every 0.73 s — so the Pi may **not keep up** at that speed. Measure it on real photos
before the flight:

```bash
~/fieldeye-bin/fieldeye-detect FOLDER_WITH_REAL_PHOTOS /tmp/wyniki --raz
# last line: "Podsumowanie: N zdjec, ... srednio X s na zdjecie" (average X s per photo)
```

If X is above the photo interval: fly slower, take photos less often (lower overlap — 50% is
still enough for the map, which does not need a mosaic), and do not use `--tta`.

---

## 12. Common problems

| Symptom | Cause / fix |
|---|---|
| `hailortcli: command not found` | `hailo-all` not installed (section 2) |
| `Unsupported hef version` | HailoRT on the Pi is not 4.23 (section 1) |
| error that the Hailo device is busy / not available | another program uses the Hailo-8 (the service, `check_hef.sh`); only one at a time |
| `brak modelu: ...` (model missing) | no `.hef` next to the binary and no path given |
| `Exec format error` | binary built on another system — build it on the Pi |
| `Permission denied` | `chmod +x ~/fieldeye-bin/fieldeye-detect` |
| build fails | the end of `~/fieldeye/build-hailo.log`; `python3 -c "import hailo_platform, cv2, numpy"` must work |
| every photo `bez GPS` (without GPS) | no `heading_deg`, no `.geojson` with exactly the same name, or no GPS fix — the reason is printed next to the photo |
| `(w kolejce: N)` keeps growing | the Pi does not keep up (section 11) |
| `PRZECHYL ... poza mapa oprysku` | the photo was tilted more than `--maks-przechyl` |

Code: [`scripts/build_binary.sh`](../scripts/build_binary.sh),
[`scripts/fieldeye_detect_main.py`](../scripts/fieldeye_detect_main.py) (entry point),
[`mission/detect_cli.py`](../mission/detect_cli.py) (the program),
[`mission/spray_map.py`](../mission/spray_map.py) (telemetry and map),
[`mission/detector.py`](../mission/detector.py) (`HailoDetector`).
