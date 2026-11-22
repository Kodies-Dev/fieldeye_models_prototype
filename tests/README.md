# tests — automatic tests

**Part of the pipeline:** all parts. About 215 tests; none of them needs the real datasets,
a GPU or a Hailo accelerator — they build small synthetic inputs with a known answer.

```bash
python -m pytest tests -q                                        # all, 1–3 min
python -m pytest tests -q --deselect tests/test_gui_flight.py    # without the GUI tests, faster
python -m pytest tests/test_hailo.py -v                          # one file
```

Run from the project root. With `ultralytics` installed one test is skipped on purpose (it
checks the error message shown when the library is missing).

| File | Tests | Part | What it checks |
|---|---|---|---|
| `test_normalization.py` | 32 | data | class aliases, EPPO codes with growth stage, deduplication, group split without leaks, tiling, YOLO format, CLI |
| `test_new_datasets.py` | 9 | data | EWIS1 adapter: maize only, swapped sizes in XML, rotated boxes, wrong file names, held-out day, build with rescaling |
| `test_colors.py` | 5 | data | WeedsGalore colours: linear unchanged, sRGB gamma correction (16- and 8-bit), `--weedsgalore-kolory` |
| `test_models.py` | 32 | model | training plan, dataset check, label reading, AP50 and metrics at a threshold, per-day results, calibration images, device choice (training itself needs Ultralytics and is not run) |
| `test_season.py` | 4 | model | oversampling of early days: number of copies, no leak to val/test, re-running adds nothing, command options |
| `test_hailo.py` | 17 | model / Hailo | `HailoDetector` without hardware: letterbox, decoding Hailo NMS output, detector choice by extension, AP50 of any detector, command defaults, syntax of `final_model.sh` and `check_hef.sh` |
| `test_compile_hef.py` | 4 | Hailo | `compile_hef.py` without the compiler: NMS configuration for 1280, model script patching, RGB calibration images |
| `test_inference.py` | 10 | deployment | IoU, WBF, TTA flip returns boxes to the same place, tiles (offsets and overlap duplicates), dummy detector, `zbuduj_z_plikow` |
| `test_detect_cli.py` | 9 | deployment | `fieldeye-detect`: complete vs. truncated JPEG, waiting for a stable file, hidden/temporary files skipped, FIFO by write time, outputs, arguments |
| `test_spray_map.py` | 14 | deployment | telemetry format of the real drone, heading rules, missing GPS fix, tilt limits, pixel → metres, merging into spray points, full chain drone → detection → spray map → scoring |
| `test_binary.py` | 6 | deployment | ONNX detector, model next to the binary, no heavy libraries imported, `dron` / `porownaj` commands, syntax of the shell scripts |
| `test_sim_tiles.py` | 4 | simulator | EWIS1 photo rebuilt from its tiles pixel-exact, a weed from the tile overlap becomes one box |
| `test_gui_flight.py` | 5 | GUI | tab 4 of `app.py` (Streamlit AppTest): flight with detection, tiles grouped by day, drone camera mode |
| `test_global_stitcher.py` | 12 | mosaic | `global` method: rotated frames, pair accuracy, blurred frames, a foreign frame, anchoring against shrinking, frame → mosaic mapping |
| `test_stitch.py` | 15 | mosaic | `basic` and `detail` methods, sliding window, CLI |
| `test_flight_sim.py` | 17 | simulator | frame cut matches the source, margins, repeatability, verdict thresholds of the evaluation |
| `test_camera.py` | 11 | simulator | Arducam model at 5 m, 4:3 frames, motion blur, true boxes in a rectangular frame |
| `test_generate_field.py` | 9 | test data | synthetic field generator with YOLO labels |

Test data: `tests/dane/telemetria_dron.geojson` — an example telemetry file in the format of
the real drone (the reference for `mission/spray_map.py`).

## Principle

Test inputs are built from a **known** image or from known boxes, so the correct result is
known in advance. Tests check not only that the code runs, but that the result matches the
truth — e.g. the position of a box after mapping back, the PSNR of a mosaic, or the distance
of a spray point from the true weed.
