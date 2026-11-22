# scripts — the final model, Hailo, the binary and test data

**Part of the pipeline:** tooling around every part. All scripts are run from the project
root folder.

---

## Final model — one script

```bash
caffeinate -i bash scripts/final_model.sh     # about 4 h with training (caffeinate keeps a Mac awake)
streamlit run app.py
```

`final_model.sh` rebuilds the final model and everything `app.py` needs from the raw data in `dane/`:

1. **normalization** (`normalization/`): WeedsGalore → `zbior_lotnicze/`, Sudars → `zbior_naziemne/`,
   EWIS1 → `zbior_ewis1_podzial/` (16 May 2022 as test); stops if any label is unknown,
2. **training** (`models/`): oversampling of the early days, plan `plan_dane_ewis1_bt.json`,
   → `modele_dane_ewis1_bt/model.pt`, copied to `models/weights/fieldeye_final.pt`,
3. **evaluation** on the EWIS1 and WeedsGalore tests, without and with TTA → `wyniki/model_koncowy/*.json`,
4. **ONNX package for Hailo** → `hailo/hailo_final.zip` (ONNX 1280, opset 11, 128 calibration
   images, `compile_hef.py`),
5. **results table**.

Needs an active `.venv` with `requirements.txt`. Steps already done are skipped, so the script
can be stopped and started again.

---

## Hailo-8 (other machines)

| File | Where | What it does |
|---|---|---|
| [`HAILO.md`](HAILO.md) | — | step-by-step compilation to `.hef`: Mac → WSL → Raspberry Pi |
| [`HAILO_PIPELINE.md`](HAILO_PIPELINE.md) | — | how it works inside: NMS configuration, the cut at the head, INT8 quantization, `HailoDetector`, the check |
| `compile_hef.py` | WSL (Ubuntu 22.04, Hailo DFC 3.33, Model Zoo 2.17) | ONNX → HEF directly through the Dataflow Compiler (parse → optimize INT8 → compile), with its own NMS configuration: image 1280, 1 class, threshold 0.05, up to 300 boxes |
| `check_hef.sh` | Raspberry Pi | accelerator, `parse-hef`, speed, accuracy on the EWIS1 test and comparison with the `.pt` (AP50 0.812 / 0.817) |

---

## The `fieldeye-detect` binary and the Pi

| File | What it does |
|---|---|
| `fieldeye-detect.sh` | photos in a folder → weed model → result in the terminal, from source (activates `.venv` itself) |
| `fieldeye_detect_main.py` | entry point of the binary: detection, `dron` (simulated drone with GPS), `porownaj` (score spray points against ground truth) |
| `build_binary.sh` | builds the binary with PyInstaller: `onnx` (Mac / Ubuntu, no PyTorch), `hailo` (Pi with `hailo-all`), `pt` (PyTorch); **build on the system it will run on** |
| `przygotuj_folder.sh` | self-contained folder: binary + models + examples + `uruchom.sh`; no project or Python needed to run it |
| `spakuj_na_pi.sh` | on the Mac: runs the binary tests and packs code, 132 EWIS1 test tiles and the `.hef` into `~/Downloads/fieldeye_pi.zip` |
| `na_pi.sh` | on the Pi, inside the unpacked package: `sprawdz` (check setup), `test` (132 tiles through `.hef`), `gps` (simulated flight + spray map + scoring), `na-zywo` (photo every 0.7 s: does Hailo keep up), `binarka` (build and run the binary), `wszystko` (all) |

Full description: [`docs/fieldeye-detect.md`](../docs/fieldeye-detect.md).

---

## Mosaic stitching and test data

| File | What it does |
|---|---|
| `generate_field.py` | procedural drone photos of a field (carrot, maize, beet) with YOLO labels of the weeds and a preview — known ground truth for testing |
| `make_test_data.py` | older, simpler generator: a synthetic field cut into overlapping frames (`make dane`) |
| `simulate_flight.sh` | copies images into a folder at intervals, like a drone in flight (copy to a temporary name, then rename — the watching program never sees half a file) |

```bash
python scripts/generate_field.py dane_testowe --liczba 6 --uprawa mix
python -m stitchlive.cli watch inflow --renderuj-co 4 -o results        # terminal 1
./scripts/simulate_flight.sh dane/test/images inflow 1                  # terminal 2
```
