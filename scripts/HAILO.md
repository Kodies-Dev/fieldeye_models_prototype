# FIELDEYE final model on Hailo-8 (Raspberry Pi 5 + AI HAT+ 26 TOPS)

| | |
|---|---|
| model | `models/weights/fieldeye_final.pt` (YOLOv8n, 1 class "weed", input 1280) |
| HEF | `fieldeye_final_1280_h8.hef` (5.4 MB, Hailo-8, HailoRT 4.23) — inside `model_fin.zip` (`*.hef` files are ignored by git) |
| path | `.pt` → ONNX + calibration images (Mac) → HEF (WSL, Ubuntu 22.04) → run (Raspberry Pi) |

The HEF does not replace the `.pt`: training, evaluation and the simulator on a computer keep
using the `.pt`. What happens inside each step: [`HAILO_PIPELINE.md`](HAILO_PIPELINE.md).

---

## 1. Mac: ONNX and calibration images

Step 4 of `scripts/final_model.sh` does this and writes `hailo/hailo_final.zip`. By hand:

```bash
mkdir -p hailo/final
python -m models.cli onnx models/weights/fieldeye_final.pt -o hailo/final/fieldeye_final_1280.onnx --wejscie 1280 --opset 11

rm -rf /tmp/kal_src && mkdir -p /tmp/kal_src
cp zbior_lotnicze/dataset/images/train/*.jpg zbior_ewis1_podzial/dataset/images/train/*.jpg /tmp/kal_src/
python -m models.cli kalibracja /tmp/kal_src -o hailo/final/kalibracja_1280 --liczba 128 --rozmiar 1280
cp scripts/compile_hef.py hailo/final/
cd hailo/final && zip -r ~/Desktop/hailo_final.zip . && cd ../..
```

- the export must print the output shape `(1, 5, 33600)`; a fixed input and opset 11 are what the Hailo compiler recommends,
- calibration images come only from the training parts (WeedsGalore + EWIS1), never from a test set.

## 2. WSL: compilation to HEF

The Hailo compiler runs only on Linux x86: Windows + WSL with Ubuntu 22.04, Python 3.10,
at least 16 GB of memory (RAM + swap, `.wslconfig`).

| Component | Version | Notes |
|---|---|---|
| HailoRT on the Raspberry Pi | 4.23.0 | the compiler must come from the same release |
| Hailo Dataflow Compiler | 3.33.0 | `hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl` |
| Hailo Model Zoo | 2.17.0 | `hailo_model_zoo-2.17.0-py3-none-any.whl` |
| do not use | 3.34.0 / 5.x | for HailoRT 4.24 / for Hailo-10 and Hailo-15 |

```bash
sudo apt install -y python3.10-venv python3-dev python3-tk graphviz graphviz-dev git build-essential
python3 -m venv ~/hailo_venv && source ~/hailo_venv/bin/activate
pip install --upgrade pip wheel
pip install hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl hailo_model_zoo-2.17.0-py3-none-any.whl

# the Model Zoo package lacks the NMS configs - copy them from the v2.17 branch
git clone --depth 1 --branch v2.17 https://github.com/hailo-ai/hailo_model_zoo.git ~/hmz_src
cp -r ~/hmz_src/hailo_model_zoo/cfg/postprocess_config ~/hailo_venv/lib/python3.10/site-packages/hailo_model_zoo/cfg/

# compilation (after unpacking hailo_final.zip into ~/fieldeye/final)
cd ~/fieldeye/final
python compile_hef.py fieldeye_final_1280.onnx kalibracja_1280 -o fieldeye_final_1280_h8.hef --rozmiar 1280 2>&1 | tee compile_final.log
```

**Why `compile_hef.py` and not just `hailomz compile`:** the Model Zoo NMS config uses image
size 640, threshold 0.2 and 100 boxes per class. The script takes the YOLOv8n model script and
NMS template from Model Zoo and sets image 1280, 1 class, threshold 0.05 and up to 300 boxes,
the six final Conv layers of the head (`/model.22/cv2.x`, `cv3.x`) and the network name
`yolov8n` (the NMS config refers to it). Stages: parse → optimize (INT8 quantization on the
calibration images) → compile; a `.har` file of each stage is kept.

The final model fits in one accelerator context (47% compute, 28% memory). Without a GPU and
with fewer than 1024 calibration images the compiler falls back to optimization level 0
(64 images, basic quantization) — it works, but may lose more accuracy than full optimization
(GPU in WSL and about 1024 images).

| Symptom | Fix |
|---|---|
| `Temporary failure resolving` | manual DNS in `/etc/resolv.conf` (`nameserver 8.8.8.8`) |
| `No such file ... yolov8n_nms_config.json` | copy `postprocess_config` from the v2.17 branch |
| `The layer named yolov8n/conv41 doesn't exist` | network name `yolov8n` (set in `compile_hef.py`) |
| `invalid choice: 'hailo8'` | Model Zoo 2.17 instead of 5.x |
| `Unsupported hef version` (on the Pi) | use the compiler from the HailoRT release installed on the Pi |

## 3. Raspberry Pi 5 with AI HAT+: check

`fieldeye_final_1280_h8.hef` in the repository folder and the EWIS1 test set with a `data.yaml`
(`zbior_ewis1_podzial/dataset/data.yaml` from `final_model.sh`, or a test package `test_ewis1/`):

```bash
bash scripts/check_hef.sh                                                     # default: test_ewis1/data.yaml
bash scripts/check_hef.sh fieldeye_final_1280_h8.hef zbior_ewis1_podzial/dataset/data.yaml
```

The package made by `scripts/spakuj_na_pi.sh` contains the test tiles but no `data.yaml`;
inside it, create one first:

```bash
printf 'path: .\ntrain: images/test\nval: images/test\ntest: images/test\nnames: {0: chwast}\n' \
    > zbior_ewis1_podzial/dataset/data.yaml
```

The script checks the accelerator (`hailortcli fw-control identify`), the file (`parse-hef`:
input 1280×1280×3, output with NMS), speed (`hailortcli run`) and accuracy on the EWIS1 test
without and with TTA, compared with the `.pt`:

| | AP50 `.pt` | recall | precision |
|---|---|---|---|
| without TTA | 0.812 | 79.1% | 81.1% |
| with TTA | 0.817 | 78.4% | 84.3% |

The difference is the cost of quantization and compilation: up to about 2 pp is good, up to
5 pp acceptable; more means the compilation or the calibration images need a look. The NMS on
the chip drops detections below 0.05, so the HEF's AP50 may come out about 0.01 lower for that
reason alone. Not measured yet; the first run on the Pi gave 4090 detections on the 132 test
tiles (4313 with the `.pt` on a Mac), about 0.03 s per tile.

## 4. Use in flight

```bash
./fieldeye-detect PHOTOS RESULTS fieldeye_final_1280_h8.hef                 # binary
python3 -m mission.detect_cli PHOTOS RESULTS fieldeye_final_1280_h8.hef     # from source
```

Defaults: threshold 0.15, tile 950 px (1.5 m of field from 5 m). With a `.hef` the network
input size is built into the file. Full description (telemetry, spray map, Pi package):
[`docs/fieldeye-detect.md`](../docs/fieldeye-detect.md).
