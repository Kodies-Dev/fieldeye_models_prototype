# normalization — public datasets → YOLO training data

**Part of the pipeline:** data (first step, before `models/`).

Every public weed dataset comes in its own format: separate colour channels with masks,
Pascal VOC XML, rotated boxes, different class names for the same plant. This package reads
each of them, converts everything into one intermediate format (the *manifest*), and writes a
clean YOLO dataset: deduplicated, split without leaks, tiled to the scale of our drone.

It is run as step 1 of `scripts/final_model.sh`, once per dataset.

---

## Datasets used by the final model

| Dataset | Option | What it is | Output folder | Role in training |
|---|---|---|---|---|
| WeedsGalore | `--weedsgalore` | drone at 5 m, maize, 4 flight days, multispectral camera, masks | `zbior_lotnicze/` | train, **validation and test** (authors' split: 104 / 26 / 26 tiles) |
| Sudars | `--sudars` | photos from the ground, vegetables, Pascal VOC | `zbior_naziemne/` | train only (1438 tiles) |
| EWIS1 | `--ewis1` | another drone at 5 m, **ordinary RGB camera**, maize (sorghum skipped by default), XML + masks | `zbior_ewis1_podzial/` | train (228 tiles); flight day 16 May 2022 held out as the second test (132 tiles) |
| own data | `--wlasne` | photos from our own drone (YOLO or Pascal VOC) | — | prepared for the future; not used yet |

Raw data are expected in `dane/` (ignored by git):

```
dane/
├── weedsgalore-dataset/2023-05-25/images/<id>_R.png ... _RE.png, semantics/, instances/
├── sudars/dataset/raw images/*.jpg + annotations/*.xml
└── EWIS1 Maize, Sorghum & Weed Dataset/     (or dane/ewis1/)  images/, bounding_boxes/*.xml, masks/
```

Downloads: WeedsGalore — https://doidata.gfz.de/weedsgalore_e_celikkan_2024/ (the authors'
split is inside the archive); Sudars — https://data.mendeley.com/datasets/nj4vtk4tt6/1;
EWIS1 — "EWIS1 Maize, Sorghum & Weed Dataset" on Mendeley Data.

---

## How to run

The three commands from `scripts/final_model.sh`:

```bash
python -m normalization.build --weedsgalore dane/weedsgalore-dataset -o zbior_lotnicze --tryb-klas chwasty
python -m normalization.build --sudars dane/sudars/dataset -o zbior_naziemne --tryb-klas chwasty
python -m normalization.build --ewis1 dane/ewis1 -o zbior_ewis1_podzial --tryb-klas chwasty \
    --skala ewis1=0.47 --proporcje 1 0 0 --ewis1-test-dni 20220516
```

Without any data on disk (miniature imitations of the dataset formats, used by the tests):

```python
from normalization.examples import utworz
from normalization.build import zbuduj
zbuduj(utworz("dane_przykladowe"), "zbior_testowy", kafelek=320)
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `-o` | `zbior` | output folder |
| `--tryb-klas` | `binarnie` | `chwasty` = one class "weed", crops become background (**final model**); `binarnie` = weed / crop; `gatunki` = species |
| `--kafelek` | 896 | tile size in pixels; `0` = no tiling |
| `--zakladka` | 0.1 | overlap between neighbouring tiles |
| `--udzial-pustych` | 0.15 | share of tiles without weeds (negative examples), counted over the whole dataset |
| `--prog-dedup` | 5 | max. Hamming distance of perceptual hashes for two images to be duplicates |
| `--skala SOURCE=F` | — | rescale a source to the drone's ground resolution (2.5 mm/px), e.g. `ewis1=0.47` (1.18 → 2.5 mm/px) |
| `--proporcje T V T` | 0.70 0.15 0.15 | split ratios for data without an official split; `1 0 0` = everything to training |
| `--ewis1-test-dni` | — | EWIS1 flight days (YYYYMMDD) held out as test, the rest goes to training |
| `--ewis1-z-sorgo` | off | also use sorghum photos and photos without a crop name |
| `--weedsgalore-kolory` | `liniowe` | `srgb` = gamma-corrected WeedsGalore colours (tested, not used by the final model) |
| `--wlasny-podzial` | off | ignore the authors' splits and split everything by groups |
| `--tylko-wlasne-do-testu` | off | test set only from own data (when there are any) |
| `--splity-weedsgalore`, `--taksonomia`, `--bez-obrazow`, `--seed` | — | custom split files, custom `taxonomy.yaml`, labels only (fast check), random seed |

---

## Pipeline

```
adapters → deduplication → class mode → split → rescale → tiling → YOLO export
```

The order matters twice:

- **Deduplication before the split** — otherwise copies of one image could end up in
  training and in test at the same time.
- **Tiling after the split** — tiles inherit the split of their source image; tiling earlier
  would scatter tiles of one photo across train and test (the same leak on a smaller scale).

| Step | Function | What happens |
|---|---|---|
| read | `adapters.ADAPTERY[name]()` | each dataset → list of `Record` (image path, size, boxes in absolute pixels, group, source) |
| deduplicate | `steps.deduplikuj()` | 64-bit perceptual hash (DCT); images within `--prog-dedup` bits are duplicates; the higher-priority source wins (`wlasne > weedsgalore > sudars > others`); dropped pairs go to `duplikaty.json` |
| class mode | `steps.zastosuj_tryb_klas()` | species → training classes via `taxonomy.yaml` (final model: everything → `chwast`, crops removed) |
| split | `steps.podziel()`, `steps.sprawdz_podzial()` | split by **group** (flight day, mission, folder), deterministic from a SHA-256 of the group name — adding new data never reshuffles the existing split; official splits (WeedsGalore, held-out EWIS1 day) are kept |
| rescale | `steps.przeskaluj()` | only numbers change (size, boxes, GSD); pixels are resized at export |
| tile | `steps.kafelkuj_wszystkie()` | 896 px tiles with 10% overlap; a box stays only if ≥ 40% of it is inside the tile; empty tiles chosen to reach 15% globally |
| export | `export.eksportuj()` | crops and saves `.jpg` + YOLO `.txt` + `data.yaml`; `podglad/` with drawn boxes; `build_info.json` |

### Output

```
zbior_lotnicze/
├── dataset/
│   ├── images/train|val|test/
│   ├── labels/train|val|test/       YOLO: "<class> <x_center> <y_center> <w> <h>" (0–1)
│   └── data.yaml                    ready for Ultralytics
├── manifest.jsonl                   every tile with its boxes (after tiling)
├── manifest_przed_kafelkami.jsonl   the same for whole photos
├── duplikaty.json                   what was dropped as a duplicate and why
├── podglad/                         sample tiles with boxes drawn — look at them before training
└── build_info.json                  sources, all parameters, statistics, unknown labels, split check
```

`build_info.json` is what lets you say months later exactly which data a model was trained on.

---

## Files

| File | Role |
|---|---|
| `build.py` | entry point (`python -m normalization.build`) and `zbuduj()`: runs all steps in the right order, writes `build_info.json` |
| `adapters.py` | one reader per dataset: `weedsgalore()` (channels + instance masks → boxes), `sudars()`, `ewis1()` (fixes swapped sizes, boxes for rotated photos, wrong file names in XML), `wlasne()` (own data), shared `pascal_voc()`; registry `ADAPTERY` |
| `manifest.py` | the intermediate format: `Record` (one image or tile) and `LabeledObject` (one box); JSONL read/write, statistics |
| `taxonomy.py` + `taxonomy.yaml` | species dictionary: aliases from all datasets → one key (e.g. `GP`, `GASPA`, `GASPA_V1`, WeedsGalore mask value 4 → `galinsoga_parviflora`), EPPO codes with growth stage, role weed/crop, rare species folded into `chwast_inny` |
| `steps.py` | deduplication, class mode, split and split check, rescaling, tiling with empty-tile selection |
| `export.py` | the only place where YOLO files are written; box preview; WeedsGalore colour mode |
| `examples.py` | tiny synthetic copies of the dataset formats for tests (`utworz()`) |

Why a manifest: instead of one converter per dataset there is one intermediate format.
An adapter only knows how to read its dataset; everything after it works on `Record`.
Boxes are kept in absolute pixels until export, so tiling is a simple subtraction of the
tile offset. Adding a new dataset means writing one adapter.

---

## What to check after a build

| Message | What to do |
|---|---|
| `UWAGA: N nierozpoznanych etykiet` (unknown labels) | add the alias to `taxonomy.yaml` and rebuild — otherwise those objects silently disappear |
| `BLAD: te same grupy w kilku czesciach podzialu!` | a bug — one of our groups landed in two parts; do not train |
| `UWAGA: oficjalny podzial autorow rozdziela te grupy` | expected for WeedsGalore: its official split is stratified by date, so every day has tiles in train, val and test |

Before training, open a dozen images in `podglad/` for every dataset: a half-pixel shift or
swapped axes is invisible in the metrics but ruins training. Then:

```bash
python -m models.cli sprawdz zbior_lotnicze/dataset/data.yaml
```

**Open point:** the WeedsGalore code only names weeds `weed_1`…`weed_4` (mask values 2–5).
The species assignment in `taxonomy.yaml` comes from the paper (arXiv:2502.13103) and the
order has not been verified. It does not matter for the final model (one class "weed"),
only for `--tryb-klas gatunki`.

---

## Own drone data (when available)

```bash
python -m normalization.build ... --wlasne dane/moje_misje --tylko-wlasne-do-testu
```

Put each mission in its own subfolder — the subfolder name becomes the split group, so frames
of one flight never end up in both train and test:

```
moje_misje/
├── 2026-05-12_field_A/   images/ + labels/  (YOLO + classes.txt, or Pascal VOC)
└── 2026-05-20_field_B/   images/ + labels/
```

Tests: `tests/test_normalization.py`, `tests/test_new_datasets.py`, `tests/test_colors.py`.
