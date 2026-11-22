# Normalization pipeline — how it works inside

How `python -m normalization.build` turns three public datasets, each in its own format, into
one YOLO training set. For the commands, options and output layout see
[`README.md`](README.md); this page explains the flow and the hardest parts of the code.

**Input:** raw datasets in `dane/` (WeedsGalore, Sudars, EWIS1). **Output:** `zbior_*/dataset/`
(YOLO), `manifest*.jsonl`, `build_info.json`. **Runs on:** the training computer (Mac / PC),
step 1 of `scripts/final_model.sh`.

---

## 1. Flow

Everything is driven by `build.zbuduj()`. Between the adapters and the export the pipeline
works only on numbers in `Record` objects (paths, sizes, boxes, split) — pixels are read,
resized and cropped once, at the very end.

```
build.zbuduj(zrodla, wyjscie, ...)
│
├─ Taxonomy(taxonomy.yaml)                     one species dictionary for the whole build
│
├─ 1. read       ADAPTERY[name](path, tax, **options)  → list[Record]
│                  weedsgalore()  _R/_G/_B channel files + semantic and instance masks → boxes
│                  sudars()       Pascal VOC XML  (through pascal_voc())
│                  ewis1()        Pascal VOC XML + masks, maize only, fixes the box layout
│
├─ 2. dedup      steps.deduplikuj()            perceptual hash, near-identical images dropped
├─ 3. classes    steps.zastosuj_tryb_klas()    species → training class ("chwast"); crops removed
├─ 4. split      steps.podziel()               train / val / test per GROUP (deterministic)
│                steps.sprawdz_podzial()       leak check → build_info.json
├─ 4b. rescale   steps.przeskaluj()            EWIS1 × 0.47 (numbers only)
│                → manifest_przed_kafelkami.jsonl
├─ 5. tile       steps.kafelkuj_wszystkie()    896 px tiles, 10% overlap, 15% empty tiles
│                → manifest.jsonl
└─ 6. export     export.eksportuj()            pixels: load → resize → crop → .jpg + .txt, data.yaml
                 export.podglad_ramek()        12 tiles with drawn boxes → podglad/
                 export.zapisz_info()          build_info.json (written last = "build finished")
```

The order is not arbitrary: deduplication must come **before** the split (otherwise copies of
one image land in train and test), and tiling **after** the split (tiles inherit the split of
their photo, so tiles of one photo can never be scattered across train and test).

Results of the final build:

| Dataset | Folder | train / val / test (tiles) |
|---|---|---|
| WeedsGalore | `zbior_lotnicze/` | 104 / 26 / 26 (authors' split) |
| Sudars | `zbior_naziemne/` | 1438 train used (split 70/15/15 by group; only `train` goes to training) |
| EWIS1 maize | `zbior_ewis1_podzial/` | 228 / 0 / 132 (16 May 2022 held out) |

---

## 2. Technologies

| Technology | Used for | Where |
|---|---|---|
| **OpenCV** (`opencv-contrib-python`) | reading images (also 16-bit channels with `IMREAD_UNCHANGED` and a fast 1/8-size read for image sizes), connected components on masks, DCT for the perceptual hash, `INTER_AREA` resizing, JPEG writing, drawing preview boxes | `adapters.py`, `steps.py`, `export.py` |
| **NumPy** | masks and instance IDs, majority vote of a mask under an instance, percentile colour stretch, seeded random generator (`default_rng`) | everywhere |
| **PyYAML** | `taxonomy.yaml`; class names from a `data.yaml` next to own data | `taxonomy.py`, `adapters.py` |
| **Pillow** | exact EWIS1 image size from the file header without decoding a 40 MB PNG | `adapters._ewis1_uklad()` |
| Python standard library | `xml.etree.ElementTree` (Pascal VOC), `hashlib.sha256` (split), `dataclasses` (`Record`), JSON / JSONL, `argparse`, `logging`, `re` | — |
| **YOLO label format** (Ultralytics layout) | the only output format: `images/<split>/`, `labels/<split>/`, `data.yaml` | `export.py` |

---

## 3. Files in this process

| File | Role in the flow |
|---|---|
| `build.py` | `zbuduj()` runs the steps in order and writes `build_info.json`; `main()` is the command line |
| `adapters.py` | step 1: one reader per dataset, registry `ADAPTERY` |
| `taxonomy.py` + `taxonomy.yaml` | names from every dataset → one species key; species → training class |
| `manifest.py` | the intermediate format: `Record`, `LabeledObject`, JSONL read / write, statistics |
| `steps.py` | steps 2–5: deduplication, class mode, split, rescaling, tiling |
| `export.py` | step 6: the only place where pixels are written |
| `examples.py` | miniature fake datasets for the tests |

---

## 4. Key classes and functions

### `Record` and `LabeledObject` — the manifest (`manifest.py`)

One `Record` is one photo (or, after step 5, one tile); one `LabeledObject` is one plant box.
Three design decisions make the rest of the pipeline simple:

- **Boxes are kept in absolute pixels** (`xyxy = [x0, y0, x1, y1]`) until the export. Tiling is
  then a subtraction of the tile corner, rescaling a multiplication.
- **`grupa` is the split key.** A group is a flight day, a mission or a folder — never a single
  photo. Neighbouring frames of one flight overlap by 60–80%, so a per-photo split would put the
  same piece of field into train and test and inflate the score.
- **Pixels are lazy.** `skala` (rescale factor) and `kafelek` (tile corner) only *describe* what
  the export has to do with the image; the image file itself is not touched before step 6.

### `Taxonomy` — names → species → class (`taxonomy.py`)

The same weed is `GP` in Sudars, mask value `4` in WeedsGalore and `GASPA_V1` as an EPPO code
with growth stage. `Taxonomy` builds one alias table from `taxonomy.yaml` and answers:

| Call | Example | Used by |
|---|---|---|
| `klucz(name)` | `"ZEAMX_V1"` → `"zea_mays"` | `pascal_voc()`, `wlasne()` |
| `z_maski(dataset, value)` | `("weedsgalore", 4)` → `"galinsoga_parviflora"` | `weedsgalore()` |
| `etykieta(key, mode)` | `("zea_mays", "chwasty")` → `None` (crop becomes background) | `zastosuj_tryb_klas()` |
| `raport_nieznanych()` | `[("dandelionn", 3)]` | `build_info.json` |

`klucz()` normalizes the name (lower case, letters and digits only, so `Zea mays`, `zea_mays`
and `ZEA-MAYS` are equal). If the whole name is unknown it tries the part before the first `_`
(EPPO code without the growth stage). A name that still does not match is **counted, not
silently dropped** — `final_model.sh` stops when `nierozpoznane_etykiety` is not empty.

### `ramki_z_maski()` and `weedsgalore()` — boxes from masks (`adapters.py`)

WeedsGalore has no boxes, only masks: a semantic mask (pixel value = class) and an instance mask
(pixel value = plant number). The instance PNG stores the plant number in its three colour
channels, so the adapter first rebuilds one integer per pixel:

```python
ins = (R << 16) | (G << 8) | B
```

`ramki_z_maski(maska, instancje, min_pole=15)` then:

1. **with instances** — for every plant number: the pixels of that plant (`np.nonzero`), its box
   from the min / max coordinates, and its class by **majority vote** of the semantic mask under
   those pixels (`np.bincount(values).argmax()`). Two touching weeds of the same species stay two
   boxes;
2. **without instances** — for every class separately `cv2.connectedComponentsWithStats` with
   8-connectivity; touching plants of one class merge into one box;
3. objects smaller than 15 pixels are dropped as noise.

`weedsgalore()` creates one `Record` per `*_R.png` file. `Record.obraz` is the path **without**
the channel suffix — the export adds `_B`, `_G`, `_R` itself. The split comes from the authors'
`splits/{train,val,test}.txt`, the group is the flight day (`weedsgalore_2023-05-25`), altitude
5 m, 2.5 mm/px.

### `pascal_voc()` — one XML reader for Sudars, EWIS1 and own data (`adapters.py`)

Reads every `*.xml` recursively. The awkward parts are around the XML, not in it:

- **Finding the image** — next to the XML, in `../raw images/`, in `../images/`, anywhere in the
  dataset, and finally by file name with a different extension (XML says `.jpg`, file is `.png`).
- **Size 0 in the XML** — the size is read from the image (fast read at 1/8 size, `_rozmiar()`).
- **Group** — the parent folder (`train_set`, a session day). A generic folder name such as
  `annotations` says nothing about the session, so then every photo is its own group; otherwise
  the whole dataset would be one group and would fall into one split part.

### `ewis1()`, `_ewis1_uklad()`, `_ewis1_przeksztalc()` — repairing EWIS1 boxes (`adapters.py`)

The hardest adapter. `ewis1()` calls `pascal_voc()`, keeps only files with `Maize` in the name
(young sorghum looks like grass, and grass is a weed in WeedsGalore), sets GSD 1.18 mm/px, takes
field and day from the file name (`Loew_20220512_Maize_010.png` → group `ewis1_Loew_20220512`)
and marks the held-out day as `test`.

Then the boxes have to be repaired. In the EWIS1 XML files **width and height are swapped**, and
for some photos the boxes are written in a rotated frame. Which of four layouts applies is not
written anywhere, so `_ewis1_uklad()` finds out from the authors' masks (which are in the image
frame):

1. The real size `W × H` is read from the image header (Pillow, no decoding).
2. If the XML size equals the real size, the only candidate is `bez_zmian` (unchanged);
   otherwise four candidates are tested:

   | Layout | `[x0, y0, x1, y1]` in the image frame |
   |---|---|
   | `bez_zmian` | `[x0, y0, x1, y1]` |
   | `zamiana_osi` (axes swapped) | `[y0, x0, y1, x1]` |
   | `obrot_prawo` (rotated right) | `[h − y1, x0, h − y0, x1]` |
   | `obrot_lewo` (rotated left) | `[y0, w − x1, y1, w − x0]` |

   (`w, h` = size written in the XML.)
3. The mask is read at 1/4 size. An even sample of up to 60 boxes is transformed with every
   candidate; a box counts as a **hit** when more than 15% of its pixels are plant on the mask.
4. The layout with the most hits wins and is applied to all boxes; a warning is logged when fewer
   than 60% of the sampled boxes hit. Boxes more than 1 px outside the image are dropped.

The log line `ewis1: uklad ramek wzgledem zdjecia: {...}` shows how many photos had which layout.

### `phash()` and `deduplikuj()` — removing duplicates (`steps.py`)

A checksum does not recognise the same photo after recompression or resizing; a perceptual hash
does. `phash()`:

```
grey image → 32 × 32 (INTER_AREA) → 2-D DCT → top-left 8 × 8 (lowest frequencies)
→ bit = 1 where the coefficient > median (DC term left out of the median) → 64-bit string
```

`deduplikuj(rekordy, prog=5)` sorts the records by source priority
(`PRIORYTET = ["wlasne", "weedsgalore", "sudars"]`, everything else last), so when two images
match the one from the more valuable dataset survives. Each new image is compared by **Hamming
distance** (number of differing bits) with already accepted images; ≤ 5 bits = duplicate, written
to `duplikaty.json`. To avoid comparing everything with everything, the hash is cut into
`prog + 1` pieces (6 pieces of 10–11 bits for `prog = 5`) and accepted images are kept in buckets
by piece. Two hashes that differ in at most 5 bits must share at least one identical piece
(pigeonhole principle), so comparing a new image only with images that share a piece still finds
every duplicate.

### `podziel()` and `sprawdz_podzial()` — leak-free split (`steps.py`)

`podziel()` assigns a split **once per group**, deterministically:

```python
x = int.from_bytes(sha256(f"fieldeye-v1:{group}").digest()[:8], "big") / 2**64   # 0 ≤ x < 1
split = "train" if x < 0.70 else "val" if x < 0.85 else "test"
```

The same group always lands in the same part, and adding a new mission never reshuffles the
existing split (a random split would). Records that already have a split from their adapter
(WeedsGalore authors' files, EWIS1 held-out day) are kept and marked `split_oficjalny`. With
`--proporcje 1 0 0` everything without an official split goes to training.

`sprawdz_podzial()` collects, for every group, the set of parts it appears in. A group in more
than one part is reported in one of two lists: **`bledy`** (a group *we* split — a bug, never
expected) or **`oficjalne`** (split by the dataset authors — expected for WeedsGalore, whose split
is stratified by date).

### `przeskaluj()` — matching the drone's ground resolution (`steps.py`)

EWIS1 is shot at 1.18 mm/px, our drone and WeedsGalore at about 2.5 mm/px, so EWIS1 is scaled
by `1.18 / 2.5 ≈ 0.47`: weeds get the same size in pixels as in flight. Only numbers change —
size, boxes, `gsd_mm / f` — and `Record.skala` remembers the factor for the export. Boxes
smaller than 1 px after scaling are dropped. A 5472 × 3648 EWIS1 photo becomes 2572 × 1715 px.

### `kafelkuj()` — cutting tiles and clipping boxes (`steps.py`)

A photo larger than the tile is cut into `896 × 896` tiles. Tile corners:

```python
krok  = int(896 * (1 - 0.1))                               # 806 px step
poz_x = sorted({min(x, W - 896) for x in range(0, W, krok)} | {W - 896})
```

The last tile is pushed back to the edge, so no strip of the photo is lost. For the
2572 × 1715 EWIS1 photo: x ∈ {0, 806, 1612, 1676}, y ∈ {0, 806, 819} → 12 tiles.

Every box is clipped to the tile. It is kept only if **at least 40%** of its original area is
inside (`min_widocznosc=0.4`) — slivers cut off at the edge would teach the model wrong shapes.
A tile `Record` inherits everything from its photo (group, split, scale) plus `kafelek=[x, y]`.
Photos smaller than a tile (WeedsGalore, 600 px) stay whole.

### `kafelkuj_wszystkie()` — empty tiles chosen globally (`steps.py`)

Tiles without weeds are negative examples (bare soil, crop only); without them the model fires on
every irregularity. They are chosen over the **whole dataset**, so that they make up
`udzial_pustych = 15%` of the result:

```python
ile = round(N_with_objects * p / (1 - p))     # 100 tiles with weeds → 18 empty (18 / 118 ≈ 15%)
```

Choosing per photo would give a photo with one weed tile one empty tile — 50% instead of 15%.
The choice uses a seeded generator, so a rebuild picks the same tiles.

### `eksportuj()` and the image loading (`export.py`)

For every record: load the image, resize by `skala`, crop the tile, write `.jpg` (quality 92) and
the label file:

```
<class> <x_center> <y_center> <width> <height>       # fractions of the tile, clipped to 0–1
cx = (x0 + x1) / 2 / W     w = (x1 - x0) / W
```

A label file is written **even when empty** — that is what makes an empty tile a negative example
for Ultralytics. File names are the record id with `/` replaced by `__`
(`ewis1__Loew_20220516_Maize_003_k806_0`); `data.yaml` gets the absolute dataset path.

Loading (`_wczytaj_obraz()` → `_wczytaj_obraz_z_dysku()` → `_wczytaj_bez_skali()`):

- **Cache of the last image.** Tiles of one photo come one after another, so the last decoded
  image is kept; a 40 MB EWIS1 PNG is decoded once instead of twelve times.
- **WeedsGalore colours.** The `_B`, `_G`, `_R` files are stacked into one BGR image. 8-bit
  channels in the `liniowe` mode (final model) are used as they are. 16-bit channels are
  stretched with **one common** range for all three channels (percentiles 0.5 and 99.5) —
  stretching each channel separately changes the colour balance with the image content (soil
  turned pink under dense vegetation). The `srgb` mode adds sRGB gamma on top (tested, not used).
- **Resizing** uses `INTER_AREA` when shrinking (averages pixels, no aliasing on crop rows).

`podglad_ramek()` draws the boxes on 12 random tiles. A half-pixel shift or swapped axes are
invisible in the metrics but ruin training — look at `podglad/` before every training.

---

Tests: `tests/test_normalization.py`, `tests/test_new_datasets.py`, `tests/test_colors.py`.
Next step of the pipeline: [`models/PIPELINE.md`](../models/PIPELINE.md).
