"""Czy model uczony na WeedsGalore zawodzi na zwyklych zdjeciach RGB przez kolory?

WeedsGalore to surowe pomiary kamery multispektralnej (liniowe, 16 bit), ktore
normalizacja rozciaga liniowo do 8 bitow. Zwykly aparat (EWIS1, Arducam) zapisuje
zdjecie z korekta gamma i balansem bieli. Skrypt robi z kafelkow testowych EWIS1
"zdjecia w stylu WeedsGalore": zdejmuje gamme (sRGB -> liniowo) i rozciaga
percentylami 0,5-99,5 wspolnie dla trzech kanalow, tak samo jak normalizacja.

    python scripts/color_check.py zbior_ewis1_podzial/dataset -o zbior_ewis1_jak_wg
    python -m models.cli ocen-detektor modele_sezon_nadprob/model.pt zbior_ewis1_jak_wg/data.yaml --split test --prog 0.15

Jesli AP50 wyraznie wzrosnie wzgledem oryginalnych kafelkow, roznica miedzy
polami wynika glownie z przetwarzania kolorow, a nie z samych roslin.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


def srgb_na_liniowe(img):
    """Odwrotnosc korekty gamma sRGB (wartosci 0-1)."""
    x = img.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def jak_weedsgalore(img):
    """Liniowe jasnosci + wspolne rozciagniecie percentylami, jak w normalization/export.py."""
    f = srgb_na_liniowe(img)
    lo, hi = np.percentile(f, 0.5), np.percentile(f, 99.5)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zbior", help="katalog zbioru z images/test i labels/test")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args(argv)
    src, out = Path(a.zbior), Path(a.output)
    if out.exists():
        shutil.rmtree(out)
    (out / "images" / a.split).mkdir(parents=True)
    shutil.copytree(src / "labels" / a.split, out / "labels" / a.split)
    n = 0
    for p in sorted((src / "images" / a.split).iterdir()):
        img = cv2.imread(str(p))
        if img is None:
            continue
        cv2.imwrite(str(out / "images" / a.split / p.name), jak_weedsgalore(img))
        n += 1
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out.resolve()), "train": f"images/{a.split}", "val": f"images/{a.split}",
         a.split: f"images/{a.split}", "names": {0: "chwast"}}, allow_unicode=True))
    print(f"  {n} kafelkow w stylu WeedsGalore -> {out}/data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())