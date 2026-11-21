"""Pole do symulowanego lotu odtworzone z kafelkow z polozeniem w nazwie (EWIS1)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mission.sim_flight import _scal_powtorzone, pole_ze_zdjecia, zdjecia_z_kafelkow  # noqa: E402


def _zbior(tmp_path, ramka):
    """Zdjecie 700 x 500 pociete na 4 kafelki 400 x 300 z zakladka, jak przy normalizacji."""
    rng = np.random.default_rng(0)
    zdj = rng.integers(0, 255, (500, 700, 3), dtype=np.uint8)
    im, lb = tmp_path / "images" / "test", tmp_path / "labels" / "test"
    im.mkdir(parents=True); lb.mkdir(parents=True)
    for x, y in [(0, 0), (300, 0), (0, 200), (300, 200)]:
        k = zdj[y:y + 300, x:x + 400]
        nazwa = f"ewis1__Loew_20220516_Maize_006_k{x}_{y}"
        cv2.imwrite(str(im / f"{nazwa}.png"), k)
        x0, y0, x1, y1 = ramka                                   # ramka na zdjeciu, przycieta do kafelka
        cx0, cy0, cx1, cy1 = max(x0, x), max(y0, y), min(x1, x + 400), min(y1, y + 300)
        if cx1 > cx0 and cy1 > cy0:
            cx, cy, w, h = ((cx0 + cx1) / 2 - x) / 400, ((cy0 + cy1) / 2 - y) / 300, (cx1 - cx0) / 400, (cy1 - cy0) / 300
            (lb / f"{nazwa}.txt").write_text(f"0 {cx} {cy} {w} {h}\n")
    cv2.imwrite(str(im / "weedsgalore__2023-05-25__0001.png"), zdj[:100, :100])   # bez polozenia - pomijany
    return zdj, im


def test_odtworzenie_zdjecia_co_do_piksela(tmp_path):
    zdj, im = _zbior(tmp_path, (320, 220, 360, 250))
    z = zdjecia_z_kafelkow(im.glob("*"))
    assert list(z) == ["ewis1__Loew_20220516_Maize_006"] and len(z["ewis1__Loew_20220516_Maize_006"]) == 4
    pole = pole_ze_zdjecia(z["ewis1__Loew_20220516_Maize_006"])
    assert pole.shape == zdj.shape and np.array_equal(pole, zdj)


def test_chwast_z_zakladki_to_jedna_ramka(tmp_path):
    _, im = _zbior(tmp_path, (320, 220, 360, 250))                 # lezy w zakladce wszystkich 4 kafelkow
    pole, ramki = pole_ze_zdjecia(zdjecia_z_kafelkow(im.glob("*"))["ewis1__Loew_20220516_Maize_006"], True)
    assert len(ramki) == 1
    assert np.allclose(ramki[0], (320, 220, 360, 250), atol=1)


def test_przyciety_chwast_zostaje_w_calosci(tmp_path):
    _, im = _zbior(tmp_path, (380, 100, 420, 140))                 # przyciety w kafelku k0_0, caly w k300_0
    _, ramki = pole_ze_zdjecia(zdjecia_z_kafelkow(im.glob("*"))["ewis1__Loew_20220516_Maize_006"], True)
    assert len(ramki) == 1 and np.allclose(ramki[0], (380, 100, 420, 140), atol=1)


def test_rozne_chwasty_zostaja_osobno():
    r = _scal_powtorzone([(0, 0, 10, 10), (20, 20, 30, 30), (1, 1, 9, 9)])
    assert len(r) == 2
