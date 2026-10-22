"""Kolory WeedsGalore: liniowe (jak dotad) i sRGB (z korekta gamma, jak zwykly aparat)."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from normalization import export as E  # noqa: E402
from normalization.manifest import Record  # noqa: E402


@pytest.fixture
def kanaly_wg(tmp_path):
    """Trzy 16-bitowe kanaly jak w WeedsGalore: rampa jasnosci od ciemnego do jasnego."""
    rampa = np.tile(np.linspace(1000, 40000, 256, dtype=np.uint16), (8, 1))
    baza = tmp_path / "2023-05-25_0001"
    for k in "RGB":
        cv2.imwrite(f"{baza}_{k}.png", rampa)
    yield Record(id="x", obraz=str(baza), szer=256, wys=8, zrodlo="weedsgalore")
    E.ustaw_kolory_weedsgalore("liniowe")


def test_liniowe_bez_zmian(kanaly_wg):
    E.ustaw_kolory_weedsgalore("liniowe")
    img = E._wczytaj_bez_skali(kanaly_wg)
    srodek = img[0, 128, 0]
    assert 120 <= srodek <= 135                       # liniowo: polowa zakresu ~ polowa jasnosci


def test_srgb_rozjasnia_ciemne_partie(kanaly_wg):
    E.ustaw_kolory_weedsgalore("liniowe")
    lin = E._wczytaj_bez_skali(kanaly_wg).astype(int)
    E.ustaw_kolory_weedsgalore("srgb")
    srgb = E._wczytaj_bez_skali(kanaly_wg).astype(int)
    assert srgb[0, 128, 0] > lin[0, 128, 0] + 40      # korekta gamma podnosi srodek
    assert srgb[0, 0, 0] <= 5 and srgb[0, -1, 0] >= 250   # konce zakresu bez zmian
    assert (np.diff(srgb[0, :, 0]) >= 0).all()        # dalej rosnaco


def test_srgb_odwrotnosc_color_check():
    sys.path.insert(0, str(ROOT / "scripts"))
    from color_check import srgb_na_liniowe
    x = np.linspace(0, 1, 50)
    y = E._liniowe_na_srgb(x)
    assert np.allclose(srgb_na_liniowe(np.round(y * 255)), x, atol=0.01)


def test_zly_tryb():
    with pytest.raises(ValueError):
        E.ustaw_kolory_weedsgalore("xyz")


def test_opcja_budowy():
    from normalization import build
    zlap = {}

    def falszywe(*a, **k):
        zlap["tryb"] = E.KOLORY_WG["tryb"]
        raise SystemExit(0)
    stare, build.zbuduj = build.zbuduj, falszywe
    try:
        with pytest.raises(SystemExit):
            build.main(["--weedsgalore", "x", "--weedsgalore-kolory", "srgb"])
    finally:
        build.zbuduj = stare
        E.ustaw_kolory_weedsgalore("liniowe")
    assert zlap["tryb"] == "srgb"


def test_skrypty_bez_bledow_skladni():
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / "color_experiment.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_srgb_dziala_tez_dla_kanalow_8_bit(tmp_path):
    rampa = np.tile(np.linspace(10, 250, 256).astype(np.uint8), (8, 1))
    baza = tmp_path / "2023-05-25_0002"
    for k in "RGB":
        cv2.imwrite(f"{baza}_{k}.png", rampa)
    r = Record(id="y", obraz=str(baza), szer=256, wys=8, zrodlo="weedsgalore")
    try:
        E.ustaw_kolory_weedsgalore("liniowe")
        lin = E._wczytaj_bez_skali(r).astype(int)
        assert lin[0, 128, 0] == rampa[0, 128]        # 8 bit liniowo: bez zmian, jak dotad
        E.ustaw_kolory_weedsgalore("srgb")
        assert E._wczytaj_bez_skali(r)[0, 128, 0] > lin[0, 128, 0] + 40
    finally:
        E.ustaw_kolory_weedsgalore("liniowe")
