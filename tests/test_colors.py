"""Testy kolorow WeedsGalore: liniowe (jak dotad) i sRGB (z korekta gamma, jak zwykly aparat).

Po co: WeedsGalore ma osobne 16-bitowe kanaly R, G, B. normalization/export.py sklada je
w jedno zdjecie 8-bit - liniowo (tryb modelu koncowego) albo z korekta gamma sRGB.
Testy pilnuja, zeby tryb liniowy nie zmienial jasnosci, a sRGB rozjasnial ciemne partie
bez psucia koncow zakresu.

Testowane: normalization/export.py (ustaw_kolory_weedsgalore, _wczytaj_bez_skali, KOLORY_WG)
           i opcja --weedsgalore-kolory w normalization/build.py.

    python -m pytest tests/test_colors.py -q
"""
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
    """Trzy 16-bitowe kanaly jak w WeedsGalore: rampa jasnosci od ciemnego do jasnego.

    Po co: sztuczne zdjecie, na ktorym latwo sprawdzic jasnosc w srodku i na koncach.
           Po tescie przywraca tryb "liniowe", zeby nie psuc innych testow.

    Argumenty:
        tmp_path - fixture pytest: pusty folder tymczasowy

    Zwraca:
        Record - rekord zdjecia 256x8, np. Record(id="x", obraz=".../2023-05-25_0001", ...)

    Uzywana w:
        test_liniowe_bez_zmian(), test_srgb_rozjasnia_ciemne_partie()
    """
    rampa = np.tile(np.linspace(1000, 40000, 256, dtype=np.uint16), (8, 1))
    baza = tmp_path / "2023-05-25_0001"
    for k in "RGB":
        cv2.imwrite(f"{baza}_{k}.png", rampa)
    yield Record(id="x", obraz=str(baza), szer=256, wys=8, zrodlo="weedsgalore")
    E.ustaw_kolory_weedsgalore("liniowe")


def test_liniowe_bez_zmian(kanaly_wg):
    """Tryb liniowy: polowa zakresu 16-bit daje mniej wiecej polowe jasnosci 8-bit.

    Po co: model koncowy byl uczony na kolorach liniowych - ten tryb nie moze sie zmienic.

    Argumenty:
        kanaly_wg - fixture: rekord z rampa jasnosci

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    E.ustaw_kolory_weedsgalore("liniowe")
    img = E._wczytaj_bez_skali(kanaly_wg)
    srodek = img[0, 128, 0]
    assert 120 <= srodek <= 135                       # liniowo: polowa zakresu ~ polowa jasnosci


def test_srgb_rozjasnia_ciemne_partie(kanaly_wg):
    """Tryb sRGB: srodek jasniejszy niz liniowo, konce zakresu bez zmian, dalej rosnaco.

    Po co: korekta gamma ma dzialac jak w zwyklym aparacie, a nie przesuwac cala skale.

    Argumenty:
        kanaly_wg - fixture: rekord z rampa jasnosci

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    E.ustaw_kolory_weedsgalore("liniowe")
    lin = E._wczytaj_bez_skali(kanaly_wg).astype(int)
    E.ustaw_kolory_weedsgalore("srgb")
    srgb = E._wczytaj_bez_skali(kanaly_wg).astype(int)
    assert srgb[0, 128, 0] > lin[0, 128, 0] + 40      # korekta gamma podnosi srodek
    assert srgb[0, 0, 0] <= 5 and srgb[0, -1, 0] >= 250   # konce zakresu bez zmian
    assert (np.diff(srgb[0, :, 0]) >= 0).all()        # dalej rosnaco


def test_zly_tryb():
    """Nieznany tryb kolorow konczy sie bledem ValueError.

    Po co: literowka w opcji ma zatrzymac budowe zbioru, a nie cicho dac zle kolory.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza, ze poleci wyjatek

    Uzywana w:
        pytest (python -m pytest tests)
    """
    with pytest.raises(ValueError):
        E.ustaw_kolory_weedsgalore("xyz")


def test_opcja_budowy():
    """Opcja --weedsgalore-kolory srgb w build.py ustawia tryb przed budowa zbioru.

    Po co: sprawdza samo przekazanie opcji, bez budowania - zbuduj() jest podmienione
           na atrape, ktora zapamietuje tryb i konczy program.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
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


def test_srgb_dziala_tez_dla_kanalow_8_bit(tmp_path):
    """Kanaly 8-bit: liniowo bez zmian, sRGB rozjasnia srodek.

    Po co: czesc zdjec moze miec kanaly 8-bit zamiast 16-bit - oba tryby maja dzialac i wtedy.

    Argumenty:
        tmp_path - fixture pytest: pusty folder tymczasowy

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
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
