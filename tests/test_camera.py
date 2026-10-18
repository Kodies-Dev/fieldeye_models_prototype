"""Model kamery drona i symulacja lotu z kadrem prostokatnym (kamera 4:3)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stitchlive.camera import (ARDUCAM_64MP_16MP, ARDUCAM_64MP_64MP, opis_lotu,  # noqa: E402
                               odstep_zdjec_s, rozmycie_px)
from stitchlive.flight_sim import (SimSettings, _rozmyj_ruchem, macierz_kadru,  # noqa: E402
                                   marginesy, naroza_kadru, rozloz, wytnij)


# ---------------------------------------------------------------- kamera

def test_arducam_z_5_m():
    """Liczby, na ktorych opiera sie symulacja lotu FIELDEYE."""
    k = ARDUCAM_64MP_16MP
    w, h = k.zasieg_m(5.0)
    assert w == pytest.approx(7.25, abs=0.01) and h == pytest.approx(5.44, abs=0.01)
    assert k.gsd_mm(5.0) == pytest.approx(1.57, abs=0.01)
    assert ARDUCAM_64MP_64MP.gsd_mm(5.0) == pytest.approx(0.79, abs=0.01)
    poz, pion = k.kat_widzenia_st()
    assert 70 < poz < 75 and 54 < pion < 58                  # producent: 72 x 54,6 st.
    assert k.kadr_w_zrodle(5.0, 2.5) == (2902, 2176)
    assert k.kafelek_px(2.5) == 600                          # kafelek treningowy WeedsGalore


def test_rozmycie_i_odstep_zdjec():
    assert rozmycie_px(3.0, 1 / 1000, 2.5) == pytest.approx(1.2)
    assert rozmycie_px(10.0, 1 / 1000, 2.5) == pytest.approx(4.0)
    assert odstep_zdjec_s(5.44, 0.7, 3.0) == pytest.approx(0.544, abs=0.001)


def test_opis_lotu():
    o = opis_lotu(ARDUCAM_64MP_16MP, 5.0, 2.5, 3.0, 1 / 1000, 0.7)
    assert o["kadr_px"] == (2902, 2176) and o["kafelek_px"] == 600
    assert o["kafelek_kamery_px"] == 956                     # na dronie: kafelek zdjecia 16 MP
    assert 20 <= o["kafelkow_na_kadr"] <= 40


# ---------------------------------------------------------------- kadr prostokatny

@pytest.fixture
def pole():
    rng = np.random.default_rng(3)
    return cv2.GaussianBlur(rng.integers(0, 255, (900, 1400, 3), dtype=np.uint8), (7, 7), 0)


def test_prostokatny_kadr_to_dokladny_wycinek(pole):
    M = macierz_kadru(700, 450, 0, 400, 400, 300)
    kadr = wytnij(pole, M, 400, 300)
    assert kadr.shape == (300, 400, 3)
    wzor = pole[300:600, 500:900]
    assert np.abs(kadr.astype(int) - wzor.astype(int)).mean() < 2.0


def test_naroza_prostokata_i_srodek_po_obrocie():
    M = macierz_kadru(700, 450, 30, 400, 400, 300)
    nar = np.array(naroza_kadru(M, 400, 300))
    assert np.allclose(nar.mean(axis=0), (700, 450))         # obrot wokol srodka
    assert np.linalg.norm(nar[1] - nar[0]) == pytest.approx(400)
    assert np.linalg.norm(nar[3] - nar[0]) == pytest.approx(300)


def test_marginesy_prostokata():
    mx, my = marginesy(400, SimSettings(kadr=400, kadr_wys=300))
    assert mx == pytest.approx(202) and my == pytest.approx(152)
    mx_o, my_o = marginesy(400, SimSettings(kadr=400, kadr_wys=300, obrot=5))
    assert mx_o > mx and my_o > my


def test_lot_z_kadrem_4_3(pole, tmp_path):
    u = SimSettings(kadr=400, kadr_wys=300, overlap=0.7, sidelap=0.5, jitter=0, ekspozycja=0)
    pr = rozloz(pole, tmp_path, u)
    k = pr["kadry"]
    assert all(r["kadr_px"] == [400, 300] for r in k)
    assert cv2.imread(str(tmp_path / "images" / k[0]["plik"])).shape == (300, 400, 3)
    # odstep miedzy liniami z wysokosci kadru: 300 * (1 - 0,5) = 150 px
    ys = sorted({round(r["srodek"][1]) for r in k})
    assert np.diff(ys).min() == pytest.approx(150, abs=1)
    # krok wzdluz linii z szerokosci: 400 * (1 - 0,7) = 120 px
    pierwsza = sorted(r["srodek"][0] for r in k if r["linia"] == 0)
    assert np.diff(pierwsza).max() == pytest.approx(120, abs=1)


def test_kwadratowy_kadr_bez_zmian(pole, tmp_path):
    """Stare ustawienia (bez kadr_wys) daja to samo co przed zmiana."""
    pr = rozloz(pole, tmp_path, SimSettings(kadr=400, jitter=0))
    assert all(r["kadr_px"] == [400, 400] for r in pr["kadry"])


def test_rozmycie_ruchem_wzdluz_lotu():
    img = np.zeros((41, 41, 3), np.uint8)
    img[:, 20] = 255                                         # pionowa linia: lot w poziomie ja rozmywa
    r = _rozmyj_ruchem(img, 4.0)
    assert r[20, 20, 0] < 120 and r[20, 18, 0] > 20          # rozlana w poziomie
    img2 = np.zeros((41, 41, 3), np.uint8)
    img2[20, :] = 255                                        # pozioma linia: zostaje ostra
    assert _rozmyj_ruchem(img2, 4.0)[20, 20, 0] == 255


def test_rozmycie_w_symulacji(pole, tmp_path):
    ostry = rozloz(pole, tmp_path / "a", SimSettings(kadr=300, jitter=0, ekspozycja=0))
    rozm = rozloz(pole, tmp_path / "b", SimSettings(kadr=300, jitter=0, ekspozycja=0, rozmycie_ruch_px=5))
    a = cv2.imread(str(tmp_path / "a" / "images" / ostry["kadry"][0]["plik"]), 0)
    b = cv2.imread(str(tmp_path / "b" / "images" / rozm["kadry"][0]["plik"]), 0)
    ostrosc = lambda x: cv2.Laplacian(x, cv2.CV_64F).var()
    assert ostrosc(b) < 0.6 * ostrosc(a)


def test_prawda_w_prostokatnym_kadrze():
    """Ramka w srodku kadru 4:3 trafia w srodek kadru - takze po przekazaniu kadr_px."""
    pytest.importorskip("streamlit")
    from app import prawda_w_kadrze
    info = {"srodek": [700, 450], "kat_st": 0.0, "zasieg_px": 400, "kadr_px": [400, 300]}
    r = prawda_w_kadrze([(690, 440, 710, 460)], info, 400)
    assert len(r) == 1
    x0, y0, x1, y1 = r[0]
    assert (x0, y0, x1, y1) == pytest.approx((190, 140, 210, 160), abs=0.01)