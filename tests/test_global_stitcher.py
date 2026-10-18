"""Testy metody 'global' - na kadrach wycietych z jednego obrazu, ze znana prawda."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stitchlive.global_stitcher import GlobalStitcher        # noqa: E402
from stitchlive.flight_sim import SimSettings, rozloz, margines  # noqa: E402
from stitchlive.field import zbuduj_pole                 # noqa: E402
from stitchlive.evaluation import ocen                       # noqa: E402


def _zloz(katalog, **kw):
    s = GlobalStitcher(work_megapix=0.3, **kw)
    for p in sorted(Path(katalog).glob("*.jpg")):
        s.dodaj(p)
    s.zakoncz()
    return s


@pytest.fixture(scope="module")
def pole():
    return zbuduj_pole(1600, 1100, seed=3)


@pytest.fixture(scope="module")
def obrocone(pole, tmp_path_factory):
    """Trasa z losowym obrotem kadrow - przypadek, ktory psuje cv2.detail."""
    out = tmp_path_factory.mktemp("obrot")
    prawda = rozloz(pole, out, SimSettings(tryb="trasa", kadr=400, obrot=8, seed=5))
    return out, prawda


def test_sklada_kadry_obrocone(obrocone, pole):
    out, prawda = obrocone
    s = _zloz(out / "images")
    assert s.mozaika is not None
    assert len(s.S) == len(prawda["kadry"])          # wszystkie kadry w mozaice
    w = ocen(s.mozaika, pole, prawda)
    assert w["ok"] and w["psnr_db"] > 24


def test_pary_sa_dokladne(obrocone):
    out, _ = obrocone
    s = _zloz(out / "images")
    assert s.blad_px < 2.0                            # mediana bledu par w pikselach


def test_kadr_odniesienia_to_tozsamosc(obrocone):
    out, _ = obrocone
    s = _zloz(out / "images")
    np.testing.assert_allclose(s.S[0], [1, 0, 0, 0], atol=1e-3)


def test_losowe_ciecie_z_zakloceniami(pole, tmp_path):
    u = SimSettings(tryb="losowy", kadr=400, liczba=15, obrot=5, skala=0.05,
                   rozmycie=0.3, szum=3, seed=11)
    prawda = rozloz(pole, tmp_path, u)
    s = _zloz(tmp_path / "images")
    assert len(s.S) >= 13                            # dopuszczamy zgubienie pojedynczych
    w = ocen(s.mozaika, pole, prawda)
    assert w["ok"] and w["psnr_db"] > 22


def test_rozmyty_kadr_nie_odpada(tmp_path, pole):
    """Zapasowy detektor ma uratowac kadr, na ktorym domyslny prog FAST nic nie widzi."""
    u = SimSettings(tryb="trasa", kadr=400, rozmycie=1.0, seed=2)   # kazdy kadr rozmyty
    rozloz(pole, tmp_path, u)
    s = _zloz(tmp_path / "images")
    assert len(s.obrazy) == len(list((tmp_path / "images").glob("*.jpg")))


def test_zakoncz_renderuje_gdy_licznik_nie_doszedl(obrocone):
    out, _ = obrocone
    s = GlobalStitcher(work_megapix=0.3, renderuj_co=10_000)
    for p in sorted((out / "images").glob("*.jpg")):
        s.dodaj(p)
    assert s.mozaika is None                         # przy tak duzym progu - brak renderu
    assert s.zakoncz() and s.mozaika is not None


def test_zakres_par_ogranicza_dopasowania(obrocone):
    out, _ = obrocone
    pelny = _zloz(out / "images")
    ogr = _zloz(out / "images", zakres_par=3)
    assert all(abs(i - j) <= 3 for i, j in ogr.pary)
    assert len(ogr.pary) < len(pelny.pary)


def test_niepowiazany_kadr_nie_psuje_mozaiki(obrocone, tmp_path):
    out, _ = obrocone
    kat = tmp_path / "img"
    kat.mkdir()
    for p in sorted((out / "images").glob("*.jpg")):
        (kat / p.name).write_bytes(p.read_bytes())
    # obcy kadr - szum, bez zwiazku z polem
    obcy = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    cv2.imwrite(str(kat / "IMG_9999.jpg"), obcy)
    s = _zloz(kat)
    assert s.mozaika is not None
    obcy_idx = len(s.obrazy) - 1                     # obcy kadr jest ostatni
    assert obcy_idx not in s.S                       # nie trafil do mozaiki


def test_margines_nie_jest_przesadny():
    u = SimSettings(obrot=2.0, skala=0.02)
    # przy malym obrocie i skali margines ma byc bliski polowie kadru,
    # a nie teoretycznego maksimum (sqrt(2) * 1.33)
    assert margines(640, u) < 640 / 2 * 1.25


def test_na_mozaike_trafia_w_te_same_miejsca(tmp_path):
    """Punkt z kadru przeliczony na mozaike ma tam ten sam kolor co w kadrze.

    Uzywane w GUI do nanoszenia wykrytych chwastow na mozaike.
    """
    import cv2
    import numpy as np
    from stitchlive.global_stitcher import GlobalStitcher
    from stitchlive.field import zbuduj_pole
    from stitchlive.flight_sim import SimSettings, rozloz

    zrodlo = zbuduj_pole(1600, 1100, seed=3)
    prawda = rozloz(zrodlo, tmp_path, SimSettings(kadr=500, overlap=0.7, sidelap=0.5, obrot=2.0, seed=3))
    pliki = [tmp_path / "images" / k["plik"] for k in prawda["kadry"]]
    s = GlobalStitcher(renderuj_co=10 ** 6)
    for p in pliki:
        s.dodaj(p)
    s.zakoncz()
    moz = cv2.GaussianBlur(s.mozaika, (9, 9), 0).astype(float)

    rng = np.random.default_rng(0)
    trafione, przesuniete = [], []
    for p in pliki[::3]:
        kadr = cv2.GaussianBlur(cv2.imread(str(p)), (9, 9), 0).astype(float)
        h, w = kadr.shape[:2]
        pkt = rng.uniform([60, 60], [w - 60, h - 60], size=(40, 2))
        for nazwa, przes in (("ok", 0), ("zle", 80)):
            q = s.na_mozaike(p, pkt + przes, (h, w))
            if q is None:
                continue
            for (u, v), (x, y) in zip(pkt, q):
                x, y = int(round(x)), int(round(y))
                if 0 <= y < moz.shape[0] and 0 <= x < moz.shape[1] and moz[y, x].sum() > 0:
                    d = np.abs(kadr[int(v), int(u)] - moz[y, x]).mean()
                    (trafione if nazwa == "ok" else przesuniete).append(d)
    assert len(trafione) > 50
    # w dobrym miejscu kolory prawie identyczne, a 80 px obok - wyraznie rozne
    assert np.median(trafione) < 12
    assert np.median(trafione) * 2 < np.median(przesuniete)


def test_na_mozaike_nieznany_kadr():
    from stitchlive.global_stitcher import GlobalStitcher
    assert GlobalStitcher().na_mozaike("nie_ma.jpg", [[0, 0]], (100, 100)) is None


def test_powtorzony_teren_nie_sciska_mozaiki(tmp_path):
    """Ten sam fragment pola w dwoch miejscach (np. kafelki z roznych dni lotu).

    Wczesniej sprzeczne pary "godzily sie" przez sciesnienie kadrow prawie do
    zera: mozaika zapadala sie do rozmiaru jednego kadru, a wykrycia ze
    wszystkich kadrow ladowaly w jednym miejscu.
    """
    import numpy as np
    from stitchlive.global_stitcher import GlobalStitcher
    from stitchlive.evaluation import ocen
    from stitchlive.field import zbuduj_pole
    from stitchlive.flight_sim import SimSettings, rozloz

    duze = zbuduj_pole(1800, 1800, seed=11)
    kaf = [duze[r * 600:(r + 1) * 600, c * 600:(c + 1) * 600].copy() for r in range(3) for c in range(3)]
    kaf[8], kaf[6] = kaf[0].copy(), kaf[2].copy()
    zrodlo = np.vstack([np.hstack(kaf[r * 3:(r + 1) * 3]) for r in range(3)])
    prawda = rozloz(zrodlo, tmp_path, SimSettings(tryb="trasa", kadr=700, overlap=0.7, sidelap=0.5,
                                                  obrot=2.0, ekspozycja=0.03, seed=7))
    s = GlobalStitcher(renderuj_co=10 ** 6)
    for k in prawda["kadry"]:
        s.dodaj(tmp_path / "images" / k["plik"])
    s.zakoncz()
    skale = [np.hypot(v[0], v[1]) for v in s.S.values()]
    assert min(skale) > 0.9 and max(skale) < 1.1
    assert s.mozaika.shape[0] > 1200 and s.mozaika.shape[1] > 1200
    assert ocen(s.mozaika, zrodlo, prawda)["psnr_db"] > 24
