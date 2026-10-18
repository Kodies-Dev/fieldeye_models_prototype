"""Testy normalizacji: adaptery, taksonomia, deduplikacja, podzial, kafelkowanie, eksport.

Najwazniejsze testy to te o przeciekach (podzial po grupach) i o poprawnosci
przeliczania ramek - blad w nich jest niewidoczny w metrykach, a psuje trening.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from normalization import steps as K  # noqa: E402
from normalization import manifest as M  # noqa: E402
from normalization.adapters import ADAPTERY                # noqa: E402
from normalization.build import zbuduj                     # noqa: E402
from normalization.export import eksportuj                # noqa: E402
from normalization.examples import utworz                 # noqa: E402
from normalization.taxonomy import Taxonomy            # noqa: E402


@pytest.fixture(scope="module")
def dane(tmp_path_factory):
    return utworz(tmp_path_factory.mktemp("zrodla"))


# ---------------------------------------------------------------- taksonomia

def test_aliasy_prowadza_do_jednego_gatunku():
    """Zoltlica w trzech zbiorach zapisana roznie - ma byc jedna klasa."""
    t = Taxonomy()
    klucze = {t.klucz(n) for n in ("GP", "Galinsoga parviflora", "galinsoga_parviflora")}
    assert klucze == {"galinsoga_parviflora"}


def test_kod_eppo_z_faza_wzrostu():
    """ZEAMX_V1 i ZEAMX_V3 to ta sama kukurydza, rozne fazy."""
    t = Taxonomy()
    assert t.klucz("ZEAMX_V1") == t.klucz("ZEAMX_V3") == t.klucz("ZEAMX")
    assert t.faza("ZEAMX_V1") == "V1"
    assert t.faza("ZEAMX") is None


def test_nieznana_nazwa_trafia_do_raportu():
    t = Taxonomy()
    assert t.klucz("cos_czego_nie_ma") is None
    assert ("cos_czego_nie_ma", 1) in t.raport_nieznanych()


def test_tryb_binarnie_zwija_gatunki_do_rol():
    t = Taxonomy()
    assert t.etykieta("galinsoga_parviflora", "binarnie") == "chwast"
    assert t.etykieta("zea_mays", "binarnie") == "uprawa"


def test_tryb_chwasty_pomija_uprawy():
    t = Taxonomy()
    assert t.etykieta("galinsoga_parviflora", "chwasty") == "chwast"
    assert t.etykieta("zea_mays", "chwasty") is None


def test_gatunki_rzadkie_zwijane_do_klasy_zbiorczej():
    """Sorgo aleppo jest w Polsce efemerofitem - nie ma sensu osobnej klasy."""
    t = Taxonomy()
    assert t.etykieta("sorghum_halepense", "gatunki") != "sorghum_halepense"


# ---------------------------------------------------------------- adaptery

def test_wszystkie_adaptery_cos_wczytuja(dane):
    t = Taxonomy()
    for nazwa, sciezka in dane.items():
        kw = {"splity": None} if nazwa == "weedsgalore" else {}
        rek = ADAPTERY[nazwa](sciezka, t, **kw)
        assert rek, nazwa
        assert all(r.szer > 0 and r.wys > 0 for r in rek), nazwa


def test_weedsgalore_czyta_kanaly_i_maski(dane):
    t = Taxonomy()
    rek = ADAPTERY["weedsgalore"](dane["weedsgalore"], t, splity=None)
    assert all(Path(r.obraz + "_R.png").exists() for r in rek)
    assert any(r.obiekty for r in rek)
    assert all(o.klasa != "nieznana" for r in rek for o in r.obiekty)


def test_weedsgalore_grupa_to_data_lotu(dane):
    t = Taxonomy()
    rek = ADAPTERY["weedsgalore"](dane["weedsgalore"], t, splity=None)
    assert all(r.grupa.startswith("weedsgalore_2023-") for r in rek)


def test_wlasne_czyta_nazwy_klas_z_classes_txt(dane):
    """W formacie YOLO klasa to indeks - nazwy sa w classes.txt obok."""
    t = Taxonomy()
    rek = ADAPTERY["wlasne"](dane["wlasne"], t)
    assert any(r.obiekty for r in rek)
    # bez wczytania classes.txt wszystko trafiloby do raportu jako "0", "1"
    assert not [n for n, _ in t.raport_nieznanych() if n.isdigit()]


def test_ramki_miesza_sie_w_obrazie(dane):
    t = Taxonomy()
    for nazwa, sciezka in dane.items():
        kw = {"splity": None} if nazwa == "weedsgalore" else {}
        for r in ADAPTERY[nazwa](sciezka, t, **kw):
            for o in r.obiekty:
                x0, y0, x1, y1 = o.xyxy
                assert 0 <= x0 < x1 <= r.szer and 0 <= y0 < y1 <= r.wys, (nazwa, r.id)


# ---------------------------------------------------------------- deduplikacja

def test_deduplikacja_wykrywa_ten_sam_obraz(dane, tmp_path):
    """Imitacje zawieraja celowy duplikat miedzy zbiorami."""
    t = Taxonomy()
    rek = []
    for nazwa, sciezka in dane.items():
        kw = {"splity": None} if nazwa == "weedsgalore" else {}
        rek += ADAPTERY[nazwa](sciezka, t, **kw)
    zostaly, odrzucone = K.deduplikuj(rek, prog=5)
    assert len(odrzucone) >= 1
    assert len(zostaly) == len(rek) - len(odrzucone)


def test_deduplikacja_nie_usuwa_roznych_zdjec(tmp_path):
    from normalization.manifest import Record
    rng = np.random.default_rng(0)
    rek = []
    for i in range(4):
        p = tmp_path / f"{i}.png"
        cv2.imwrite(str(p), rng.integers(0, 255, (80, 80, 3), dtype=np.uint8))
        rek.append(Record(id=str(i), obraz=str(p), szer=80, wys=80, zrodlo="x"))
    _, odrzucone = K.deduplikuj(rek, prog=5)
    assert odrzucone == []


# ---------------------------------------------------------------- podzial

def _rek(grupa, split=None, zrodlo="x"):
    from normalization.manifest import Record
    return Record(id=f"{grupa}/{id(object())}", obraz="a", szer=10, wys=10,
                  zrodlo=zrodlo, grupa=grupa, split=split)


def test_podzial_trzyma_grupe_w_jednej_czesci():
    rek = [_rek(f"misja_{i}") for i in range(40) for _ in range(3)]
    K.podziel(rek)
    assert K.sprawdz_podzial(rek)["bledy"] == {}


def test_podzial_jest_powtarzalny():
    a = [_rek(f"m_{i}") for i in range(30)]
    b = [_rek(f"m_{i}") for i in range(30)]
    K.podziel(a)
    K.podziel(b)
    assert [r.split for r in a] == [r.split for r in b]


def test_podzial_nie_przetasowuje_po_dodaniu_grup():
    """Dodanie nowej misji nie moze zmienic przydzialu istniejacych."""
    a = [_rek(f"m_{i}") for i in range(20)]
    K.podziel(a)
    przed = {r.grupa: r.split for r in a}
    b = [_rek(f"m_{i}") for i in range(30)]       # 20 starych + 10 nowych
    K.podziel(b)
    po = {r.grupa: r.split for r in b}
    assert all(po[g] == s for g, s in przed.items())


def test_oficjalny_podzial_zachowywany_i_oznaczony():
    rek = [_rek("wg_2023-06-15", "train"), _rek("wg_2023-06-15", "test")]
    K.podziel(rek, zachowaj_istniejace=True)
    assert [r.split for r in rek] == ["train", "test"]
    kon = K.sprawdz_podzial(rek)
    assert kon["bledy"] == {}                     # to nie blad kodu...
    assert "wg_2023-06-15" in kon["oficjalne"]    # ...tylko wlasnosc zbioru


def test_wlasny_podzial_ignoruje_oficjalny():
    rek = [_rek("wg_2023-06-15", "train"), _rek("wg_2023-06-15", "test")]
    K.podziel(rek, zachowaj_istniejace=False)
    assert len({r.split for r in rek}) == 1       # cala grupa w jednej czesci
    assert K.sprawdz_podzial(rek)["bledy"] == {}


def test_tylko_wlasne_do_testu():
    rek = [_rek(f"m_{i}", zrodlo="weedsgalore") for i in range(30)]
    rek += [_rek(f"w_{i}", zrodlo="wlasne") for i in range(30)]
    K.podziel(rek, tylko_wlasne_do_testu=True)
    assert all(r.zrodlo == "wlasne" for r in rek if r.split == "test")


# ---------------------------------------------------------------- kafelkowanie

def _rek_z_obiektem(xyxy, szer=1000, wys=1000):
    from normalization.manifest import LabeledObject, Record
    return Record(id="t", obraz="a", szer=szer, wys=wys, zrodlo="x", grupa="g",
                  obiekty=[LabeledObject(klasa="chwast", zrodlowa_nazwa="chwast", xyxy=list(xyxy))])


def test_kafelkowanie_przelicza_wspolrzedne():
    r = _rek_z_obiektem((520, 520, 560, 560))
    kafelki = K.kafelkuj(r, kafelek=400, zakladka=0.0)
    maja = [k for k in kafelki if k.obiekty]
    assert maja
    for k in maja:
        x, y = k.kafelek
        o = k.obiekty[0]
        # ramka w kafelku + przesuniecie kafelka = ramka w oryginale
        assert o.xyxy[0] + x == pytest.approx(520, abs=1)
        assert o.xyxy[1] + y == pytest.approx(520, abs=1)


def test_kafelkowanie_odrzuca_szczatkowe_ramki():
    """Ramka przycieta na krawedzi kafelka uczy model blednych wzorcow."""
    r = _rek_z_obiektem((380, 380, 420, 420))     # na granicy kafelkow 400
    kafelki = K.kafelkuj(r, kafelek=400, zakladka=0.0, min_widocznosc=0.9)
    for k in kafelki:
        for o in k.obiekty:
            w = o.xyxy[2] - o.xyxy[0]
            h = o.xyxy[3] - o.xyxy[1]
            assert w * h >= 0.9 * 40 * 40 * 0.99


def test_kafelki_dziedzicza_grupe_i_split():
    r = _rek_z_obiektem((100, 100, 140, 140))
    r.split = "val"
    for k in K.kafelkuj(r, kafelek=400, zakladka=0.0):
        assert k.grupa == "g" and k.split == "val"


def test_udzial_pustych_kafelkow_ograniczony():
    rekordy = [_rek_z_obiektem((50, 50, 90, 90), 2000, 2000) for _ in range(3)]
    for r in rekordy:
        r.split = "train"
    wynik = K.kafelkuj_wszystkie(rekordy, kafelek=400, zakladka=0.0,
                                 udzial_pustych=0.2, seed=0)
    puste = sum(1 for k in wynik if not k.obiekty)
    assert puste / len(wynik) == pytest.approx(0.2, abs=0.08)   # udzial globalny
    assert puste > 0                              # ale jakies zostaly


# ---------------------------------------------------------------- eksport

def test_eksport_format_yolo(tmp_path):
    r = _rek_z_obiektem((100, 200, 300, 400), szer=1000, wys=1000)
    r.split = "train"
    r.obraz = str(tmp_path / "x.png")
    cv2.imwrite(r.obraz, np.zeros((1000, 1000, 3), np.uint8))
    eksportuj([r], tmp_path / "ds", ["chwast"], kopiuj_obrazy=True)
    txt = next((tmp_path / "ds" / "labels" / "train").glob("*.txt"))
    idx, cx, cy, w, h = txt.read_text().split()
    assert idx == "0"
    assert float(cx) == pytest.approx(0.2) and float(cy) == pytest.approx(0.3)
    assert float(w) == pytest.approx(0.2) and float(h) == pytest.approx(0.2)


def test_eksport_data_yaml(tmp_path):
    r = _rek_z_obiektem((10, 10, 20, 20))
    r.split = "train"
    r.obraz = str(tmp_path / "x.png")
    cv2.imwrite(r.obraz, np.zeros((1000, 1000, 3), np.uint8))
    eksportuj([r], tmp_path / "ds", ["chwast", "uprawa"], kopiuj_obrazy=True)
    tekst = (tmp_path / "ds" / "data.yaml").read_text()
    assert "0: chwast" in tekst and "1: uprawa" in tekst


def test_manifest_zapis_i_odczyt(tmp_path):
    r = _rek_z_obiektem((1, 2, 3, 4))
    M.zapisz([r], tmp_path / "m.jsonl")
    wczytane = M.wczytaj(tmp_path / "m.jsonl")
    assert len(wczytane) == 1
    assert wczytane[0].obiekty[0].xyxy == [1, 2, 3, 4]


# ---------------------------------------------------------------- caly potok

def test_caly_potok(dane, tmp_path):
    info = zbuduj(dane, tmp_path / "zbior", tryb_klas="binarnie", kafelek=320)
    ds = tmp_path / "zbior" / "dataset"
    assert (ds / "data.yaml").exists()
    assert info["kontrola_podzialu"]["bledy"] == {}
    assert set(info["klasy"]) <= {"chwast", "uprawa"}
    # kazdy obraz ma swoj plik etykiet
    for split in ("train", "val", "test"):
        obrazy = list((ds / "images" / split).glob("*")) if (ds / "images" / split).exists() else []
        for o in obrazy:
            assert (ds / "labels" / split / (o.stem + ".txt")).exists()


def test_potok_z_wlasnym_podzialem(dane, tmp_path):
    info = zbuduj(dane, tmp_path / "z2", kafelek=0, oficjalne_splity=False)
    assert info["kontrola_podzialu"]["bledy"] == {}
    assert info["kontrola_podzialu"]["oficjalne"] == {}


def test_cli_dziala(dane, tmp_path):
    r = subprocess.run([sys.executable, "-m", "normalization.build",
                        "--sudars", str(dane["sudars"]), "--wlasne", str(dane["wlasne"]),
                        "-o", str(tmp_path / "z"), "--kafelek", "320"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "z" / "dataset" / "data.yaml").exists()
    assert (tmp_path / "z" / "build_info.json").exists()


def test_sudars_nie_jest_jedna_grupa(dane):
    """Wszystkie XML-e w jednym folderze "annotations" - grupa nie moze byc jedna,
    bo caly zbior trafilby wtedy do jednej czesci podzialu."""
    t = Taxonomy()
    rek = ADAPTERY["sudars"](dane["sudars"], t)
    assert len({r.grupa for r in rek}) > 1


def test_kafelkowanie_zdjec_mniejszych_niz_kafelek():
    """WeedsGalore ma 600 px, domyslny kafelek 896 - zdjecie przechodzi bez ciecia.
    Wczesniej ta sciezka zwracala liste zamiast pary i potok sie wywracal."""
    male = _rek_z_obiektem((10, 10, 50, 50), szer=600, wys=600)
    puste = _rek_z_obiektem((10, 10, 50, 50), szer=600, wys=600)
    puste.obiekty = []
    for r in (male, puste):
        r.split = "train"
    wynik = K.kafelkuj_wszystkie([male, puste], kafelek=896, udzial_pustych=0.5, seed=0)
    assert male in wynik
    assert len(wynik) == 2                        # 1 z obiektem + 1 pusty przy udziale 50%


def test_kanaly_16bit_zachowuja_proporcje_kolorow(tmp_path):
    """Kanaly 16-bitowe o roznych zakresach - kolor nie moze sie przesunac.
    Wczesniej kazdy kanal rozciagany osobno: gleba wychodzila rozowa."""
    from normalization.export import _wczytaj_obraz
    from normalization.manifest import Record
    h = w = 40
    # "gleba": R > G > B; plus jasna "roslina" w zielonym kanale
    R = np.full((h, w), 3000, np.uint16)
    G = np.full((h, w), 2000, np.uint16)
    B = np.full((h, w), 1000, np.uint16)
    G[:10, :10] = 9000
    for k, c in (("R", R), ("G", G), ("B", B)):
        cv2.imwrite(str(tmp_path / f"x_{k}.png"), c)
    img = _wczytaj_obraz(Record(id="x", obraz=str(tmp_path / "x"), szer=w, wys=h, zrodlo="weedsgalore"))
    b, g, r = (int(v) for v in img[30, 30])     # piksel gleby
    assert r > g > b                             # kolejnosc kanalow zachowana
