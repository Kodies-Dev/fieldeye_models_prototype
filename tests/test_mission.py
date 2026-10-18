"""Testy potoku misji: geometria, telemetria, detektor, scalanie.

Najwazniejszy jest test_znacznik_trafia_we_wlasciwe_miejsce - sprawdza caly
lancuch na znacznikach umieszczonych w ZNANYCH miejscach obrazu.
"""

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mission import geo  # noqa: E402
from mission.detector import Detection, DummyDetector, TiledDetector  # noqa: E402
from mission.config import MissionConfig                        # noqa: E402
from mission.post_flight import raport, scal_detekcje, zapisz_geojson    # noqa: E402
from mission.telemetry import czytaj_csv, zbuduj_kadr               # noqa: E402
from mission.in_flight import przetworz_katalog, wczytaj_wyniki         # noqa: E402

LAT0, LON0, GSD, KADR, AGL = 50.2645, 19.0231, 0.01, 640, 30.0
SENS = 6.287
FOC = AGL * SENS / (GSD * KADR)      # ogniskowa dajaca dokladnie GSD = 1 cm/px


# ---------------------------------------------------------------- geometria

def test_gsd_wzor():
    assert geo.gsd(30, 6.287, 6.0, 4056) == pytest.approx(0.00775, abs=1e-5)


def test_gsd_rosnie_liniowo_z_wysokoscia():
    assert geo.gsd(60, 6.287, 6.0, 4056) == pytest.approx(2 * geo.gsd(30, 6.287, 6.0, 4056))


def test_gsd_odrzuca_bledne_dane():
    with pytest.raises(ValueError):
        geo.gsd(30, 6.287, 0, 4056)


def test_strefa_utm_dla_polski():
    assert geo.strefa_utm(50.26, 19.02) == 32634


@pytest.mark.parametrize("kurs, oczek", [
    (0,   (0.0, 100.0)),      # kurs polnoc: gora kadru = polnoc
    (90,  (100.0, 0.0)),      # kurs wschod: gora kadru = wschod
    (180, (0.0, -100.0)),
    (270, (-100.0, 0.0)),
])
def test_kierunki_gora_kadru(kurs, oczek):
    """Gora kadru (mniejsze y) to zawsze kierunek lotu."""
    e, n = geo.piksel_na_metry(320, 220, 640, 640, 1.0, kurs)
    assert e == pytest.approx(oczek[0], abs=1e-6)
    assert n == pytest.approx(oczek[1], abs=1e-6)


def test_srodek_kadru_to_pozycja_gps():
    u = geo.LocalFrame(LAT0, LON0)
    e, n = geo.piksel_na_metry(320, 320, 640, 640, 0.01, 123)
    assert (e, n) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_uklad_tam_i_z_powrotem():
    u = geo.LocalFrame(LAT0, LON0)
    e, n = u.na_metry(50.2650, 19.0240)
    lat, lon = u.na_stopnie(e, n)
    assert (lat, lon) == pytest.approx((50.2650, 19.0240), abs=1e-9)


def test_odleglosc_stopien_szerokosci():
    """0,001 stopnia szerokosci to ok. 111 m."""
    assert geo.odleglosc_m(50.0, 19.0, 50.001, 19.0) == pytest.approx(111.2, abs=1.0)


# ---------------------------------------------------------------- dane testowe

def _dms(v):
    v = abs(v)
    d = int(v)
    m = int((v - d) * 60)
    s = (v - d - m / 60) * 3600
    return ((d, 1), (m, 1), (int(round(s * 10000)), 10000))


def _latlon(px, py):
    return (LAT0 - py * GSD / 111320.0,
            LON0 + px * GSD / (111320.0 * math.cos(math.radians(LAT0))))


@pytest.fixture(scope="module")
def lot(tmp_path_factory):
    """Symulowany lot: pole ze znacznikami w znanych miejscach, kadry z EXIF."""
    piexif = pytest.importorskip("piexif")
    cel = tmp_path_factory.mktemp("lot")
    (cel / "images").mkdir()

    W, H, nx, ny = 2400, 1400, 5, 3
    rng = np.random.default_rng(3)
    pole = np.full((H, W, 3), (60, 75, 95), np.uint8)
    pole = np.clip(pole + rng.normal(0, 9, (H, W, 1)), 0, 255).astype(np.uint8)
    for x in range(0, W, 40):
        cv2.line(pole, (x, 0), (x, H), (55, 140, 60), 11)
    znaczniki = [(300, 300), (760, 420), (1180, 700), (1600, 500), (2050, 900), (900, 1100)]
    for mx, my in znaczniki:
        cv2.circle(pole, (mx, my), 16, (90, 245, 120), -1)

    kx, ky = (W - KADR) // (nx - 1), (H - KADR) // (ny - 1)
    wiersze, kadry, idx = [], [], 0
    for j in range(ny):
        xs = range(nx) if j % 2 == 0 else range(nx - 1, -1, -1)
        kurs = 90.0 if j % 2 == 0 else 270.0
        for i in xs:
            idx += 1
            x0, y0 = i * kx, j * ky
            kadr = pole[y0:y0 + KADR, x0:x0 + KADR].copy()
            # obrot zgodny z konwencja: gora kadru = kierunek lotu
            M = cv2.getRotationMatrix2D((KADR / 2, KADR / 2), kurs, 1.0)
            kadr = cv2.warpAffine(kadr, M, (KADR, KADR), borderMode=cv2.BORDER_REFLECT)
            lat, lon = _latlon(x0 + KADR / 2, y0 + KADR / 2)
            nazwa = f"IMG_{idx:04d}.jpg"
            cv2.imwrite(str(cel / "images" / nazwa), kadr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            gps = {piexif.GPSIFD.GPSLatitudeRef: b"N", piexif.GPSIFD.GPSLatitude: _dms(lat),
                   piexif.GPSIFD.GPSLongitudeRef: b"E", piexif.GPSIFD.GPSLongitude: _dms(lon),
                   piexif.GPSIFD.GPSAltitudeRef: 0, piexif.GPSIFD.GPSAltitude: (int(AGL * 100), 100),
                   piexif.GPSIFD.GPSImgDirectionRef: b"T",
                   piexif.GPSIFD.GPSImgDirection: (int(kurs * 100), 100)}
            piexif.insert(piexif.dump({"0th": {}, "Exif": {}, "GPS": gps, "1st": {},
                                       "thumbnail": None}), str(cel / "images" / nazwa))
            wiersze.append(dict(plik=nazwa, lat=f"{lat:.8f}", lon=f"{lon:.8f}",
                                agl=AGL, kurs=kurs))
            kadry.append((nazwa, x0, y0, kurs))

    with open(cel / "telemetria.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(wiersze[0]))
        w.writeheader()
        w.writerows(wiersze)

    cfg = MissionConfig()
    cfg.kamera.szer_matrycy_mm = SENS
    cfg.kamera.ogniskowa_mm = FOC
    cfg.lot.agl_domyslne = AGL
    return dict(kat=cel, kadry=kadry, cfg=cfg,
                znaczniki=[dict(px=p[0], py=p[1], lat=_latlon(*p)[0], lon=_latlon(*p)[1])
                           for p in znaczniki])


# ---------------------------------------------------------------- telemetria

def test_exif_czytany_poprawnie(lot):
    k = zbuduj_kadr(lot["kat"] / "images" / "IMG_0001.jpg", None, lot["cfg"])
    assert k.ma_pozycje
    assert (k.szer, k.wys) == (KADR, KADR)
    assert k.agl == pytest.approx(AGL, abs=0.05)
    assert k.kurs == pytest.approx(90.0, abs=0.1)


def test_csv_nadpisuje_exif(lot):
    tele = czytaj_csv(lot["kat"] / "telemetria.csv")
    k = zbuduj_kadr(lot["kat"] / "images" / "IMG_0001.jpg", tele, lot["cfg"])
    assert k.ma_pozycje and k.agl == pytest.approx(AGL)


def test_brak_kursu_bierze_domyslny(lot, tmp_path):
    cfg = MissionConfig()
    cfg.lot.kurs_domyslny = 42.0
    img = tmp_path / "bez_exif.jpg"
    cv2.imwrite(str(img), np.zeros((100, 120, 3), np.uint8))
    k = zbuduj_kadr(img, None, cfg)
    assert not k.ma_pozycje                      # brak GPS
    assert k.kurs == 42.0 and k.agl == cfg.lot.agl_domyslne


# ---------------------------------------------------------------- caly lancuch

def test_znacznik_trafia_we_wlasciwe_miejsce(lot):
    """NAJWAZNIEJSZY TEST: znacznik o znanym polozeniu -> wyliczone wspolrzedne.

    Blad ponizej 20 cm przy GSD 1 cm/px, czyli ponizej 20 pikseli.
    Reszta to zaokraglenie pozycji GPS zapisanej w EXIF.
    """
    cfg = lot["cfg"]
    tele = czytaj_csv(lot["kat"] / "telemetria.csv")
    u = geo.LocalFrame(LAT0, LON0)
    bledy = []
    for z in lot["znaczniki"]:
        for nazwa, x0, y0, kurs in lot["kadry"]:
            ox, oy = z["px"] - x0, z["py"] - y0
            if not (20 <= ox < KADR - 20 and 20 <= oy < KADR - 20):
                continue
            M = cv2.getRotationMatrix2D((KADR / 2, KADR / 2), kurs, 1.0)
            px, py = M @ np.array([ox, oy, 1.0])
            k = zbuduj_kadr(lot["kat"] / "images" / nazwa, tele, cfg)
            g = geo.gsd(k.agl, cfg.kamera.szer_matrycy_mm, cfg.kamera.ogniskowa_mm, k.szer)
            lat, lon = geo.piksel_na_wspolrzedne(px, py, k, g, u)
            bledy.append(geo.odleglosc_m(z["lat"], z["lon"], lat, lon))
            break
    assert len(bledy) == len(lot["znaczniki"])
    assert max(bledy) < 0.20, f"najwiekszy blad {max(bledy):.2f} m"


def test_gsd_zgodne_z_zalozeniem(lot):
    cfg = lot["cfg"]
    assert geo.gsd(AGL, cfg.kamera.szer_matrycy_mm,
                   cfg.kamera.ogniskowa_mm, KADR) == pytest.approx(GSD, abs=1e-6)


def test_przetworz_katalog_daje_wspolrzedne(lot, tmp_path):
    det = DummyDetector()
    plik, u = przetworz_katalog(lot["kat"] / "images", tmp_path / "w.jsonl",
                                det, lot["cfg"], lot["kat"] / "telemetria.csv")
    wyniki = wczytaj_wyniki(plik)
    assert len(wyniki) == len(lot["kadry"])
    assert all(r["kadr"]["lat"] is not None for r in wyniki)
    z_pozycja = [d for r in wyniki for d in r["detekcje"] if "lat" in d]
    assert len(z_pozycja) > 0
    assert all(49 < d["lat"] < 51 and 18 < d["lon"] < 20 for d in z_pozycja)


def test_wyniki_zapisywane_na_biezaco(lot, tmp_path):
    """Plik ma byc czytelny nawet gdy misja zostanie przerwana."""
    plik = tmp_path / "w.jsonl"
    przetworz_katalog(lot["kat"] / "images", plik, DummyDetector(), lot["cfg"], limit=3)
    wiersze = plik.read_text(encoding="utf-8").strip().split("\n")
    assert len(wiersze) == 3
    for w in wiersze:                             # kazdy wiersz osobnym JSON-em
        json.loads(w)


# ---------------------------------------------------------------- detektor

def test_atrapa_zwraca_ramki_w_obrazie():
    img = np.full((300, 400, 3), (60, 70, 90), np.uint8)
    cv2.circle(img, (200, 150), 20, (60, 240, 110), -1)
    det = DummyDetector().wykryj(img)
    assert len(det) >= 1
    for d in det:
        assert 0 <= d.x0 < d.x1 <= 400 and 0 <= d.y0 < d.y1 <= 300


def test_atrapa_przechodzi_domyslny_prog_mapy():
    """Pseudo-pewnosc atrapy musi byc powyzej domyslnego progu mapy (0.30).

    Inaczej pelna misja z domyslna konfiguracja dawala pusta mape bez
    zadnego komunikatu - trudny do zdiagnozowania blad.
    """
    from mission.config import MissionConfig

    img = np.full((400, 400, 3), (60, 70, 90), np.uint8)
    for x, y in ((80, 80), (200, 150), (320, 300)):
        cv2.circle(img, (x, y), 14, (60, 240, 110), -1)
    det = DummyDetector().wykryj(img)
    assert det
    prog = MissionConfig().mapa.min_pewnosc_na_mape
    assert all(d.pewnosc >= prog for d in det)


def test_atrapa_ma_wlasna_klase():
    """Zeby wynik atrapy nie zostal wziety za prawdziwa detekcje chwastu."""
    img = np.full((200, 200, 3), (60, 70, 90), np.uint8)
    cv2.circle(img, (100, 100), 15, (60, 240, 110), -1)
    assert all(d.klasa == "atrapa" for d in DummyDetector().wykryj(img))


def test_kafelki_przeliczaja_wspolrzedne_na_caly_kadr():
    """Detekcja z kafelka musi trafic we wlasciwe miejsce calego zdjecia."""
    class _ConstantDetector:
        nazwa = "staly"
        def wykryj(self, obraz):
            return [Detection(10, 10, 30, 30, "x", 0.9)]

    d = TiledDetector(_ConstantDetector(), kafelek=100, zakladka=0.0)
    wynik = d.wykryj(np.zeros((300, 300, 3), np.uint8))
    assert len(wynik) == 9                        # siatka 3x3 kafelkow
    # ramka (10,10) w kafelku o poczatku (x,y) ma trafic na (x+10, y+10)
    rogi = {(int(r.x0), int(r.y0)) for r in wynik}
    assert (10, 10) in rogi and (110, 110) in rogi and (210, 210) in rogi


def test_kafelki_usuwaja_duplikaty_z_zakladki():
    class _ConstantDetector:
        nazwa = "staly"
        def wykryj(self, obraz):
            return [Detection(0, 0, 100, 100, "x", 0.9)]

    d = TiledDetector(_ConstantDetector(), kafelek=100, zakladka=0.5, prog_iou=0.3)
    # przy duzej zakladce te same ramki sie pokrywaja - maja zostac scalone
    assert len(d.wykryj(np.zeros((150, 150, 3), np.uint8))) < 4


def test_maly_kadr_bez_kafelkowania():
    class _ConstantDetector:
        nazwa = "staly"
        def wykryj(self, obraz):
            return [Detection(1, 1, 5, 5, "x", 0.5)]

    d = TiledDetector(_ConstantDetector(), kafelek=896)
    assert len(d.wykryj(np.zeros((640, 640, 3), np.uint8))) == 1


# ---------------------------------------------------------------- scalanie

def _wynik(punkty):
    return [{"kadr": {"plik": f"IMG_{i}.jpg", "lat": 50.0, "lon": 19.0},
             "gsd_m_px": 0.01,
             "detekcje": [{"bbox": [0, 0, 10, 10], "klasa": k, "pewnosc": p,
                           "lat": la, "lon": lo}]}
            for i, (la, lo, k, p) in enumerate(punkty)]


def test_scalanie_laczy_bliskie_punkty():
    # dwa punkty 10 cm od siebie - to ta sama roslina
    w = _wynik([(50.0, 19.0, "chwast", 0.9), (50.0 + 0.1 / 111320, 19.0, "chwast", 0.7)])
    ros = scal_detekcje(w, promien_m=0.35)
    assert len(ros) == 1
    assert ros[0]["obserwacje"] == 2
    assert ros[0]["pewnosc"] == 0.9               # zostaje najpewniejsza


def test_scalanie_nie_laczy_dalekich():
    w = _wynik([(50.0, 19.0, "chwast", 0.9), (50.0 + 5.0 / 111320, 19.0, "chwast", 0.8)])
    assert len(scal_detekcje(w, promien_m=0.35)) == 2


def test_scalanie_nie_laczy_roznych_klas():
    w = _wynik([(50.0, 19.0, "chwast", 0.9), (50.0, 19.0, "uprawa", 0.8)])
    assert len(scal_detekcje(w, promien_m=0.35)) == 2


def test_scalanie_filtruje_po_pewnosci():
    w = _wynik([(50.0, 19.0, "chwast", 0.9), (50.0 + 5.0 / 111320, 19.0, "chwast", 0.1)])
    assert len(scal_detekcje(w, promien_m=0.35, min_pewnosc=0.5)) == 1


def test_scalanie_filtruje_po_klasie():
    w = _wynik([(50.0, 19.0, "chwast", 0.9), (50.0 + 5.0 / 111320, 19.0, "uprawa", 0.9)])
    assert len(scal_detekcje(w, promien_m=0.35, klasy=("chwast",))) == 1


def test_detekcje_bez_wspolrzednych_pomijane():
    w = [{"kadr": {"plik": "a.jpg"}, "gsd_m_px": None,
          "detekcje": [{"bbox": [0, 0, 1, 1], "klasa": "chwast", "pewnosc": 0.9}]}]
    assert scal_detekcje(w) == []


# ---------------------------------------------------------------- wyjscie

def test_geojson_ma_poprawna_strukture(tmp_path):
    ros = [{"lat": 50.0, "lon": 19.0, "klasa": "chwast", "pewnosc": 0.9,
            "obserwacje": 2, "kadry": ["a.jpg"], "rozmiar_m": [0.1, 0.1]}]
    gj = json.loads(zapisz_geojson(ros, tmp_path / "x.geojson").read_text())
    assert gj["type"] == "FeatureCollection"
    f = gj["features"][0]
    assert f["geometry"]["coordinates"] == [19.0, 50.0]      # GeoJSON: [lon, lat]
    assert f["properties"]["obserwacje"] == 2


def test_raport_liczy_duplikaty():
    w = _wynik([(50.0, 19.0, "chwast", 0.9), (50.0 + 0.1 / 111320, 19.0, "chwast", 0.7)])
    ros = scal_detekcje(w, promien_m=0.35)
    r = raport(w, ros)
    assert r["detekcji_lacznie"] == 2 and r["roslin_po_scaleniu"] == 1
    assert r["duplikatow_usunietych"] == 1


# ---------------------------------------------------------------- CLI

def test_cli_pelna_misja(lot, tmp_path):
    r = subprocess.run([sys.executable, "-m", "mission.cli", "wlocie",
                        str(lot["kat"] / "images"), "-o", str(tmp_path),
                        "--telemetria", str(lot["kat"] / "telemetria.csv"),
                        "--agl", str(AGL)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "wyniki.jsonl").exists()

    r = subprocess.run([sys.executable, "-m", "mission.cli", "polocie",
                        str(tmp_path / "wyniki.jsonl"), "-o", str(tmp_path)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    for plik in ("chwasty.geojson", "slad_lotu.geojson", "mapa.html", "raport.json"):
        assert (tmp_path / plik).exists(), plik
