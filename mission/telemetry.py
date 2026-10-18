"""Skad program wie, gdzie bylo zrobione zdjecie.

Dwa zrodla, w kolejnosci waznosci:
  1. plik CSV z telemetria kapsuly (GPS + IMU + barometr) - dokladniejszy,
     bo barometr daje wysokosc NAD TERENEM, a nie nad poziomem morza
  2. EXIF zdjecia - zapisuja go drony komercyjne (DJI itp.)

Jesli brakuje kursu, uzywany jest kurs domyslny z konfiguracji; jesli
brakuje pozycji - zdjecie nie da sie zgeoreferowac i jest pomijane
(ale trafia do mozaiki po wyladowaniu).
"""

import csv
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2

log = logging.getLogger(__name__)


@dataclass
class FrameInfo:
    """Jedno zdjecie wraz z tym, co wiemy o momencie jego zrobienia."""
    plik: str
    szer: int = 0
    wys: int = 0
    lat: float = None
    lon: float = None
    agl: float = None          # wysokosc nad terenem [m] - z barometru
    kurs: float = None         # stopnie, 0 = polnoc, zgodnie z zegarem
    przechyl: float = None     # roll [st.] - na razie tylko zapisywany
    pochylenie: float = None   # pitch [st.]
    czas: float = None

    @property
    def ma_pozycje(self):
        return self.lat is not None and self.lon is not None

    def slownik(self):
        return asdict(self)


# ---------------------------------------------------------------- EXIF

def _dms_na_stopnie(dms, ref):
    d = float(dms[0][0]) / float(dms[0][1])
    m = float(dms[1][0]) / float(dms[1][1])
    s = float(dms[2][0]) / float(dms[2][1])
    v = d + m / 60.0 + s / 3600.0
    return -v if ref in (b"S", b"W", "S", "W") else v


def _ulamek(v):
    try:
        return float(v[0]) / float(v[1])
    except Exception:
        return None


def czytaj_exif(sciezka):
    """Pozycja, wysokosc i kurs z EXIF. Pusty slownik, gdy brak danych."""
    try:
        import piexif
    except ImportError:
        return {}
    try:
        exif = piexif.load(str(sciezka))
    except Exception:
        return {}
    gps = exif.get("GPS") or {}
    if not gps:
        return {}

    out = {}
    try:
        if piexif.GPSIFD.GPSLatitude in gps:
            out["lat"] = _dms_na_stopnie(gps[piexif.GPSIFD.GPSLatitude],
                                         gps.get(piexif.GPSIFD.GPSLatitudeRef, b"N"))
            out["lon"] = _dms_na_stopnie(gps[piexif.GPSIFD.GPSLongitude],
                                         gps.get(piexif.GPSIFD.GPSLongitudeRef, b"E"))
        if piexif.GPSIFD.GPSAltitude in gps:
            alt = _ulamek(gps[piexif.GPSIFD.GPSAltitude])
            if alt is not None:
                if gps.get(piexif.GPSIFD.GPSAltitudeRef, 0) == 1:
                    alt = -alt
                # UWAGA: to wysokosc nad poziomem morza, nie nad terenem
                out["alt_npm"] = alt
        if piexif.GPSIFD.GPSImgDirection in gps:
            out["kurs"] = _ulamek(gps[piexif.GPSIFD.GPSImgDirection])
    except Exception as e:
        log.debug("EXIF %s: %s", sciezka, e)
    return out


# ---------------------------------------------------------------- CSV

def czytaj_csv(sciezka):
    """Telemetria z kapsuly. Naglowek:

        plik,lat,lon,agl,kurs,przechyl,pochylenie,czas

    Wymagane: plik, lat, lon. Reszta opcjonalna.
    Zwraca {nazwa_pliku: {pole: wartosc}}.
    """
    out = {}
    with open(sciezka, newline="", encoding="utf-8") as f:
        for w in csv.DictReader(f):
            nazwa = w.get("plik") or w.get("filename") or w.get("nazwa")
            if not nazwa:
                continue
            rec = {}
            for k in ("lat", "lon", "agl", "kurs", "przechyl", "pochylenie", "czas"):
                v = w.get(k)
                if v not in (None, ""):
                    try:
                        rec[k] = float(v)
                    except ValueError:
                        pass
            out[Path(nazwa).name] = rec
    return out


# ---------------------------------------------------------------- razem

def zbuduj_kadr(sciezka, telemetria=None, cfg=None):
    """Sklada Kadr ze zdjecia + EXIF + (nadpisujacej go) telemetrii z CSV."""
    sciezka = Path(sciezka)
    dane = czytaj_exif(sciezka)
    if telemetria:
        dane.update(telemetria.get(sciezka.name, {}))

    img = cv2.imread(str(sciezka), cv2.IMREAD_REDUCED_COLOR_8)   # tylko po rozmiar
    if img is None:
        raise IOError(f"nie mozna wczytac {sciezka}")
    wys, szer = img.shape[0] * 8, img.shape[1] * 8

    agl = dane.get("agl")
    if agl is None and dane.get("alt_npm") is not None and cfg is not None:
        # wysokosc nad terenem = nad poziomem morza minus wysokosc pola
        agl = dane["alt_npm"] - cfg.wysokosc_terenu_npm
    if agl is None and cfg is not None:
        agl = cfg.agl_domyslne

    kurs = dane.get("kurs")
    if kurs is None and cfg is not None:
        kurs = cfg.kurs_domyslny
    if kurs is not None and cfg is not None:
        kurs += cfg.korekta_kursu        # montaz kamery wzgledem osi drona

    return FrameInfo(plik=sciezka.name, szer=szer, wys=wys,
                lat=dane.get("lat"), lon=dane.get("lon"), agl=agl, kurs=kurs,
                przechyl=dane.get("przechyl"), pochylenie=dane.get("pochylenie"),
                czas=dane.get("czas"))
