"""Przeliczanie pikseli na wspolrzedne geograficzne.

Kluczowa obserwacja calego etapu w locie: zeby nanniesc chwast na mape,
NIE potrzeba mozaiki. Wystarcza pozycja GPS, kurs i wysokosc lotu tego
jednego zdjecia. Mozaika jest potrzebna dopiero czlowiekowi - do ogladania.

Dlatego detekcja i mapowanie moga sie dziac w locie, kadr po kadrze,
a skladanie mozaiki dopiero po wyladowaniu.

Zakladamy zdjecia z nadiru (kamera pionowo w dol) nad plaskim terenem.
Przy wyraznym przechyle drona pojawia sie blad - patrz README, sekcja
"Ograniczenia".
"""

import math

from pyproj import CRS, Transformer


def gsd(agl_m, szer_matrycy_mm, ogniskowa_mm, szer_obrazu_px):
    """Ile metrow terenu przypada na jeden piksel (Ground Sample Distance).

    Z podobienstwa trojkatow w modelu kamery otworkowej:
        GSD = (wysokosc * szerokosc_matrycy) / (ogniskowa * szerokosc_obrazu)

    UWAGA: ogniskowa musi byc RZECZYWISTA (z karty katalogowej obiektywu),
    nie "ekwiwalent 35 mm" - to najczestszy blad, daje kilkukrotnie zla skale.
    """
    if ogniskowa_mm <= 0 or szer_obrazu_px <= 0:
        raise ValueError("ogniskowa i szerokosc obrazu musza byc dodatnie")
    return (agl_m * szer_matrycy_mm) / (ogniskowa_mm * szer_obrazu_px)


def strefa_utm(lat, lon):
    """Kod EPSG strefy UTM dla punktu. Polska poludniowa: strefa 34 -> 32634."""
    strefa = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + strefa


class LocalFrame:
    """Przelicznik WGS84 (stopnie) <-> lokalny uklad metryczny (metry).

    W stopniach nie da sie liczyc geometrii: stopien dlugosci ma inna
    dlugosc w metrach niz stopien szerokosci i zmienia sie z szerokoscia
    geograficzna. Dlatego wszystko liczymy w metrach (UTM), a na stopnie
    przechodzimy dopiero przy zapisie wyniku.
    """

    __slots__ = ("crs", "_do_utm", "_do_wgs", "ox", "oy", "ref_lat", "ref_lon")

    def __init__(self, ref_lat, ref_lon):
        self.crs = CRS.from_epsg(strefa_utm(ref_lat, ref_lon))
        self._do_utm = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._do_wgs = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        self.ref_lat, self.ref_lon = ref_lat, ref_lon
        # poczatek ukladu lokalnego w srodku misji - wspolrzedne sa wtedy male
        self.ox, self.oy = self._do_utm.transform(ref_lon, ref_lat)

    @property
    def epsg(self):
        return self.crs.to_epsg()

    def na_metry(self, lat, lon):
        """-> (east, north) w metrach wzgledem punktu odniesienia."""
        x, y = self._do_utm.transform(lon, lat)
        return x - self.ox, y - self.oy

    def na_stopnie(self, east, north):
        """-> (lat, lon)."""
        lon, lat = self._do_wgs.transform(east + self.ox, north + self.oy)
        return lat, lon


def piksel_na_metry(px, py, szer_px, wys_px, gsd_m, kurs_st):
    """Piksel zdjecia -> przesuniecie (east, north) w metrach od srodka kadru.

    Uklad obrazu: x w prawo, y W DOL. Uklad terenu: E na wschod, N na polnoc.
    Kurs liczony zgodnie z ruchem wskazowek zegara od polnocy (0 = polnoc).

    Kierunek "przod drona" to gora obrazu (-y). Przy kursie theta:
        przod  = ( sin theta,  cos theta)   w (E, N)
        prawo  = ( cos theta, -sin theta)

    Punkt (dx, dy) w obrazie = dx * prawo + (-dy) * przod, stad wzory nizej.
    """
    dx = (px - szer_px / 2.0) * gsd_m
    dy = (py - wys_px / 2.0) * gsd_m
    t = math.radians(kurs_st or 0.0)
    east = dx * math.cos(t) - dy * math.sin(t)
    north = -dx * math.sin(t) - dy * math.cos(t)
    return east, north


def piksel_na_wspolrzedne(px, py, kadr, gsd_m, uklad):
    """Piksel zdjecia -> (lat, lon). kadr: obiekt z lat, lon, kurs, szer, wys."""
    e0, n0 = uklad.na_metry(kadr.lat, kadr.lon)
    de, dn = piksel_na_metry(px, py, kadr.szer, kadr.wys, gsd_m, kadr.kurs)
    return uklad.na_stopnie(e0 + de, n0 + dn)


def zasieg_kadru_m(szer_px, wys_px, gsd_m):
    """Ile metrow terenu obejmuje kadr - do raportu i planowania misji."""
    return szer_px * gsd_m, wys_px * gsd_m


def odleglosc_m(lat1, lon1, lat2, lon2):
    """Odleglosc dwoch punktow w metrach (wzor haversine).

    Uzywane do scalania tej samej rosliny widzianej na kilku zdjeciach.
    """
    R = 6371000.0
    f1, f2 = math.radians(lat1), math.radians(lat2)
    df, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(df / 2) ** 2 + math.cos(f1) * math.cos(f2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
