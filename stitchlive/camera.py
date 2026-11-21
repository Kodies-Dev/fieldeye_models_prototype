"""Model kamery drona: ile pola widzi zdjecie z danej wysokosci i jak to przeniesc na symulator.

Symulator tnie kadry z obrazu zrodlowego o znanej rozdzielczosci terenowej
(GSD, mm pola na piksel - dla WeedsGalore ok. 2,5 mm/px). Zeby kadr odpowiadal
prawdziwemu zdjeciu z drona, liczymy z optyki kamery i wysokosci lotu, jaki
kawalek pola obejmuje zdjecie, i wycinamy z zrodla dokladnie tyle.

Kadr symulowany ma rozdzielczosc ZRODLA, a nie kamery: powiekszenie zrodla
do rozdzielczosci kamery nie dodaloby zadnych szczegolow. Na dronie zdjecie
dzieli sie na kafelki obejmujace tyle pola, co kafelek treningowy (1,5 m);
w symulatorze dzieje sie to samo, tylko w pikselach zrodla.

    from stitchlive.camera import ARDUCAM_64MP_16MP as kam
    kam.zasieg_m(5.0)            # (7.25, 5.44) m
    kam.gsd_mm(5.0)              # 1.57 mm/px
    kam.kadr_w_zrodle(5.0, 2.5)  # (2902, 2177) px zrodla

Uzywany w: app.py (zakladka 4, tryb kamery drona), stitchlive/cli.py (--wysokosc),
mission/detect_cli.py (GSD z wysokosci, gdy telemetria nie podaje gsd), tests/test_camera.py.
"""

import math
from dataclasses import dataclass

# Kafelek treningowy WeedsGalore: 600 px przy ok. 2,5 mm/px, czyli ok. 1,5 m pola.
# Kafelek o takim zasiegu daje na wejsciu sieci chwasty tej samej wielkosci co w treningu.
KAFELEK_TRENINGOWY_M = 1.5
GSD_WEEDSGALORE_MM = 2.5


@dataclass(frozen=True)
class CameraModel:
    """Kamera: matryca, ogniskowa i rozdzielczosc - z tego wszystkie liczby lotu.

    Pola:
        nazwa            - opis, np. "Arducam 64MP, tryb 16 MP (4624x3472)"
        matryca_szer_mm, matryca_wys_mm - rozmiar matrycy, np. 7.4 x 5.55
        ogniskowa_mm     - RZECZYWISTA ogniskowa obiektywu, np. 5.1
        obraz_szer_px, obraz_wys_px - rozdzielczosc zdjecia, np. 4624 x 3472

    frozen=True - obiekt niezmienny (stala kamery, nie da sie jej przypadkiem zmienic).

    Uzywana w:
        ARDUCAM_64MP_16MP, ARDUCAM_64MP_64MP, RPI_HQ_6MM (ten plik), opis_lotu()
    """
    nazwa: str
    matryca_szer_mm: float
    matryca_wys_mm: float
    ogniskowa_mm: float
    obraz_szer_px: int
    obraz_wys_px: int

    def zasieg_m(self, wysokosc_m):
        """(szerokosc, wysokosc) pola na zdjeciu w metrach - kamera skierowana pionowo w dol.

        Argumenty:
            wysokosc_m - wysokosc lotu nad polem, np. 5.0

        Zwraca:
            (szer_m, wys_m), np. (7.25, 5.44) dla Arducam z 5 m

        Uzywana w:
            gsd_mm(), kadr_w_zrodle(), opis_lotu() (ten plik), mission/spray_map.py gsd_m()
        """
        return (self.matryca_szer_mm / self.ogniskowa_mm * wysokosc_m,
                self.matryca_wys_mm / self.ogniskowa_mm * wysokosc_m)

    def gsd_mm(self, wysokosc_m):
        """Ile mm pola przypada na piksel zdjecia.

        Argumenty:
            wysokosc_m - wysokosc lotu, np. 5.0

        Zwraca:
            float, np. 1.57 (mm/px)
        """
        return self.zasieg_m(wysokosc_m)[0] * 1000.0 / self.obraz_szer_px

    def kadr_w_zrodle(self, wysokosc_m, gsd_zrodla_mm):
        """(szerokosc, wysokosc) kadru w pikselach zrodla o podanym GSD.

        Po co: symulator wycina z obrazu zrodlowego tyle pola, ile widzi kamera.

        Argumenty:
            wysokosc_m    - wysokosc lotu, np. 5.0
            gsd_zrodla_mm - GSD obrazu zrodlowego, np. 2.5 (WeedsGalore)

        Zwraca:
            (szer_px, wys_px), np. (2902, 2177)
        """
        w, h = self.zasieg_m(wysokosc_m)
        return round(w * 1000 / gsd_zrodla_mm), round(h * 1000 / gsd_zrodla_mm)

    def kafelek_px(self, gsd_mm, kafelek_m=KAFELEK_TRENINGOWY_M):
        """Bok kafelka w px obrazu o danym GSD, obejmujacego kafelek_m pola.

        Argumenty:
            gsd_mm    - GSD obrazu, np. 2.5
            kafelek_m - ile pola ma objac kafelek, domyslnie 1.5 m (jak przy treningu)

        Zwraca:
            int, np. 600 (najmniej 32)
        """
        return max(32, round(kafelek_m * 1000 / gsd_mm))


def rozmycie_px(predkosc_ms, czas_naswietlania_s, gsd_mm):
    """O ile pikseli przesuwa sie obraz w czasie naswietlania: v * t / GSD.

    Argumenty:
        predkosc_ms         - predkosc drona [m/s], np. 3.0
        czas_naswietlania_s - np. 0.001
        gsd_mm              - mm pola na piksel

    Zwraca:
        float [px], np. 1.2

    Uzywana w:
        opis_lotu() (ten plik)
    """
    return predkosc_ms * czas_naswietlania_s * 1000.0 / gsd_mm


def odstep_zdjec_s(zasieg_wzdluz_m, pokrycie, predkosc_ms):
    """Co ile sekund zdjecie, zeby kolejne zachodzily na siebie o `pokrycie`.

    Argumenty:
        zasieg_wzdluz_m - dlugosc pola na zdjeciu wzdluz lotu [m]
        pokrycie        - np. 0.7 (70%)
        predkosc_ms     - predkosc drona [m/s]

    Zwraca:
        float [s]

    Uzywana w:
        opis_lotu() (ten plik)
    """
    return zasieg_wzdluz_m * (1 - pokrycie) / predkosc_ms


# Kamera FIELDEYE: Arducam 64MP Hawkeye (matryca 1/1.7", 7,4 x 5,55 mm, obiektyw 5,1 mm).
ARDUCAM_64MP_16MP = CameraModel("Arducam 64MP, tryb 16 MP (4624x3472)", 7.4, 5.55, 5.1, 4624, 3472)
ARDUCAM_64MP_64MP = CameraModel("Arducam 64MP, tryb 64 MP (9152x6944)", 7.4, 5.55, 5.1, 9152, 6944)
# Poprzednie zalozenie projektu - do porownan.
RPI_HQ_6MM = CameraModel("Raspberry Pi HQ, obiektyw 6 mm (4056x3040)", 6.287, 4.712, 6.0, 4056, 3040)

KAMERY = {k.nazwa: k for k in (ARDUCAM_64MP_16MP, ARDUCAM_64MP_64MP, RPI_HQ_6MM)}


def opis_lotu(kamera, wysokosc_m, gsd_zrodla_mm, predkosc_ms, czas_naswietlania_s, pokrycie):
    """Wszystkie liczby lotu w jednym slowniku - do wyswietlenia i do ustawien symulatora.

    Argumenty:
        kamera              - CameraModel, np. ARDUCAM_64MP_16MP
        wysokosc_m          - wysokosc lotu, np. 5.0
        gsd_zrodla_mm       - GSD obrazu zrodlowego, np. 2.5
        predkosc_ms         - predkosc drona, np. 3.0
        czas_naswietlania_s - np. 0.001
        pokrycie            - pokrycie kolejnych zdjec, np. 0.7

    Zwraca:
        dict: zasieg_m, gsd_kamery_mm, kadr_px, kafelek_px, kafelkow_na_kadr,
        kafelek_kamery_px, rozmycie_px, rozmycie_kamery_px, odstep_zdjec_s

    Uzywana w:
        app.py (zakladka 4), stitchlive/cli.py _ustawienia() (--wysokosc)
    """
    zw, zh = kamera.zasieg_m(wysokosc_m)
    kw, kh = kamera.kadr_w_zrodle(wysokosc_m, gsd_zrodla_mm)
    kaf = kamera.kafelek_px(gsd_zrodla_mm)
    return {"zasieg_m": (round(zw, 2), round(zh, 2)),
            "gsd_kamery_mm": round(kamera.gsd_mm(wysokosc_m), 2),
            "kadr_px": (kw, kh),
            "kafelek_px": kaf,
            "kafelkow_na_kadr": math.ceil(kw / (kaf * 0.85)) * math.ceil(kh / (kaf * 0.85)),
            "kafelek_kamery_px": kamera.kafelek_px(kamera.gsd_mm(wysokosc_m)),
            # rozmycie w pikselach kadru symulowanego (GSD zrodla), czyli takie, jakie zobaczy model
            "rozmycie_px": round(rozmycie_px(predkosc_ms, czas_naswietlania_s, gsd_zrodla_mm), 2),
            "rozmycie_kamery_px": round(rozmycie_px(predkosc_ms, czas_naswietlania_s,
                                                    kamera.gsd_mm(wysokosc_m)), 2),
            "odstep_zdjec_s": round(odstep_zdjec_s(zw, pokrycie, predkosc_ms), 2)}