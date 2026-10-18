"""ETAP 1 - w locie. Kazde zdjecie sprawdzane osobno, od razu po zrobieniu.

Dla kazdego zdjecia po kolei:
    wczytaj -> telemetria (GPS, kurs, wysokosc) -> detekcja chwastow
    -> przelicz srodek kazdej ramki na wspolrzedne -> dopisz do pliku -> ZWOLNIJ

Dwie wlasnosci, ktore sa tu najwazniejsze:

1. Zdjecia NIE sa skladane w mozaike. Zeby nanniesc chwast na mape,
   wystarcza pozycja GPS, kurs i wysokosc tego jednego zdjecia. Skladanie
   jest ciezkie obliczeniowo i nie musi sie dziac w powietrzu.

2. Zajetosc pamieci jest stala, niezalezna od dlugosci misji. W pamieci
   jest najwyzej jedno zdjecie; wyniki dopisywane sa na biezaco do pliku
   JSONL (jeden wiersz = jedno zdjecie). Gdy zasilanie padnie w polowie
   lotu, wszystko do tego momentu jest juz na dysku.
"""

import json
import logging
import time
from pathlib import Path

import cv2

from . import geo
from .telemetry import czytaj_csv, zbuduj_kadr

log = logging.getLogger(__name__)

ROZSZERZENIA = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


class JsonlWriter:
    """Dopisuje wyniki wiersz po wierszu i od razu wypycha na dysk.

    flush() po kazdym zdjeciu kosztuje ulamek sekundy, a ratuje cala misje
    w razie nagłego odciecia zasilania.
    """

    def __init__(self, sciezka):
        self.sciezka = Path(sciezka)
        self.sciezka.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.sciezka, "a", encoding="utf-8")

    def dopisz(self, rekord):
        self._f.write(json.dumps(rekord, ensure_ascii=False) + "\n")
        self._f.flush()

    def zamknij(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.zamknij()


def przetworz_kadr(sciezka, detektor, cfg, uklad, telemetria=None):
    """Jedno zdjecie: detekcja + przeliczenie na wspolrzedne.

    Zwraca rekord (slownik) gotowy do zapisu. Gdy brak pozycji GPS,
    detekcje sa policzone, ale bez wspolrzednych - zdjecie i tak trafi
    pozniej do mozaiki.
    """
    t0 = time.time()
    kadr = zbuduj_kadr(sciezka, telemetria, cfg)
    obraz = cv2.imread(str(sciezka), cv2.IMREAD_COLOR)
    if obraz is None:
        raise IOError(f"nie mozna wczytac {sciezka}")
    kadr.wys, kadr.szer = obraz.shape[:2]

    detekcje = detektor.wykryj(obraz)
    del obraz                                   # zwalniamy od razu po detekcji

    rozdz = geo.gsd(kadr.agl, cfg.kamera.szer_matrycy_mm,
                    cfg.kamera.ogniskowa_mm, kadr.szer) if kadr.agl else None

    lista = []
    for d in detekcje:
        rec = d.slownik()
        if kadr.ma_pozycje and rozdz:
            px, py = d.srodek
            lat, lon = geo.piksel_na_wspolrzedne(px, py, kadr, rozdz, uklad)
            rec["lat"], rec["lon"] = round(lat, 8), round(lon, 8)
            # przyblizona wielkosc rosliny w terenie
            rec["rozmiar_m"] = [round((d.x1 - d.x0) * rozdz, 3),
                                round((d.y1 - d.y0) * rozdz, 3)]
        lista.append(rec)

    return {"kadr": kadr.slownik(), "gsd_m_px": round(rozdz, 6) if rozdz else None,
            "detekcje": lista, "czas_s": round(time.time() - t0, 3)}


def przetworz_katalog(katalog, wyjscie, detektor, cfg, telemetria_csv=None,
                      limit=None, postep=None):
    """Tryb wsadowy: przetwarza gotowy katalog zdjec (np. po locie, do testow)."""
    pliki = sorted(p for p in Path(katalog).iterdir() if p.suffix in ROZSZERZENIA)
    if limit:
        pliki = pliki[:limit]
    if not pliki:
        raise FileNotFoundError(f"brak zdjec w {katalog}")

    tele = czytaj_csv(telemetria_csv) if telemetria_csv else None
    uklad = _uklad_z_pierwszego(pliki, tele, cfg)

    wyjscie = Path(wyjscie)
    with JsonlWriter(wyjscie) as zapis:
        for i, p in enumerate(pliki, 1):
            try:
                rek = przetworz_kadr(p, detektor, cfg, uklad, tele)
            except Exception as e:                # jedno zle zdjecie nie konczy misji
                log.error("%s: %s", p.name, e)
                continue
            zapis.dopisz(rek)
            n = sum(1 for d in rek["detekcje"] if "lat" in d)
            log.info("[%d/%d] %s: %d detekcji (%d na mapie), %.2f s",
                     i, len(pliki), p.name, len(rek["detekcje"]), n, rek["czas_s"])
            if postep:
                postep(i, len(pliki))
    return wyjscie, uklad


def obserwuj_katalog(katalog, wyjscie, detektor, cfg, telemetria_csv=None,
                     interwal=1.0, timeout=None, postep=None):
    """Tryb w locie: czeka na nowe zdjecia i przetwarza je na biezaco.

    Zdjecie jest brane dopiero, gdy jego rozmiar przestanie rosnac -
    inaczej program zlapalby plik w polowie zapisu.
    """
    katalog = Path(katalog)
    katalog.mkdir(parents=True, exist_ok=True)
    tele = czytaj_csv(telemetria_csv) if telemetria_csv else None

    widziane, uklad = set(), None
    ostatnie, i = time.time(), 0

    with JsonlWriter(wyjscie) as zapis:
        while True:
            nowe = [p for p in sorted(katalog.iterdir())
                    if p.suffix in ROZSZERZENIA and p.name not in widziane
                    and _plik_gotowy(p)]
            for p in nowe:
                widziane.add(p.name)
                if uklad is None:
                    uklad = _uklad_z_pierwszego([p], tele, cfg)
                try:
                    rek = przetworz_kadr(p, detektor, cfg, uklad, tele)
                except Exception as e:
                    log.error("%s: %s", p.name, e)
                    continue
                zapis.dopisz(rek)
                i += 1
                n = sum(1 for d in rek["detekcje"] if "lat" in d)
                log.info("[%d] %s: %d detekcji (%d na mapie), %.2f s",
                         i, p.name, len(rek["detekcje"]), n, rek["czas_s"])
                if postep:
                    postep(i, None)
                ostatnie = time.time()

            if timeout and time.time() - ostatnie > timeout:
                log.info("Brak nowych zdjec przez %.0f s - koniec misji.", timeout)
                break
            time.sleep(interwal)
    return Path(wyjscie), uklad


def _plik_gotowy(p, stabilizacja=0.4):
    try:
        r1 = p.stat().st_size
        time.sleep(stabilizacja)
        return r1 > 0 and p.stat().st_size == r1
    except OSError:
        return False


def _uklad_z_pierwszego(pliki, tele, cfg):
    """Uklad lokalny zaczepiamy w pierwszym zdjeciu z pozycja GPS."""
    for p in pliki:
        k = zbuduj_kadr(p, tele, cfg)
        if k.ma_pozycje:
            log.info("Uklad odniesienia: %.6f, %.6f", k.lat, k.lon)
            return geo.LocalFrame(k.lat, k.lon)
    log.warning("Zadne zdjecie nie ma pozycji GPS - detekcje bez wspolrzednych.")
    return None


def wczytaj_wyniki(sciezka):
    """Odczyt pliku JSONL z etapu w locie."""
    with open(sciezka, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
