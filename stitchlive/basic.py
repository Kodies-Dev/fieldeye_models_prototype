"""Skladacz 'basic' - opakowanie gotowego cv2.Stitcher z OpenCV.

Co robi
-------
Po KAZDYM nowym zdjeciu sklada mozaike od nowa ze wszystkich dotychczasowych
zdjec jednym wywolaniem cv2.Stitcher (tryb SCANS - kamera przesuwana nad
plaskim terenem, model afiniczny; PANORAMA - kamera obracana w miejscu).
Wszystko robi OpenCV w srodku: punkty, dopasowanie, polozenie kadrow,
wyrownanie jasnosci, szwy, mieszanie.

Po co
-----
Punkt odniesienia dla pozostalych metod:
  - czy nasz material w ogole daje sie zlozyc (najprostsza droga, bez naszego kodu),
  - jak szybko rosnie czas, gdy skladamy wszystko od nowa (raport_czasu) -
    to uzasadnia metody przyrostowe: detail.py i global_stitcher.py.

Wady (dlatego nie jest domyslna)
--------------------------------
  - czas rosnie z kazdym kadrem (kazde dodaj() wczytuje i sklada wszystko od nowa),
  - przy kadrach obroconych o kilka stopni mozaika wychodzi rozmyta
    (to samo co w detail.py) - stad metoda 'global',
  - przy bledzie zwraca tylko kod (KODY), bez szczegolow, ktory kadr zawinil.

Interfejs
---------
Ten sam co pozostale skladacze, zeby app.py i cli.py mogly ich uzywac zamiennie:
    s = BasicStitcher(downscale=0.5)
    for sciezka in zdjecia:
        s.dodaj(sciezka)        # True, gdy mozaika zostala zlozona
    s.mozaika                   # gotowy obraz (BGR) albo None
    s.raport_czasu()            # [{"zdjec", "czas_s", "ok"}, ...]
Brak zakoncz() i raport() - nie sa potrzebne (mozaika zawsze aktualna);
app.py sprawdza je przez hasattr().

Uzywany w
---------
  app.py, zakladka 3 (linia 390) - metoda "basic";
      downscale = min(1, Mpx_z_suwaka / 0.6 * 0.5), przy 0.6 Mpx -> 0.5
  cli.py (linia 27) - --metoda basic (batch / watch / demo)
"""

import logging
import time

import cv2

log = logging.getLogger(__name__)

# kody bledow cv2.Stitcher przetlumaczone na komunikaty
KODY = {
    cv2.Stitcher_OK: "OK",
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "za malo zdjec albo za male pokrycie",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "nie udalo sie wyznaczyc homografii",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "nie udalo sie dostroic parametrow kamer",
}


class BasicStitcher:
    """Pelne przeliczenie mozaiki po kazdym nowym zdjeciu."""

    def __init__(self, downscale=0.5, tryb="scans", min_zdjec=2):
        self.downscale = downscale
        self.min_zdjec = min_zdjec
        self.sciezki = []
        self.mozaika = None
        self.ostatni_czas = 0.0
        self.historia = []          # (liczba_zdjec, czas_s, sukces)

        # SCANS - kamera przesuwana nad plaska scena (dron, skaner) -> model afiniczny.
        # PANORAMA - kamera obracana w miejscu -> model rotacyjny. Dla drona: SCANS.
        self._tryb = (cv2.Stitcher_SCANS if tryb == "scans"
                      else cv2.Stitcher_PANORAMA)

    def _wczytaj(self, sciezka):
        img = cv2.imread(str(sciezka))
        if img is None:
            raise IOError(f"Nie mozna wczytac: {sciezka}")
        if self.downscale != 1.0:
            img = cv2.resize(img, None, fx=self.downscale, fy=self.downscale,
                             interpolation=cv2.INTER_AREA)
        return img

    def dodaj(self, sciezka):
        """Dodaje zdjecie i przelicza mozaike. Zwraca True przy sukcesie."""
        self.sciezki.append(sciezka)
        if len(self.sciezki) < self.min_zdjec:
            return False

        t0 = time.time()
        # Wszystkie dotychczasowe zdjecia od nowa - dlatego czas rosnie z kazdym kadrem.
        # To swiadomie najprostszy wariant, punkt odniesienia dla pozostalych metod.
        obrazy = [self._wczytaj(s) for s in self.sciezki]

        stitcher = cv2.Stitcher_create(self._tryb)
        # status: 0 = OK, inne kody tlumaczy slownik KODY na gorze pliku
        status, wynik = stitcher.stitch(obrazy)
        dt = time.time() - t0

        self.ostatni_czas = dt
        self.historia.append((len(self.sciezki), dt, status == cv2.Stitcher_OK))

        if status != cv2.Stitcher_OK:
            log.warning("Skladanie nieudane przy %d zdjeciach: %s",
                        len(self.sciezki), KODY.get(status, f"kod {status}"))
            return False

        self.mozaika = wynik
        log.info("Mozaika %dx%d z %d zdjec (%.1f s)",
                 wynik.shape[1], wynik.shape[0], len(self.sciezki), dt)
        return True

    def raport_czasu(self):
        """Jak rosnie czas przeliczenia wraz z liczba zdjec.

        To jest pomiar, ktory uzasadnia (albo nie) przejscie na wariant
        przyrostowy z modulu detail.
        """
        return [{"zdjec": n, "czas_s": round(t, 2), "ok": ok}
                for n, t, ok in self.historia]
