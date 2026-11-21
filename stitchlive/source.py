"""Zrodla obrazow.

Dwa tryby:
  FolderSource  - obserwuje katalog w watku w tle, wrzuca nowe pliki do kolejki
  BatchSource  - wczytuje istniejacy katalog naraz, do testow bez symulacji lotu

Oba wrzucaja sciezki do kolejki (queue.Queue), z ktorej czyta petla w cli.py -
odczyt folderu nie blokuje skladania.
"""

from pathlib import Path
import logging
import threading
import time

log = logging.getLogger(__name__)

ROZSZERZENIA = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".tif", ".tiff"}


def lista_obrazow(katalog):
    """Posortowana lista plikow obrazow. Kolejnosc alfabetyczna odpowiada
    zwykle kolejnosci wykonania zdjec przez kapsule.

    Argumenty:
        katalog - folder zdjec

    Zwraca:
        lista Path (bez plikow ukrytych)

    Uzywana w:
        FolderSource, BatchSource (ten plik), cli.py cmd_info(), tests/test_stitch.py
    """
    # pliki ukryte (np. .tmp_ w trakcie kopiowania) sa pomijane
    return sorted(p for p in Path(katalog).iterdir()
                  if p.suffix in ROZSZERZENIA and not p.name.startswith("."))


class FolderSource:
    """Obserwuje katalog i wrzuca nowe pliki do kolejki.

    Dziala w osobnym watku, wiec odczyt z dysku nie blokuje przetwarzania.

    Uzywana w:
        cli.py cmd_watch() (python -m stitchlive.cli watch - skladanie w trakcie lotu)
    """

    def __init__(self, katalog, kolejka, interwal_s=1.0, od_poczatku=False,
                 stabilizacja_s=0.4):
        """Argumenty:
            katalog        - folder, do ktorego trafiaja zdjecia
            kolejka        - queue.Queue na sciezki
            interwal_s     - co ile sekund sprawdzac folder, np. 1.0
            od_poczatku    - True = wez tez pliki obecne przy starcie
            stabilizacja_s - odstep dwoch pomiarow rozmiaru pliku, np. 0.4
        """
        self.katalog = Path(katalog)
        self.kolejka = kolejka
        self.interwal = interwal_s
        self.stabilizacja = stabilizacja_s
        # nazwy plikow juz wrzuconych do kolejki - kazdy trafia tylko raz
        self.widziane = set()
        # flaga zatrzymania - bezpieczna do ustawiania z innego watku
        self._stop = threading.Event()
        self._watek = None

        if not od_poczatku:
            # pliki juz obecne przy starcie sa pomijane
            self.widziane = {p.name for p in lista_obrazow(self.katalog)}
            log.info("Pomijam %d plikow obecnych przy starcie", len(self.widziane))

    def start(self):
        """Uruchamia obserwacje w watku w tle (tworzy folder, jesli go nie ma)."""
        # Watek w tle. daemon=True: nie blokuje zamkniecia programu (np. po Ctrl+C).
        self.katalog.mkdir(parents=True, exist_ok=True)
        self._watek = threading.Thread(target=self._petla, daemon=True)
        self._watek.start()
        log.info("Obserwuje katalog: %s", self.katalog)

    def _petla(self):
        """Petla watku: co interwal nowe, zapisane do konca pliki -> kolejka (kazdy raz)."""
        while not self._stop.is_set():
            try:
                for p in lista_obrazow(self.katalog):
                    if p.name in self.widziane:
                        continue
                    # plik moze byc jeszcze zapisywany - czekamy az rozmiar
                    # przestanie sie zmieniac
                    if not self._plik_gotowy(p):
                        continue
                    self.widziane.add(p.name)
                    self.kolejka.put(p)
            # katalog mogl zostac chwilowo usuniety - probujemy w nastepnym obiegu
            except FileNotFoundError:
                pass
            time.sleep(self.interwal)

    def _plik_gotowy(self, p):
        """Czy plik jest zapisany do konca: rozmiar > 0 i taki sam po stabilizacja_s.

        Zwraca:
            bool
        """
        try:
            # Dwa odczyty rozmiaru w odstepie 'stabilizacja' sekund - jesli sie roznia,
            # plik jest wciaz zapisywany (np. kopiowany z drona).
            r1 = p.stat().st_size
            time.sleep(self.stabilizacja)
            return r1 > 0 and p.stat().st_size == r1
        except OSError:
            return False

    def stop(self):
        """Zatrzymuje watek (czeka najwyzej 2 s)."""
        self._stop.set()
        if self._watek:
            # czekamy na zakonczenie watku, ale nie dluzej niz 2 s
            self._watek.join(timeout=2.0)


class BatchSource:
    """Wczytuje caly katalog naraz. Do testow na materiale z wczesniejszego lotu.

    Uzywana w:
        cli.py cmd_batch(), cmd_demo()
    """

    def __init__(self, katalog, kolejka, opoznienie_s=0.0, limit=None):
        """Argumenty:
            katalog      - folder zdjec
            kolejka      - queue.Queue na sciezki
            opoznienie_s - odstep miedzy zdjeciami (udawanie lotu), 0 = bez
            limit        - najwyzej tyle zdjec albo None
        """
        self.sciezki = lista_obrazow(katalog)
        if limit:
            self.sciezki = self.sciezki[:limit]
        self.kolejka = kolejka
        self.opoznienie = opoznienie_s
        self._watek = None

    def start(self):
        """Uruchamia wrzucanie zdjec do kolejki w watku w tle."""
        self._watek = threading.Thread(target=self._petla, daemon=True)
        self._watek.start()
        log.info("Tryb wsadowy: %d obrazow", len(self.sciezki))

    def _petla(self):
        """Wszystkie zdjecia -> kolejka, na koncu None (sygnal konca dla petli w cli.py)."""
        for p in self.sciezki:
            self.kolejka.put(p)
            if self.opoznienie:
                time.sleep(self.opoznienie)
        # None w kolejce = koniec zdjec; petla glowna w cli.py konczy wtedy prace
        self.kolejka.put(None)          # sygnal konca

    def stop(self):
        """Nic nie robi - dla zgodnosci z FolderSource."""
        pass
