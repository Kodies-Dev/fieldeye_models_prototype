"""Slownik gatunkow: mapowanie nazw ze zbiorow zrodlowych na wspolne klucze.

Problem, ktory rozwiazuje: ten sam gatunek wystepuje w zbiorach pod roznymi
nazwami. Zoltlica to "GP" u Sudarsa, "weed_3" w WeedsGalore i "GASPA_V1"
w WeedMaize. Bez ujednolicenia model dostaje trzy osobne klasy, kazda
z ulamkiem danych.

Nierozpoznane nazwy NIE sa cicho pomijane - trafiaja do raportu, zeby dalo
sie je dopisac do taxonomy.yaml przed treningiem.
"""

import logging
import re
from collections import Counter
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

DOMYSLNY_PLIK = Path(__file__).with_name("taxonomy.yaml")


class Taxonomy:
    """Slownik gatunkow wczytany z YAML-a."""

    def __init__(self, sciezka=None):
        dane = yaml.safe_load(Path(sciezka or DOMYSLNY_PLIK).read_text(encoding="utf-8"))
        self.gatunki = dane["gatunki"]
        self.maski = dane.get("maski", {})

        # alias -> klucz gatunku; klucz tez jest swoim aliasem
        self._alias = {}
        for klucz, info in self.gatunki.items():
            for a in [klucz, info.get("polska", "")] + list(info.get("aliasy", [])):
                if a:
                    self._alias[self._norm(a)] = klucz

        self.nieznane = Counter()      # nazwy, ktorych nie udalo sie dopasowac

    @staticmethod
    def _norm(s):
        """Do porownywania: male litery, bez znakow innych niz litery i cyfry."""
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    def klucz(self, nazwa):
        """Nazwa ze zbioru -> klucz gatunku albo None (z zapisem do raportu).

        Radzi sobie z sufiksami faz wzrostu w stylu ZEAMX_V1 czy SOLNI_Vc:
        gdy cala nazwa nie pasuje, probuje czesci przed podkresleniem.
        """
        if nazwa is None:
            return None
        n = self._norm(nazwa)
        if n in self._alias:
            return self._alias[n]
        # kod EPPO z faza wzrostu: ZEAMX_V1 -> ZEAMX
        przed = self._norm(str(nazwa).split("_")[0])
        if przed and przed in self._alias:
            return self._alias[przed]
        self.nieznane[str(nazwa)] += 1
        return None

    def faza(self, nazwa):
        """Faza wzrostu z nazwy typu ZEAMX_V1 -> 'V1'. Brak -> None."""
        czesci = str(nazwa).split("_", 1)
        return czesci[1] if len(czesci) == 2 and czesci[1] else None

    def z_maski(self, zbior, wartosc):
        """Wartosc piksela w masce -> klucz gatunku."""
        return self.maski.get(zbior, {}).get(int(wartosc))

    def rola(self, klucz):
        return self.gatunki.get(klucz, {}).get("rola", "nieznana")

    def po_zwinieciu(self, klucz):
        """Gatunek rzadki w Polsce -> klasa zbiorcza (zwin_do w YAML-u)."""
        return self.gatunki.get(klucz, {}).get("zwin_do", klucz)

    def etykieta(self, klucz, tryb="binarnie"):
        """Klucz gatunku -> nazwa klasy w zbiorze treningowym.

        binarnie  - chwast / uprawa (zalecane na pierwsza iteracje)
        chwasty   - tylko chwasty, uprawy pomijane (None)
        gatunki   - pelna nazwa gatunku, rzadkie zwiniete do klasy zbiorczej
        """
        rola = self.rola(klucz)
        if tryb == "binarnie":
            return rola if rola in ("chwast", "uprawa") else None
        if tryb == "chwasty":
            return "chwast" if rola == "chwast" else None
        if tryb == "gatunki":
            return self.po_zwinieciu(klucz)
        raise ValueError(f"nieznany tryb klas: {tryb}")

    def raport_nieznanych(self):
        """Lista (nazwa, ile razy) posortowana malejaco. Pusta = wszystko zmapowane."""
        return self.nieznane.most_common()

    def podsumowanie(self):
        role = Counter(self.rola(k) for k in self.gatunki)
        return {"gatunkow": len(self.gatunki), "aliasow": len(self._alias), **role}
