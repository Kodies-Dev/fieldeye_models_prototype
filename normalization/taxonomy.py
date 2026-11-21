"""Slownik gatunkow: mapowanie nazw ze zbiorow zrodlowych na wspolne klucze.

Problem, ktory rozwiazuje: ten sam gatunek wystepuje w zbiorach pod roznymi
nazwami. Zoltlica to "GP" u Sudarsa, "weed_3" w WeedsGalore i "GASPA_V1"
w kodach EPPO z faza. Bez ujednolicenia model dostaje trzy osobne klasy, kazda
z ulamkiem danych.

Nierozpoznane nazwy NIE sa cicho pomijane - trafiaja do raportu, zeby dalo
sie je dopisac do taxonomy.yaml przed treningiem.

Jak dziala:
    taxonomy.yaml  - dane: kazdy gatunek ma klucz (np. zea_mays), role (uprawa/chwast)
                     i aliasy (np. maize, corn, ZEAMX); osobno wartosci pikseli w maskach
    Taxonomy       - wczytuje YAML i odpowiada na pytania:
                     klucz("maize") -> "zea_mays"        (adaptery: nazwa z XML -> gatunek)
                     z_maski("weedsgalore", 4) -> ...    (adaptery: piksel maski -> gatunek)
                     etykieta("zea_mays", "chwasty") -> None  (steps.py: gatunek -> klasa YOLO)
                     raport_nieznanych()                 (build.py: do build_info.json)

Uzywany w:
    build.py zbuduj() - jedna Taxonomy na cala budowe, przekazywana do adapterow i krokow
"""

import logging
import re
from collections import Counter
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

DOMYSLNY_PLIK = Path(__file__).with_name("taxonomy.yaml")   # taxonomy.yaml obok tego pliku


class Taxonomy:
    """Slownik gatunkow wczytany z YAML-a.

    Po co: jedno miejsce, ktore tlumaczy nazwy ze wszystkich zbiorow na te same klucze
           i decyduje, co jest chwastem, a co uprawa.

    Uzywana w:
        build.py zbuduj() (tworzy), adapters.py (klucz, z_maski, faza),
        steps.py zastosuj_tryb_klas() (etykieta), testy normalizacji
    """

    def __init__(self, sciezka=None):
        """Wczytuje YAML i buduje slownik alias -> klucz gatunku.

        Argumenty:
            sciezka - wlasny plik YAML albo None = normalization/taxonomy.yaml

        Zwraca:
            nic; ustawia self.gatunki, self.maski, self._alias, self.nieznane

        Uzywana w:
            build.py zbuduj() - Taxonomy(taksonomia)
        """
        dane = yaml.safe_load(Path(sciezka or DOMYSLNY_PLIK).read_text(encoding="utf-8"))
        self.gatunki = dane["gatunki"]          # klucz -> {polska, rola, w_polsce, aliasy, zwin_do}
        self.maski = dane.get("maski", {})      # zbior -> {wartosc piksela: klucz}

        # alias -> klucz gatunku; klucz tez jest swoim aliasem
        # (przy powtorzonym aliasie wygrywa gatunek zapisany w YAML-u nizej)
        self._alias = {}
        for klucz, info in self.gatunki.items():
            for a in [klucz, info.get("polska", "")] + list(info.get("aliasy", [])):
                if a:
                    self._alias[self._norm(a)] = klucz

        self.nieznane = Counter()      # nazwy, ktorych nie udalo sie dopasowac

    @staticmethod
    def _norm(s):
        """Do porownywania: male litery, bez znakow innych niz litery i cyfry.

        Po co: "Zea mays", "zea_mays" i "ZEA-MAYS" maja dac to samo.

        Argumenty:
            s - dowolna nazwa, np. "Zea mays"

        Zwraca:
            str, np. "zeamays"

        Uzywana w:
            __init__() (budowa aliasow), klucz() (szukanie)
        """
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    def klucz(self, nazwa):
        """Nazwa ze zbioru -> klucz gatunku albo None (z zapisem do raportu).

        Radzi sobie z sufiksami faz wzrostu w stylu ZEAMX_V1 czy SOLNI_Vc:
        gdy cala nazwa nie pasuje, probuje czesci przed podkresleniem.

        Argumenty:
            nazwa - nazwa z etykiety zbioru, np. "weed", "maize", "ZEAMX_V1"

        Zwraca:
            str albo None, np. "zea_mays"; None = nieznana (trafia do self.nieznane)

        Uzywana w:
            adapters.py - pascal_voc() (Sudars, EWIS1) i wlasne(): nazwa z XML/YOLO -> gatunek
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
        self.nieznane[str(nazwa)] += 1          # zapamietaj do raportu (ile razy)
        return None

    def faza(self, nazwa):
        """Faza wzrostu z nazwy typu ZEAMX_V1 -> 'V1'. Brak -> None.

        Argumenty:
            nazwa - nazwa z etykiety, np. "ZEAMX_V1"

        Zwraca:
            str albo None, np. "V1"

        Uzywana w:
            adapters.py pascal_voc() - pole faza w LabeledObject (nazwy z faza, np. ZEAMX_V1)
        """
        czesci = str(nazwa).split("_", 1)
        return czesci[1] if len(czesci) == 2 and czesci[1] else None

    def z_maski(self, zbior, wartosc):
        """Wartosc piksela w masce -> klucz gatunku.

        Po co: WeedsGalore ma etykiety jako maski, gdzie numer
               piksela oznacza gatunek (np. WeedsGalore: 1 = kukurydza, 5 = inny chwast).

        Argumenty:
            zbior   - nazwa zbioru z sekcji maski: w YAML, np. "weedsgalore"
            wartosc - wartosc piksela maski, np. 4

        Zwraca:
            str albo None, np. z_maski("weedsgalore", 4) -> "galinsoga_parviflora";
            None = tlo albo nieznana wartosc

        Uzywana w:
            adapters.py - weedsgalore()
        """
        return self.maski.get(zbior, {}).get(int(wartosc))

    def rola(self, klucz):
        """Rola gatunku: "chwast", "uprawa" albo "nieznana".

        Argumenty:
            klucz - klucz gatunku, np. "zea_mays"

        Zwraca:
            str, np. "uprawa"

        Uzywana w:
            etykieta() (ten plik)
        """
        return self.gatunki.get(klucz, {}).get("rola", "nieznana")

    def po_zwinieciu(self, klucz):
        """Gatunek rzadki w Polsce -> klasa zbiorcza (zwin_do w YAML-u).

        Argumenty:
            klucz - klucz gatunku, np. "gossypium"

        Zwraca:
            str - klucz klasy zbiorczej albo ten sam klucz, gdy zwin_do nie ustawione

        Uzywana w:
            etykieta() w trybie "gatunki" (ten plik)
        """
        return self.gatunki.get(klucz, {}).get("zwin_do", klucz)

    def etykieta(self, klucz, tryb="binarnie"):
        """Klucz gatunku -> nazwa klasy w zbiorze treningowym.

        binarnie  - chwast / uprawa (zalecane na pierwsza iteracje)
        chwasty   - tylko chwasty, uprawy pomijane (None)  <- model koncowy
        gatunki   - pelna nazwa gatunku, rzadkie zwiniete do klasy zbiorczej

        Argumenty:
            klucz - klucz gatunku, np. "zea_mays"
            tryb  - "binarnie", "chwasty" albo "gatunki"

        Zwraca:
            str albo None, np. "chwast"; None = obiekt usuwany (staje sie tlem)

        Uzywana w:
            steps.py zastosuj_tryb_klas() - krok 3 w build.py zbuduj()
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
        """Lista (nazwa, ile razy) posortowana malejaco. Pusta = wszystko zmapowane.

        Zwraca:
            lista krotek, np. [("dandelionn", 3)] albo []

        Uzywana w:
            build.py zbuduj() - pole "nierozpoznane_etykiety" w build_info.json
            (sprawdza je final_model.sh)
        """
        return self.nieznane.most_common()

