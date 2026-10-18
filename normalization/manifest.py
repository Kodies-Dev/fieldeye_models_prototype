"""Manifest - wspolny format posredni miedzy zbiorami a treningiem.

Po co osobny format zamiast konwersji kazdy-zbior-na-YOLO:

  4 zbiory x wlasny konwerter = 4 miejsca, gdzie cos moze sie rozjechac,
  i 4 miejsca do poprawienia przy kazdej zmianie.

  Z manifestem: kazdy zbior ma JEDEN adapter (zrodlo -> manifest), a eksport
  do YOLO jest napisany RAZ. Dodanie piatego zbioru to jeden nowy adapter.

Dodatkowo manifest pozwala robic rzeczy niemozliwe na plikach .txt z etykietami:
filtrowac po pulapie, wykrywac duplikaty, liczyc rozklad klas, dzielic
deterministycznie.

Format: JSONL - jeden wiersz JSON na zdjecie. Czytelny, dopisywalny,
nie trzeba wczytywac calosci do pamieci.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LabeledObject:
    """Jedna roslina na zdjeciu."""
    klasa: str                       # klucz gatunku z taksonomii
    xyxy: list                       # ramka w pikselach BEZWZGLEDNYCH
    zrodlowa_nazwa: str = None       # jak nazywal sie w zbiorze - do diagnostyki
    faza: str = None                 # faza wzrostu, gdy zbior ja podaje
    pole_maski: int = None           # liczba pikseli, gdy obiekt pochodzi z maski


@dataclass
class Record:
    """Jedno zdjecie wraz ze wszystkim, co o nim wiemy."""

    id: str                          # jednoznaczny w calym zbiorze laczonym
    obraz: str                       # sciezka (dla WeedsGalore: prefiks bez _R.png)
    szer: int
    wys: int
    zrodlo: str                      # weedsgalore / cofly / sudars / weedmaize / wlasne

    grupa: str = None
    """KLUCZ PODZIALU train/val/test. Najwazniejsze pole w calym manifescie.

    Wszystkie zdjecia z tej samej grupy trafiaja do tej samej czesci podzialu.
    Grupa to misja, dzien lotu albo poletko - NIGDY pojedyncze zdjecie.
    Powod: sasiednie kadry z jednego przelotu pokrywaja sie w 60-80%, wiec
    losowy podzial po zdjeciach wsadza ten sam fragment terenu do treningu
    i do testu naraz. mAP wychodzi wtedy zawyzone, a model w polu zawodzi.
    """

    domena: str = "ground"           # uav albo ground - decyduje o etapie treningu
    agl_m: float = None              # wysokosc lotu
    gsd_mm: float = None             # ile mm terenu na piksel
    phash: str = None                # skrot percepcyjny do deduplikacji
    wyczerpujace: bool = True
    """Czy zaadnotowano WSZYSTKIE chwasty w kadrze.

    False oznacza, ze zbior adnotuje tylko wybrane gatunki, a reszte
    roslinnosci traktuje jako tlo (tak robi CoFly). Mieszanie takich zbiorow
    z wyczerpujacymi daje modelowi sprzeczny sygnal - ta sama roslina raz
    jest chwastem, raz tlem.
    """

    split: str = None                # wypelniane przez kroki.podziel()
    split_oficjalny: bool = False    # True = podzial pochodzi od autorow zbioru
    kafelek: list = None             # [x, y] poczatku kafelka, gdy rekord jest kafelkiem
    skala: float = 1.0               # zdjecie przeskalowane (np. do GSD drona) - szer/wys/ramki juz w tej skali
    obiekty: list = field(default_factory=list)

    def slownik(self):
        d = asdict(self)
        d["obiekty"] = [asdict(o) if not isinstance(o, dict) else o for o in self.obiekty]
        return d

    @classmethod
    def ze_slownika(cls, d):
        d = dict(d)
        d["obiekty"] = [LabeledObject(**o) if not isinstance(o, LabeledObject) else o
                        for o in d.get("obiekty", [])]
        return cls(**d)

    @property
    def liczba_obiektow(self):
        return len(self.obiekty)


def zapisz(rekordy, sciezka):
    """Zapis manifestu do JSONL."""
    sciezka = Path(sciezka)
    sciezka.parent.mkdir(parents=True, exist_ok=True)
    with open(sciezka, "w", encoding="utf-8") as f:
        for r in rekordy:
            d = r.slownik() if isinstance(r, Record) else r
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return sciezka


def wczytaj(sciezka):
    """Odczyt manifestu z JSONL."""
    with open(sciezka, encoding="utf-8") as f:
        return [Record.ze_slownika(json.loads(l)) for l in f if l.strip()]


def statystyki(rekordy):
    """Podsumowanie manifestu - do raportu i do kontroli po kazdym kroku."""
    from collections import Counter
    zrodla = Counter(r.zrodlo for r in rekordy)
    domeny = Counter(r.domena for r in rekordy)
    klasy = Counter(o.klasa for r in rekordy for o in r.obiekty)
    splity = Counter(r.split for r in rekordy if r.split)
    puste = sum(1 for r in rekordy if not r.obiekty)
    return {
        "zdjec": len(rekordy),
        "obiektow": sum(r.liczba_obiektow for r in rekordy),
        "zdjec_bez_obiektow": puste,
        "zrodla": dict(zrodla),
        "domeny": dict(domeny),
        "klasy": dict(klasy.most_common()),
        "splity": dict(splity),
        "grup": len({r.grupa for r in rekordy if r.grupa}),
    }