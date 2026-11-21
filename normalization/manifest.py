"""Manifest - wspolny format posredni miedzy zbiorami a treningiem.

Po co osobny format zamiast konwersji kazdy-zbior-na-YOLO:

  4 zbiory x wlasny konwerter = 4 miejsca, gdzie cos moze sie rozjechac,
  i 4 miejsca do poprawienia przy kazdej zmianie.

  Z manifestem: kazdy zbior ma JEDEN adapter (zrodlo -> manifest), a eksport
  do YOLO jest napisany RAZ. Dodanie piatego zbioru to jeden nowy adapter.

Dodatkowo manifest pozwala robic rzeczy niemozliwe na plikach .txt z etykietami:
filtrowac po pulapie, wykrywac duplikaty, liczyc rozklad klas, dzielic
deterministycznie.

Co jest w pliku:
    LabeledObject - jedna roslina (klasa + ramka w pikselach)
    Record        - jedno zdjecie albo kafelek: sciezka, rozmiar, zrodlo, grupa, split, obiekty
    zapisz()      - lista Record -> plik .jsonl   (build.py: manifest.jsonl)
    wczytaj()     - plik .jsonl -> lista Record   (testy)
    statystyki()  - liczby do build_info.json      (build.py)

Przeplyw: adapters.py tworzy Record -> steps.py je zmienia (podzial, skala, kafelki)
          -> export.py zapisuje z nich format YOLO.

Format: JSONL - jeden wiersz JSON na zdjecie. Czytelny, dopisywalny,
nie trzeba wczytywac calosci do pamieci.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LabeledObject:
    """Jedna roslina na zdjeciu: klasa i ramka.

    Po co: wspolny opis obiektu bez wzgledu na to, czy zbior mial XML, maske czy YOLO.

    Pola:
        klasa          - klucz z taksonomii, np. "chwast_inny", "zea_mays"
                         (po trybie "chwasty": "chwast")
        xyxy           - ramka w pikselach zdjecia [x0, y0, x1, y1], np. [10, 10, 40, 40]
        zrodlowa_nazwa - nazwa z oryginalnego zbioru, np. "weed" - do diagnostyki
        faza           - faza wzrostu, gdy nazwa ja podaje (np. ZEAMX_V1), inaczej None
        pole_maski     - liczba pikseli obiektu, gdy ramka pochodzi z maski, inaczej None

    Uzywana w:
        adapters.py (tworzenie obiektow), steps.py (kafelkowanie - ramki przyciete do kafelka),
        Record.obiekty, export.py (zapis etykiet YOLO)
    """
    klasa: str                       # klucz gatunku z taksonomii
    xyxy: list                       # ramka w pikselach BEZWZGLEDNYCH
    zrodlowa_nazwa: str = None       # jak nazywal sie w zbiorze - do diagnostyki
    faza: str = None                 # faza wzrostu, gdy zbior ja podaje
    pole_maski: int = None           # liczba pikseli, gdy obiekt pochodzi z maski


@dataclass
class Record:
    """Jedno zdjecie (albo kafelek) wraz ze wszystkim, co o nim wiemy.

    Po co: wszystkie kroki normalizacji pracuja na liscie Record, a nie na plikach -
           dzieki temu nie musza wiedziec, z jakiego zbioru pochodzi zdjecie.

    Pola (najwazniejsze):
        id      - nazwa jednoznaczna w calym zbiorze, np. "ewis1__Loew_20220512_Maize_010"
        obraz   - sciezka do zdjecia (WeedsGalore: prefiks bez _R.png)
        szer, wys - rozmiar w px (po skali, jesli byla), np. 400, 300
        zrodlo  - "weedsgalore" / "sudars" / "ewis1" / ...
        grupa   - klucz podzialu: cala grupa trafia do jednej czesci (opis nizej)
        domena  - "uav" (dron) albo "ground" (z ziemi)
        split   - "train" / "val" / "test" - wypelnia adapter (oficjalny podzial) albo steps.podziel()
        kafelek - [x, y] lewego gornego rogu, gdy rekord jest kafelkiem, inaczej None
        skala   - wspolczynnik przeskalowania, np. 0.47 dla EWIS1
        obiekty - lista LabeledObject

    Uzywana w:
        adapters.py (tworzy), steps.py (zmienia), export.py (zapisuje YOLO),
        zapisz() / wczytaj() / statystyki() (ten plik)
    """

    id: str                          # jednoznaczny w calym zbiorze laczonym
    obraz: str                       # sciezka (dla WeedsGalore: prefiks bez _R.png)
    szer: int
    wys: int
    zrodlo: str                      # weedsgalore / sudars / ewis1 / wlasne

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
    roslinnosci traktuje jako tlo. Mieszanie takich zbiorow
    z wyczerpujacymi daje modelowi sprzeczny sygnal - ta sama roslina raz
    jest chwastem, raz tlem.
    """

    split: str = None                # wypelniane przez kroki.podziel()
    split_oficjalny: bool = False    # True = podzial pochodzi od autorow zbioru
    kafelek: list = None             # [x, y] poczatku kafelka, gdy rekord jest kafelkiem
    skala: float = 1.0               # zdjecie przeskalowane (np. do GSD drona) - szer/wys/ramki juz w tej skali
    obiekty: list = field(default_factory=list)   # nowa pusta lista dla kazdego rekordu (nie wspolna)

    def slownik(self):
        """Zamienia rekord na zwykly slownik (razem z obiektami) - gotowy do zapisu jako JSON.

        Po co: json.dumps nie umie zapisac obiektow klasy Record/LabeledObject, umie slowniki.

        Argumenty:
            brak (self - ten rekord)

        Zwraca:
            dict, np. {"id": "ewis1__...", "szer": 400, ..., "obiekty": [{"klasa": "chwast", "xyxy": [...]}]}

        Uzywana w:
            zapisz() (ten plik) - kazdy rekord -> jedna linia JSONL
        """
        d = asdict(self)
        # obiekty moga byc juz slownikami - wtedy zostaja bez zmian
        d["obiekty"] = [asdict(o) if not isinstance(o, dict) else o for o in self.obiekty]
        return d

    @classmethod
    def ze_slownika(cls, d):
        """Odwrotnosc slownik(): slownik z JSON -> Record z obiektami LabeledObject.

        Po co: odczyt manifestu z pliku z powrotem do obiektow, na ktorych pracuja kroki.

        Argumenty:
            d - slownik z jednej linii JSONL, np. {"id": "x", "obraz": "...", "obiekty": [...]}

        Zwraca:
            Record

        Uzywana w:
            wczytaj() (ten plik)
        """
        d = dict(d)                      # kopia - nie zmieniamy slownika z zewnatrz
        d["obiekty"] = [LabeledObject(**o) if not isinstance(o, LabeledObject) else o
                        for o in d.get("obiekty", [])]
        return cls(**d)                  # cls = Record; **d rozpakowuje slownik na pola

    @property
    def liczba_obiektow(self):
        """Ile roslin (ramek) jest na zdjeciu, np. 12.

        Po co: wygodny skrot do logow i statystyk; @property = uzywa sie bez nawiasow: r.liczba_obiektow

        Uzywana w:
            build.py zbuduj() (log po adapterze), statystyki() (ten plik)
        """
        return len(self.obiekty)


def zapisz(rekordy, sciezka):
    """Zapisuje liste rekordow do pliku JSONL - jeden rekord na linie.

    Po co: manifest na dysku pozwala po budowie sprawdzic kazde zdjecie i kafelek
           (skad pochodzi, jaki ma split, jakie ramki) bez zagladania w pliki YOLO.

    Argumenty:
        rekordy - lista Record (albo gotowych slownikow)
        sciezka - plik wyniku, np. "zbior_lotnicze/manifest.jsonl"

    Zwraca:
        Path - sciezka zapisanego pliku

    Uzywana w:
        build.py zbuduj() - manifest_przed_kafelkami.jsonl i manifest.jsonl
    """
    sciezka = Path(sciezka)
    sciezka.parent.mkdir(parents=True, exist_ok=True)
    with open(sciezka, "w", encoding="utf-8") as f:
        for r in rekordy:
            d = r.slownik() if isinstance(r, Record) else r
            f.write(json.dumps(d, ensure_ascii=False) + "\n")   # ensure_ascii=False: polskie znaki bez \u...
    return sciezka


def wczytaj(sciezka):
    """Wczytuje manifest JSONL z powrotem do listy Record.

    Po co: odczyt zapisanego manifestu (np. do analizy); pomija puste linie.

    Argumenty:
        sciezka - plik .jsonl, np. "zbior_lotnicze/manifest.jsonl"

    Zwraca:
        lista Record

    Uzywana w:
        tests/test_normalization.py - sprawdzenie, ze zapis i odczyt daja to samo
    """
    with open(sciezka, encoding="utf-8") as f:
        return [Record.ze_slownika(json.loads(l)) for l in f if l.strip()]


def statystyki(rekordy):
    """Podsumowanie manifestu - do raportu i do kontroli po kazdym kroku.

    Po co: liczby, ktore trafiaja do build_info.json i na terminal po budowie.

    Argumenty:
        rekordy - lista Record

    Zwraca:
        dict, np. {"zdjec": 360, "obiektow": 2150, "zdjec_bez_obiektow": 40,
                   "zrodla": {"ewis1": 360}, "domeny": {"uav": 360}, "klasy": {"chwast": 2150},
                   "splity": {"train": 228, "test": 132}, "grup": 12}

    Uzywana w:
        build.py zbuduj() - pole "statystyki" w build_info.json; main() - wypis na terminal
    """
    from collections import Counter      # Counter: liczy, ile razy wystepuje kazda wartosc
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
        "klasy": dict(klasy.most_common()),          # od najczestszej klasy
        "splity": dict(splity),
        "grup": len({r.grupa for r in rekordy if r.grupa}),   # zbior (set) = liczba roznych grup
    }
