"""Plan treningu - konfiguracja, bez zaleznosci od ultralytics.

Plan to lista etapow; kazdy kolejny etap startuje z wag poprzedniego.
Model koncowy (scripts/final_model.sh) uzywa JEDNEGO etapu na polaczonych danych
(WeedsGalore z nadprobkowaniem + Sudars + EWIS1, jeden data.yaml z nadprobkuj()).
Sudars stanowi tam wiekszosc treningu (1438 z 1866 kafelkow), ale walidacja - czyli
wybor najlepszej epoki - i test sa tylko z WeedsGalore, wiec model jest wybierany
pod perspektywe drona. Plan czytany z pliku plan_dane_ewis1_bt.json:
    {"model_bazowy": "yolov8n.pt", "batch": 16, "cierpliwosc": 30, "seed": 0,
     "etapy": [{"nazwa": "dane-ewis1-bt", "dane": "zbior_lotnicze/dataset/data_dane_ewis1_bt.yaml",
                "epoki": 100, "lr0": 0.01, "wejscie": 1280}]}

Co jest w pliku:
    Stage           - jeden etap: dane, epoki, lr0, wejscie, augmentacje
    TrainingPlan    - caly plan: model startowy, etapy, batch, cierpliwosc, seed, optymalizator
    katalog_etykiet - images/train -> labels/train
    sprawdz_dane()  - kontrola data.yaml przed treningiem (brak etykiet, puste foldery, przeciek)
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None                    # bez PyYAML dzialaja tylko plany .json


@dataclass
class Stage:
    """Jeden etap treningu.

    Po co: opis jednego przebiegu ultralytics - na jakich danych, ile epok, jakie lr.

    Pola:
        nazwa             - nazwa etapu = nazwa folderu wynikow, np. "dane-ewis1-bt"
        dane              - sciezka do data.yaml, np. "zbior_lotnicze/dataset/data_dane_ewis1_bt.yaml"
        epoki             - maks. liczba epok, np. 100 (early stopping moze skonczyc wczesniej)
        lr0               - poczatkowe tempo uczenia (learning rate), np. 0.01
        wejscie           - bok obrazu na wejsciu sieci w px, np. 1280
        zamrozone_warstwy - ile pierwszych warstw sieci nie uczyc, 0 = wszystkie sie ucza
        opis              - opis dla ludzi
        augmentacje       - dodatkowe opcje augmentacji ultralytics, np. {"degrees": 15, "flipud": 0.5}

    Uzywana w:
        TrainingPlan.etapy, training.py trenuj() (etap po etapie)
    """
    nazwa: str
    dane: str                      # sciezka do data.yaml
    epoki: int
    lr0: float
    wejscie: int = 640
    zamrozone_warstwy: int = 0     # ile pierwszych warstw zamrozic
    opis: str = ""
    # Augmentacje przekazywane wprost do ultralytics, np.
    #   {"degrees": 15, "flipud": 0.5}
    # Dla zdjec z drona z nadiru: obrot (dron leci pod roznymi kursami)
    # i odbicie pionowe (z gory zdjecie nie ma "gory" ani "dolu").
    augmentacje: dict = field(default_factory=dict)

    def slownik(self):
        """Etap jako slownik (do zapisu w JSON i do historii treningu).

        Zwraca:
            dict, np. {"nazwa": "dane-ewis1-bt", "dane": "...", "epoki": 100, "lr0": 0.01, ...}

        Uzywana w:
            TrainingPlan.slownik() (ten plik), training.py (historia etapow)
        """
        return asdict(self)


@dataclass
class TrainingPlan:
    """Caly plan. model_bazowy to punkt startu pierwszego etapu.

    Po co: wszystkie ustawienia treningu w jednym miejscu, zapisywalne do pliku
           (zeby trening dalo sie powtorzyc z tymi samymi parametrami).

    Pola:
        model_bazowy  - wagi startowe, np. "yolov8n.pt" (YOLOv8 nano, wagi COCO)
        etapy         - lista Stage
        batch         - ile obrazow naraz, np. 16
        cierpliwosc   - early stopping: stop po tylu epokach bez poprawy na walidacji, np. 30
        urzadzenie    - None = auto (CUDA -> MPS -> CPU), "cpu", "0" (pierwsza karta)
        wyjscie       - folder wynikow, np. "modele_dane_ewis1_bt" (ustawia cli z -o)
        seed          - ziarno losowania (powtarzalnosc), np. 0
        optymalizator - "SGD"; jawnie, bo inaczej ultralytics sam dobiera lr i ignoruje lr0

    Uzywana w:
        models/cli.py cmd_trenuj() (wczytaj), training.py trenuj()
    """
    model_bazowy: str = "yolov8n.pt"
    etapy: list = field(default_factory=list)
    batch: int = 16
    cierpliwosc: int = 20          # ile epok bez poprawy do zatrzymania
    urzadzenie: str = None         # None = auto, "cpu", "0"
    wyjscie: str = "modele_out"
    seed: int = 0
    # Jawny optymalizator jest potrzebny, zeby lr0 z planu bylo uzywane.
    optymalizator: str = "SGD"

    @classmethod
    def wczytaj(cls, sciezka):
        """Plan z pliku .json albo .yaml.

        Argumenty:
            sciezka - plik planu, np. "plan_dane_ewis1_bt.json"

        Zwraca:
            TrainingPlan (etapy zamienione na obiekty Stage; brakujace pola - wartosci domyslne)

        Uzywana w:
            models/cli.py cmd_trenuj() (--plan) - scripts/final_model.sh, krok 2
        """
        tekst = Path(sciezka).read_text(encoding="utf-8")
        if Path(sciezka).suffix in (".yaml", ".yml"):
            if yaml is None:
                raise RuntimeError("brak PyYAML - uzyj pliku .json")
            dane = yaml.safe_load(tekst) or {}
        else:
            import json
            dane = json.loads(tekst)
        etapy = [Stage(**e) for e in dane.pop("etapy", [])]   # slowniki -> Stage
        return cls(etapy=etapy, **dane)                       # reszta pol wprost do TrainingPlan

    def zapisz(self, sciezka):
        """Zapisuje plan do JSON.

        Argumenty:
            sciezka - plik wyniku, np. "modele_x/plan.json"

        Zwraca:
            Path zapisanego pliku

        Uzywana w:
            training.py - kopia planu obok modelu (wiadomo, jak go uczono)
        """
        import json
        p = Path(sciezka)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.slownik(), indent=2, ensure_ascii=False),
                     encoding="utf-8")
        return p

    def slownik(self):
        """Plan jako slownik (etapy tez jako slowniki).

        Zwraca:
            dict, np. {"model_bazowy": "yolov8n.pt", "etapy": [{...}], "batch": 16, ...}

        Uzywana w:
            zapisz() (ten plik)
        """
        d = asdict(self)
        d["etapy"] = [e.slownik() if hasattr(e, "slownik") else e for e in self.etapy]
        return d


def katalog_etykiet(katalog_obrazow):
    """images/train -> labels/train (konwencja YOLO).

    Zamieniane jest ostatnie wystapienie skladnika "images" w sciezce.
    Dziala na czesciach sciezki, nie na tekscie - zamiana "/images/" na
    "/labels/" w napisie nie dzialala na Windowsie, gdzie separatorem jest
    backslash, i kazdy obraz wygladal na pozbawiony etykiet.

    Argumenty:
        katalog_obrazow - Path, np. Path("zbior_lotnicze/dataset/images/train")

    Zwraca:
        Path, np. Path("zbior_lotnicze/dataset/labels/train"); bez "images" - ta sama sciezka

    Uzywana w:
        sprawdz_dane() (ten plik), season.py, evaluation.py,
        mission/sim_flight.py, mission/drone_sim.py, app.py
    """
    czesci = list(katalog_obrazow.parts)
    for i in range(len(czesci) - 1, -1, -1):          # od konca sciezki
        if czesci[i] == "images":
            czesci[i] = "labels"
            return type(katalog_obrazow)(*czesci)
    return katalog_obrazow           # nie standardowy uklad - etykiety obok obrazow


def sprawdz_dane(sciezka_yaml):
    """Kontrola data.yaml PRZED treningiem - tansze niz odkrycie bledu po godzinie.

    Zwraca (ok, lista_problemow, informacje).

    Sprawdza: czy sa klasy, czy foldery train/val (test opcjonalny) istnieja i maja obrazy,
    czy kazdy obraz ma plik etykiet, czy te same pliki nie sa w dwoch czesciach (przeciek).

    Argumenty:
        sciezka_yaml - data.yaml, np. "zbior_lotnicze/dataset/data_dane_ewis1_bt.yaml"

    Zwraca:
        (ok, problemy, info), np. (True, [], {"klasy": ["chwast"],
        "train": {"obrazow": 1866, "bez_etykiet": 0}, "val": {...}, "test": {...}})

    Uzywana w:
        training.py trenuj() - przed kazdym etapem (blad = koniec bez treningu),
        models/cli.py (komenda sprawdz), tests/test_season.py
    """
    problemy, info = [], {}
    p = Path(sciezka_yaml)
    if not p.exists():
        return False, [f"brak pliku {p}"], info

    if yaml is None:
        return True, ["brak PyYAML - pominieto kontrole data.yaml"], info
    dane = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    nazwy = dane.get("names")
    if isinstance(nazwy, dict):
        nazwy = [nazwy[k] for k in sorted(nazwy, key=lambda x: int(x))]   # {0: "chwast"} -> ["chwast"]
    if not nazwy:
        problemy.append("data.yaml nie ma listy klas (names)")
    info["klasy"] = nazwy or []

    # folder bazowy: pole "path" z data.yaml (albo folder samego pliku)
    baza = Path(dane.get("path", p.parent))
    if not baza.is_absolute():
        baza = (p.parent / baza).resolve()

    katalogi_czesci = {}
    for czesc in ("train", "val", "test"):
        wzgl = dane.get(czesc)
        if not wzgl:
            if czesc != "test":
                problemy.append(f"brak sekcji '{czesc}' w data.yaml")
            continue
        # Ultralytics dopuszcza liste katalogow, np. train: [WeedsGalore, Sudars, EWIS1]
        lista = wzgl if isinstance(wzgl, list) else [wzgl]
        razem_obr, razem_bez = 0, 0
        katalogi_czesci[czesc] = []
        for w in lista:
            kat = Path(w) if Path(w).is_absolute() else baza / w
            if not kat.exists():
                problemy.append(f"katalog '{czesc}' nie istnieje: {kat}")
                continue
            katalogi_czesci[czesc].append(kat)
            obrazy = [x for x in kat.iterdir() if x.suffix.lower() in (".jpg", ".jpeg", ".png")]
            etykiety = katalog_etykiet(kat)
            bez_etykiet = [x.name for x in obrazy
                           if not (etykiety / (x.stem + ".txt")).exists()]
            razem_obr += len(obrazy)
            razem_bez += len(bez_etykiet)
            if not obrazy:
                problemy.append(f"'{czesc}' nie zawiera zadnych obrazow: {kat}")
            if bez_etykiet:
                problemy.append(f"'{czesc}': {len(bez_etykiet)} obrazow bez pliku etykiet "
                                f"(np. {bez_etykiet[0]})")
        info[czesc] = {"obrazow": razem_obr, "bez_etykiet": razem_bez}

    # czy obrazy nie powtarzaja sie miedzy czesciami - przeciek widoczny wprost
    nazwy_czesci = {c: {x.stem for k in kats for x in k.iterdir()}
                    for c, kats in katalogi_czesci.items()}
    czesci = list(nazwy_czesci)
    for i, a in enumerate(czesci):
        for b in czesci[i + 1:]:                       # kazda para czesci raz
            wspolne = nazwy_czesci[a] & nazwy_czesci[b]   # & = czesc wspolna zbiorow
            if wspolne:
                problemy.append(f"PRZECIEK: {len(wspolne)} tych samych plikow "
                                f"w '{a}' i '{b}' (np. {sorted(wspolne)[0]})")

    return not problemy, problemy, info
