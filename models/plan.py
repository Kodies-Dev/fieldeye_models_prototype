"""Plan treningu trzyetapowego - konfiguracja, bez zaleznosci od ultralytics.

Dlaczego trzy etapy, a nie jedna wspolna pula:

  Danych naziemnych (Sudars, WeedMaize) jest kilkukrotnie wiecej niz lotniczych
  (WeedsGalore, CoFly). W jednej puli zdominowalyby trening i model
  zoptymalizowalby sie pod perspektywe, ktorej nigdy nie zobaczy w locie.

  etap 1  naziemne   - ogolna charakterystyka roslin, od wag COCO, duzo epok
  etap 2  lotnicze   - perspektywa z gory, od wag etapu 1, lr0 0.01 (patrz nizej)
  etap 3  wlasne     - nasza kamera, gleba, pole; najnizsze lr, najmniej epok

Kazdy kolejny etap startuje z wag poprzedniego.

Pierwotnie etap 2 mial lr0 0.001 ("uczy sie wolniej, zeby nie zapomniec").
Ablacja na WeedsGalore (104 obrazy, 100 epok) to obalila: lr0 0.001 dalo
mAP50 0.572, a lr0 0.01 - 0.624 (test). Na tak malym zbiorze nizsze tempo
nie zdazylo dostosowac modelu. Etap 3 (dane wlasne) zostaje przy 0.0005 -
NIESPRAWDZONE, zmierzyc, gdy beda dane.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class Stage:
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
        return asdict(self)


@dataclass
class TrainingPlan:
    """Caly plan. model_bazowy to punkt startu etapu 1."""
    model_bazowy: str = "yolov8n.pt"
    etapy: list = field(default_factory=list)
    batch: int = 16
    cierpliwosc: int = 20          # ile epok bez poprawy do zatrzymania
    urzadzenie: str = None         # None = auto, "cpu", "0"
    wyjscie: str = "modele_out"
    seed: int = 0
    # Jawny optymalizator jest potrzebny, zeby lr0 z planu bylo uzywane.
    # Wartosci lr0 w planie domyslnym (0.01 / 0.001 / 0.0005) sa dobrane pod SGD.
    optymalizator: str = "SGD"

    @staticmethod
    def domyslny(dane_naziemne=None, dane_lotnicze=None, dane_wlasne=None,
                 model_bazowy="yolov8n.pt", wejscie=640):
        """Plan z zalecanymi parametrami. Etapy bez danych sa pomijane."""
        etapy = []
        if dane_naziemne:
            etapy.append(Stage(
                "1-naziemne", dane_naziemne, epoki=100, lr0=0.01, wejscie=wejscie,
                opis="ogolna charakterystyka roslin; start z wag COCO"))
        if dane_lotnicze:
            etapy.append(Stage(
                "2-lotnicze", dane_lotnicze, epoki=100, lr0=0.01, wejscie=wejscie,
                opis="przystosowanie do perspektywy z nadiru"))
        if dane_wlasne:
            etapy.append(Stage(
                "3-wlasne", dane_wlasne, epoki=30, lr0=0.0005, wejscie=wejscie,
                opis="nasza kamera, gleba i uprawa; etap obowiazkowy"))
        if not etapy:
            raise ValueError("podaj dane co najmniej jednego etapu")
        return TrainingPlan(model_bazowy=model_bazowy, etapy=etapy)

    @classmethod
    def wczytaj(cls, sciezka):
        tekst = Path(sciezka).read_text(encoding="utf-8")
        if Path(sciezka).suffix in (".yaml", ".yml"):
            if yaml is None:
                raise RuntimeError("brak PyYAML - uzyj pliku .json")
            dane = yaml.safe_load(tekst) or {}
        else:
            import json
            dane = json.loads(tekst)
        etapy = [Stage(**e) for e in dane.pop("etapy", [])]
        return cls(etapy=etapy, **dane)

    def zapisz(self, sciezka):
        import json
        p = Path(sciezka)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.slownik(), indent=2, ensure_ascii=False),
                     encoding="utf-8")
        return p

    def slownik(self):
        d = asdict(self)
        d["etapy"] = [e.slownik() if hasattr(e, "slownik") else e for e in self.etapy]
        return d


def katalog_etykiet(katalog_obrazow):
    """images/train -> labels/train (konwencja YOLO).

    Zamieniane jest ostatnie wystapienie skladnika "images" w sciezce.
    Dziala na czesciach sciezki, nie na tekscie - zamiana "/images/" na
    "/labels/" w napisie nie dzialala na Windowsie, gdzie separatorem jest
    backslash, i kazdy obraz wygladal na pozbawiony etykiet.
    """
    czesci = list(katalog_obrazow.parts)
    for i in range(len(czesci) - 1, -1, -1):
        if czesci[i] == "images":
            czesci[i] = "labels"
            return type(katalog_obrazow)(*czesci)
    return katalog_obrazow           # nie standardowy uklad - etykiety obok obrazow


def sprawdz_dane(sciezka_yaml):
    """Kontrola data.yaml PRZED treningiem - tansze niz odkrycie bledu po godzinie.

    Zwraca (ok, lista_problemow, informacje).
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
        nazwy = [nazwy[k] for k in sorted(nazwy, key=lambda x: int(x))]
    if not nazwy:
        problemy.append("data.yaml nie ma listy klas (names)")
    info["klasy"] = nazwy or []

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
        # Ultralytics dopuszcza liste katalogow, np. train: [WeedsGalore, CoFly]
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
        for b in czesci[i + 1:]:
            wspolne = nazwy_czesci[a] & nazwy_czesci[b]
            if wspolne:
                problemy.append(f"PRZECIEK: {len(wspolne)} tych samych plikow "
                                f"w '{a}' i '{b}' (np. {sorted(wspolne)[0]})")

    return not problemy, problemy, info
