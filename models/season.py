"""Sezonowosc: zeby model dzialal nie tylko w czerwcu, ale tez na najmlodszych siewkach.

Problem (zmierzony): model wg_sudars znajduje ok. 82% chwastow 6 czerwca i tylko
27-28% 25 maja. Majowe kafelki maja ok. 41 chwastow, czerwcowe ok. 90 - wiekszosc
przykladow, na ktorych model sie uczy, pochodzi wiec z pozniejszej fazy.

Narzedzia:
  nadprobkuj()          - kafelki wczesnych dni czesciej w treningu (oversampling),
                          przez kopie z inna nazwa: dziala w kazdej wersji ultralytics
                          (scripts/final_model.sh, krok 2)

Model koncowy:
    python -m models.cli nadprobkuj zbior_lotnicze/dataset --waga 2023-05-25=3 --waga 2023-05-30=2 \\
           --dodaj-trening zbior_naziemne/dataset --dodaj-trening zbior_ewis1_podzial/dataset \\
           -o zbior_lotnicze/dataset/data_dane_ewis1_bt.yaml
"""

import logging
import re
import shutil
from pathlib import Path

import yaml

from .plan import katalog_etykiet

log = logging.getLogger(__name__)
WZORZEC_DNIA = r"\d{4}-\d{2}-\d{2}"         # data w nazwie kafelka WeedsGalore, np. 2023-05-25
ROZSZERZENIA = (".jpg", ".jpeg", ".png")
NADPROB = "train_nadprob"                   # folder na kopie: images/train_nadprob, labels/train_nadprob


def _dzien(nazwa, wzorzec=WZORZEC_DNIA):
    """Data z nazwy pliku.

    Argumenty:
        nazwa   - nazwa kafelka, np. "weedsgalore__2023-05-25__2023-05-25_0012.jpg"
        wzorzec - wyrazenie regularne daty

    Zwraca:
        str, np. "2023-05-25", albo "bez daty"

    Uzywana w:
        nadprobkuj() (ten plik)
    """
    m = re.search(wzorzec, nazwa)
    return m.group(0) if m else "bez daty"


def _nazwy_klas(katalog_zbioru):
    """Nazwy klas z data.yaml zbioru (zeby nowy data.yaml mial te same).

    Argumenty:
        katalog_zbioru - folder dataset/ z data.yaml

    Zwraca:
        dict numer -> nazwa, np. {0: "chwast"}

    Uzywana w:
        nadprobkuj() (ten plik)
    """
    p = Path(katalog_zbioru) / "data.yaml"
    if p.exists():
        return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("names", {0: "chwast"})
    return {0: "chwast"}


# ---------------------------------------------------------------- nadprobkowanie

def nadprobkuj(katalog_zbioru, wagi, wyjscie_yaml=None, dodatkowe_train=(), wzorzec=WZORZEC_DNIA):
    """Kopie kafelkow treningowych wybranych dni: waga 3 = kafelek 3 razy w treningu.

    Kopie trafiaja do images/train_nadprob i labels/train_nadprob z nazwa
    <nazwa>__kopiaN, wiec kontrola przecieku ich nie myli z innymi czesciami.
    Walidacja i test zostaja bez zmian - wynik jest porownywalny z wczesniejszymi.
    Zwraca {dzien: {"przed": n, "po": n}}.

    Po co: male chwasty z poczatku sezonu sa najtrudniejsze, a jest ich w treningu
           najmniej - kopie zwiekszaja ich udzial bez zmiany walidacji i testu.

    Argumenty:
        katalog_zbioru  - dataset WeedsGalore, np. "zbior_lotnicze/dataset"
        wagi            - dzien -> ile razy kafelek ma byc w treningu,
                          np. {"2023-05-25": 3, "2023-05-30": 2}; inne dni = 1 (bez kopii)
        wyjscie_yaml    - gdzie zapisac nowy data.yaml albo None (same kopie)
        dodatkowe_train - inne zbiory, z ktorych do treningu idzie TYLKO images/train,
                          np. ["zbior_naziemne/dataset", "zbior_ewis1_podzial/dataset"]
        wzorzec         - wyrazenie regularne daty w nazwie kafelka

    Zwraca:
        dict, np. {"2023-05-25": {"przed": 32, "po": 96}, "2023-05-30": {"przed": 32, "po": 64}, ...}
        i zapisuje data.yaml z:
            train: [images/train, images/train_nadprob, <Sudars>/images/train, <EWIS1>/images/train]
            val: images/val      (tylko WeedsGalore)
            test: images/test    (tylko WeedsGalore)

    Uzywana w:
        models/cli.py cmd_nadprobkuj() - komenda "nadprobkuj" w scripts/final_model.sh, krok 2
    """
    ds = Path(katalog_zbioru).resolve()
    obr, et = ds / "images" / "train", katalog_etykiet(ds / "images" / "train")
    cel_obr, cel_et = ds / "images" / NADPROB, ds / "labels" / NADPROB
    # stare kopie kasowane - ponowne uruchomienie nie dokleja kolejnych
    for k in (cel_obr, cel_et):
        if k.exists():
            shutil.rmtree(k)
        k.mkdir(parents=True)
    licznik = {}
    for p in sorted(obr.iterdir()):
        if p.suffix.lower() not in ROZSZERZENIA:
            continue
        d = _dzien(p.name, wzorzec)
        w = max(1, int(wagi.get(d, 1)))         # waga dnia, co najmniej 1
        c = licznik.setdefault(d, {"przed": 0, "po": 0})
        c["przed"] += 1
        c["po"] += w
        etykieta = et / (p.stem + ".txt")
        for n in range(1, w):                   # waga 3 -> kopia1, kopia2 (oryginal juz jest)
            nowa = f"{p.stem}__kopia{n}"
            shutil.copy(p, cel_obr / (nowa + p.suffix))
            if etykieta.exists():
                shutil.copy(etykieta, cel_et / (nowa + ".txt"))
            else:
                (cel_et / (nowa + ".txt")).write_text("", encoding="utf-8")   # pusty kafelek
    if wyjscie_yaml:
        # trening: oryginaly + kopie + images/train z dodatkowych zbiorow (pelne sciezki)
        train = ["images/train", f"images/{NADPROB}"]
        train += [str((Path(x).resolve() / "images" / "train")) for x in dodatkowe_train]
        dane = {"path": str(ds), "train": train, "val": "images/val", "test": "images/test",
                "names": _nazwy_klas(ds)}
        Path(wyjscie_yaml).write_text(yaml.safe_dump(dane, allow_unicode=True, sort_keys=False),
                                      encoding="utf-8")
    for d, c in sorted(licznik.items()):
        log.info("  %-12s kafelkow %3d -> %3d", d, c["przed"], c["po"])
    return licznik
