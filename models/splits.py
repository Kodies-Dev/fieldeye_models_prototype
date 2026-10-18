"""Podzial zbioru calymi dniami lotu: jeden dzien w calosci jako test.

Oficjalny podzial WeedsGalore jest warstwowany po dniach - kazdy dzien jest
w treningu i w tescie. Wynik mowi wtedy, jak model radzi sobie z NOWYMI
FRAGMENTAMI tego samego lotu. Tutaj test to caly dzien, ktorego model nie
widzial - czyli to, co bedzie w praktyce: nowy lot, inny dzien, inna faza
wzrostu.

    test   caly jeden dzien
    val    czesc kafelkow z kazdego z pozostalych dni (wybor epoki i progu)
    train  reszta

Przydzial do val jest deterministyczny (skrot nazwy pliku), wiec dwa
uruchomienia daja ten sam podzial.

Do treningu mozna dolozyc inne zbiory (np. Sudars) - trafia tylko ich czesc
treningowa, a walidacja i test zostaja z samego zbioru glownego. Dzieki temu
wybor najlepszej epoki i wynik testu dotycza zdjec z drona.
"""

import hashlib
import logging
import os
import re
import shutil
from pathlib import Path

import yaml

from .plan import katalog_etykiet

log = logging.getLogger(__name__)
ROZSZERZENIA = (".jpg", ".jpeg", ".png")
WZORZEC_DNIA = r"\d{4}-\d{2}-\d{2}"


def obrazy_wedlug_dni(katalog_zbioru, wzorzec=WZORZEC_DNIA):
    """Wszystkie obrazy zbioru (ze wszystkich czesci) pogrupowane po dniu z nazwy."""
    dni = {}
    for p in sorted(Path(katalog_zbioru, "images").rglob("*")):
        if p.suffix.lower() in ROZSZERZENIA:
            m = re.search(wzorzec, p.name)
            if m:
                dni.setdefault(m.group(0), []).append(p)
    return dni


def _klucz(p):
    """Staly, pseudolosowy porzadek plikow - ten sam przy kazdym uruchomieniu."""
    return hashlib.md5(p.name.encode()).hexdigest()


def _skopiuj(zrodlo, cel):
    """Twarde dowiazanie (bez zajmowania miejsca), a gdy sie nie da - kopia."""
    cel.parent.mkdir(parents=True, exist_ok=True)
    if cel.exists():
        cel.unlink()
    try:
        os.link(zrodlo, cel)
    except OSError:
        shutil.copy2(zrodlo, cel)


def podziel_dniami(zbior, wyjscie, dzien_testowy, udzial_val=0.15, dodatkowe_treningowe=()):
    """Buduje zbior z podzialem calymi dniami.

    zbior                 katalog "dataset" zbioru glownego (images/, labels/, data.yaml)
    dzien_testowy         np. "2023-05-25"
    udzial_val            udzial walidacji w CALYM zbiorze (np. 0.15)
    dodatkowe_treningowe  katalogi "dataset" innych zbiorow - dokladana ich czesc train

    Zapisuje wyjscie/dataset/{images,labels}/{train,val,test} i data.yaml, a gdy
    sa zbiory dodatkowe - takze data_z_dodatkowymi.yaml. Zwraca podsumowanie.
    """
    zbior, wyjscie = Path(zbior), Path(wyjscie)
    dni = obrazy_wedlug_dni(zbior)
    if dzien_testowy not in dni:
        raise ValueError(f"brak dnia {dzien_testowy}; dostepne: {', '.join(sorted(dni))}")
    if len(dni) < 2:
        raise ValueError("potrzebne sa co najmniej dwa dni")

    wszystkich = sum(len(v) for v in dni.values())
    test = dni[dzien_testowy]
    # udzial walidacji w calym zbiorze -> udzial w dniach treningowych
    udzial_w_reszcie = min(0.9, udzial_val * wszystkich / max(1, wszystkich - len(test)))
    czesci = {"train": [], "val": [], "test": list(test)}
    for d, pliki in dni.items():
        if d == dzien_testowy:
            continue
        # DOKLADNIE tyle kafelkow z kazdego dnia, ile wynika z udzialu. Losowanie
        # kazdego pliku osobno dawalo przy malych liczbach duze odchylenia
        # (np. 31 zamiast 23 w walidacji, kosztem treningu).
        ile = round(udzial_w_reszcie * len(pliki))
        kolejnosc = sorted(pliki, key=_klucz)
        czesci["val"] += kolejnosc[:ile]
        czesci["train"] += kolejnosc[ile:]

    ds = wyjscie / "dataset"
    if ds.exists():
        shutil.rmtree(ds)
    for czesc, pliki in czesci.items():
        for p in pliki:
            _skopiuj(p, ds / "images" / czesc / p.name)
            et = katalog_etykiet(p.parent) / (p.stem + ".txt")
            cel = ds / "labels" / czesc / (p.stem + ".txt")
            if et.exists():
                _skopiuj(et, cel)
            else:
                cel.parent.mkdir(parents=True, exist_ok=True)
                cel.write_text("")

    zrodlowy = yaml.safe_load((zbior / "data.yaml").read_text(encoding="utf-8")) or {}
    nazwy = zrodlowy.get("names", {0: "chwast"})
    baza = {"path": str(ds.resolve()), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": nazwy}
    (ds / "data.yaml").write_text(yaml.safe_dump(baza, allow_unicode=True, sort_keys=False),
                                  encoding="utf-8")

    info = {"dzien_testowy": dzien_testowy,
            "dni_treningowe": sorted(d for d in dni if d != dzien_testowy),
            "obrazow": {c: len(v) for c, v in czesci.items()},
            "udzial": {c: round(len(v) / wszystkich, 3) for c, v in czesci.items()}}

    if dodatkowe_treningowe:
        z_dod = dict(baza)
        z_dod["train"] = ["images/train"]
        for k in dodatkowe_treningowe:
            kat = Path(k).resolve() / "images" / "train"
            if not kat.is_dir():
                raise ValueError(f"brak czesci treningowej w {k}")
            z_dod["train"].append(str(kat))
        (ds / "data_z_dodatkowymi.yaml").write_text(
            yaml.safe_dump(z_dod, allow_unicode=True, sort_keys=False), encoding="utf-8")
        info["dodatkowe_treningowe"] = [str(k) for k in dodatkowe_treningowe]

    log.info("Podzial dniami: test=%s %s", dzien_testowy, info["obrazow"])
    return info
