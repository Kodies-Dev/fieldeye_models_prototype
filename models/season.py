"""Sezonowosc: zeby model dzialal nie tylko w czerwcu, ale tez na najmlodszych siewkach.

Problem (zmierzony): model wg_sudars znajduje ok. 82% chwastow 6 czerwca i tylko
27-28% 25 maja. Majowe kafelki maja ok. 41 chwastow, czerwcowe ok. 90 - wiekszosc
przykladow, na ktorych model sie uczy, pochodzi wiec z pozniejszej fazy.

Narzedzia:
  nadprobkuj()          - kafelki wczesnych dni czesciej w treningu (oversampling),
                          przez kopie z inna nazwa: dziala w kazdej wersji ultralytics
  wybierz_dni()         - zbior tylko z wybranych dni, z tym samym podzialem
                          (do dostrojenia modelu "wczesnego")
  progi_wedlug_dni()    - osobny prog ufnosci dla kazdego dnia, dobrany na walidacji
  ocen_z_progami_dni()  - wynik na tescie, gdy kazdy dzien ma swoj prog

    python -m models.cli nadprobkuj zbior_lotnicze/dataset --waga 2023-05-25=3 --waga 2023-05-30=2 \\
           --dodaj-trening zbior_naziemne/dataset -o zbior_lotnicze/dataset/data_sezon.yaml
    python -m models.cli prog-dni modele_X/model.pt zbior_lotnicze/dataset/data.yaml --wejscie 1280 -o progi.json
    python -m models.cli ocen-sezon modele_X/model.pt zbior_lotnicze/dataset/data.yaml --wejscie 1280 \\
           --progi progi.json -o wyniki/X_sezon.json
"""

import logging
import re
import shutil
from pathlib import Path

import yaml

from .evaluation import pr_przy_progach
from .plan import katalog_etykiet

log = logging.getLogger(__name__)
WZORZEC_DNIA = r"\d{4}-\d{2}-\d{2}"
ROZSZERZENIA = (".jpg", ".jpeg", ".png")
NADPROB = "train_nadprob"
PROGI = (0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3)


def _dzien(nazwa, wzorzec=WZORZEC_DNIA):
    m = re.search(wzorzec, nazwa)
    return m.group(0) if m else "bez daty"


def _nazwy_klas(katalog_zbioru):
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
    """
    ds = Path(katalog_zbioru).resolve()
    obr, et = ds / "images" / "train", katalog_etykiet(ds / "images" / "train")
    cel_obr, cel_et = ds / "images" / NADPROB, ds / "labels" / NADPROB
    for k in (cel_obr, cel_et):
        if k.exists():
            shutil.rmtree(k)
        k.mkdir(parents=True)
    licznik = {}
    for p in sorted(obr.iterdir()):
        if p.suffix.lower() not in ROZSZERZENIA:
            continue
        d = _dzien(p.name, wzorzec)
        w = max(1, int(wagi.get(d, 1)))
        c = licznik.setdefault(d, {"przed": 0, "po": 0})
        c["przed"] += 1
        c["po"] += w
        etykieta = et / (p.stem + ".txt")
        for n in range(1, w):
            nowa = f"{p.stem}__kopia{n}"
            shutil.copy(p, cel_obr / (nowa + p.suffix))
            if etykieta.exists():
                shutil.copy(etykieta, cel_et / (nowa + ".txt"))
            else:
                (cel_et / (nowa + ".txt")).write_text("", encoding="utf-8")
    if wyjscie_yaml:
        train = ["images/train", f"images/{NADPROB}"]
        train += [str((Path(x).resolve() / "images" / "train")) for x in dodatkowe_train]
        dane = {"path": str(ds), "train": train, "val": "images/val", "test": "images/test",
                "names": _nazwy_klas(ds)}
        Path(wyjscie_yaml).write_text(yaml.safe_dump(dane, allow_unicode=True, sort_keys=False),
                                      encoding="utf-8")
    for d, c in sorted(licznik.items()):
        log.info("  %-12s kafelkow %3d -> %3d", d, c["przed"], c["po"])
    return licznik


# ---------------------------------------------------------------- zbior z wybranych dni

def wybierz_dni(katalog_zbioru, dni, wyjscie, wzorzec=WZORZEC_DNIA):
    """Kopia zbioru tylko z wybranymi dniami, czesci podzialu bez zmian.

    Do dostrojenia modelu "wczesnego": te same kafelki testowe z danego dnia
    co w oficjalnym podziale, wiec wynik jest wprost porownywalny.
    """
    ds, out = Path(katalog_zbioru), Path(wyjscie) / "dataset"
    if out.exists():
        shutil.rmtree(out)
    dni = set(dni)
    ile = {}
    for czesc in ("train", "val", "test"):
        src = ds / "images" / czesc
        if not src.is_dir():
            continue
        et = katalog_etykiet(src)
        (out / "images" / czesc).mkdir(parents=True)
        (out / "labels" / czesc).mkdir(parents=True)
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in ROZSZERZENIA and _dzien(p.name, wzorzec) in dni:
                shutil.copy(p, out / "images" / czesc / p.name)
                l = et / (p.stem + ".txt")
                if l.exists():
                    shutil.copy(l, out / "labels" / czesc / l.name)
                ile[czesc] = ile.get(czesc, 0) + 1
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
         "test": "images/test", "names": _nazwy_klas(ds)}, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    return ile


# ---------------------------------------------------------------- progi wedlug dni

def _grupy(prawda, wzorzec=WZORZEC_DNIA):
    g = {}
    for stem in prawda:
        g.setdefault(_dzien(stem, wzorzec), []).append(stem)
    return g


def _f_beta(p, r, beta):
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r) if (p + r) else 0.0


def progi_wedlug_dni(predykcje, prawda, progi=PROGI, waga_czulosci=3.0, iou=0.5):
    """Dla kazdego dnia prog z najwyzszym F-beta (domyslnie czulosc 3 razy wazniejsza).

    Licz na czesci 'val' - prog dobrany na tescie zawyzylby wynik.
    """
    wynik = {}
    for dzien, stemy in sorted(_grupy(prawda).items()):
        w = pr_przy_progach({s: predykcje.get(s, []) for s in stemy}, {s: prawda[s] for s in stemy},
                            list(progi), iou)
        for x in w:
            x["f_beta"] = round(_f_beta(x["precyzja"], x["czulosc"], waga_czulosci), 4)
        naj = max(w, key=lambda x: (x["f_beta"], x["prog"]))
        wynik[dzien] = {"prog": naj["prog"], "f_beta": naj["f_beta"], "czulosc": naj["czulosc"],
                        "precyzja": naj["precyzja"], "obrazow": len(stemy), "wszystkie": w}
    return wynik


def ocen_z_progami_dni(predykcje, prawda, progi_dni, domyslny=0.15, iou=0.5):
    """Wynik na kazdym dniu przy jego wlasnym progu i lacznie (zsumowane trafienia)."""
    wynik, suma = {}, {"trafione": 0, "falszywe": 0, "pominiete": 0}
    for dzien, stemy in sorted(_grupy(prawda).items()):
        t = progi_dni.get(dzien, {}).get("prog", domyslny) if isinstance(progi_dni.get(dzien), dict) \
            else progi_dni.get(dzien, domyslny)
        w = pr_przy_progach({s: predykcje.get(s, []) for s in stemy}, {s: prawda[s] for s in stemy},
                            [t], iou)[0]
        w["obrazow"] = len(stemy)
        wynik[dzien] = w
        for k in suma:
            suma[k] += w[k]
    tp, fp, fn = suma["trafione"], suma["falszywe"], suma["pominiete"]
    wynik["razem"] = dict(suma, prog="wg dni", obrazow=len(prawda),
                          precyzja=round(tp / (tp + fp), 4) if tp + fp else 0.0,
                          czulosc=round(tp / (tp + fn), 4) if tp + fn else 0.0)
    return wynik