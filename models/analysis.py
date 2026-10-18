"""Analiza zbioru YOLO: co naprawde jest w danych, zanim zacznie sie trening.

    python -m models.cli analiza zbior_lotnicze/dataset/data.yaml -o analiza_lotnicze

Liczy dla kazdej czesci podzialu (train / val / test) i kazdego dnia lotu:
  - ile jest obrazow, obiektow i obrazow bez obiektow,
  - ile obiektow przypada na obraz,
  - jak duze sa obiekty w pikselach (bok = pierwiastek z pola ramki)
    i jaka czesc to obiekty male / srednie / duze wedlug progow COCO,
  - gdzie na obrazie leza obiekty (czy nie skupiaja sie np. przy krawedziach).

Zapisuje raport.json, raport.md i wykresy PNG. Wykresy wymagaja matplotlib
(instaluje sie razem z ultralytics); bez niego powstaje sam raport.

Po co: srednia z calego zbioru potrafi ukryc rzeczy, ktore potem decyduja
o wyniku modelu. W WeedsGalore najwczesniejszy dzien ma kilka razy mniejsze
i rzadsze chwasty niz pozostale - widac to od razu w tabeli dni, a w
sredniej z calego zbioru wcale.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from .plan import katalog_etykiet

log = logging.getLogger(__name__)
ROZSZERZENIA = (".jpg", ".jpeg", ".png")
WZORZEC_DNIA = r"\d{4}-\d{2}-\d{2}"
MALE, SREDNIE = 32, 96              # progi COCO dla boku obiektu w pikselach


def _rozmiar_obrazu(sciezka):
    """(szer, wys) z naglowka pliku - bez wczytywania calego obrazu."""
    from PIL import Image
    with Image.open(sciezka) as im:
        return im.size


def _katalogi(dane_yaml):
    """{czesc: [katalogi obrazow]} - obsluguje tez liste katalogow (np. train z Sudarsem)."""
    dane_yaml = Path(dane_yaml)
    d = yaml.safe_load(dane_yaml.read_text(encoding="utf-8")) or {}
    baza = Path(d.get("path") or dane_yaml.parent)
    if not baza.is_absolute():
        baza = (dane_yaml.parent / baza).resolve()
    nazwy = d.get("names", {})
    if isinstance(nazwy, list):
        nazwy = dict(enumerate(nazwy))
    wynik = {}
    for czesc in ("train", "val", "test"):
        w = d.get(czesc)
        if not w:
            continue
        lista = w if isinstance(w, list) else [w]
        wynik[czesc] = [Path(x) if Path(x).is_absolute() else baza / x for x in lista]
    return wynik, {int(k): v for k, v in nazwy.items()}


def _statystyki(wartosci):
    if not wartosci:
        return {"srednia": 0, "mediana": 0, "min": 0, "max": 0, "p10": 0, "p90": 0}
    a = np.asarray(wartosci, float)
    return {"srednia": round(float(a.mean()), 2), "mediana": round(float(np.median(a)), 2),
            "min": round(float(a.min()), 2), "max": round(float(a.max()), 2),
            "p10": round(float(np.percentile(a, 10)), 2), "p90": round(float(np.percentile(a, 90)), 2)}


def _udzialy(boki):
    if not boki:
        return {"male": 0, "srednie": 0, "duze": 0}
    a = np.asarray(boki)
    n = len(a)
    return {"male": round(float((a < MALE).sum() / n), 4),
            "srednie": round(float(((a >= MALE) & (a < SREDNIE)).sum() / n), 4),
            "duze": round(float((a >= SREDNIE).sum() / n), 4)}


def zbierz(dane_yaml, wzorzec_dnia=WZORZEC_DNIA):
    """Jedno przejscie po zbiorze -> lista obrazow z ich obiektami (w pikselach)."""
    katalogi, nazwy = _katalogi(dane_yaml)
    obrazy = []
    for czesc, kats in katalogi.items():
        for kat in kats:
            if not kat.is_dir():
                log.warning("brak katalogu %s", kat)
                continue
            kat_et = katalog_etykiet(kat)
            for p in sorted(kat.iterdir()):
                if p.suffix.lower() not in ROZSZERZENIA:
                    continue
                w, h = _rozmiar_obrazu(p)
                m = re.search(wzorzec_dnia, p.name)
                rekord = {"plik": p.name, "czesc": czesc, "zrodlo": kat.parent.parent.parent.name,
                          "dzien": m.group(0) if m else "bez daty", "szer": w, "wys": h, "obiekty": []}
                et = kat_et / (p.stem + ".txt")
                if et.exists():
                    for linia in et.read_text(encoding="utf-8").split("\n"):
                        cz = linia.split()
                        if len(cz) < 5:
                            continue
                        k = int(float(cz[0]))
                        cx, cy, bw, bh = (float(v) for v in cz[1:5])
                        rekord["obiekty"].append({"klasa": nazwy.get(k, str(k)), "cx": cx, "cy": cy,
                                                  "w_px": bw * w, "h_px": bh * h})
                obrazy.append(rekord)
    return obrazy


def podsumuj(obrazy):
    """Lista obrazow -> slownik z wszystkimi liczbami raportu."""
    def blok(grupa):
        boki = [float(np.sqrt(o["w_px"] * o["h_px"])) for r in grupa for o in r["obiekty"]]
        na_obraz = [len(r["obiekty"]) for r in grupa]
        return {"obrazow": len(grupa), "obiektow": int(sum(na_obraz)),
                "puste": int(sum(1 for n in na_obraz if n == 0)),
                "obiekty_na_obraz": _statystyki(na_obraz),
                "bok_px": _statystyki(boki), "udzial": _udzialy(boki)}

    raport = {"razem": blok(obrazy), "czesci": {}, "dni": {}, "dni_x_czesci": {}, "klasy": {},
              "rozmiary_obrazow": {}}
    for c in ("train", "val", "test"):
        g = [r for r in obrazy if r["czesc"] == c]
        if g:
            raport["czesci"][c] = blok(g)
    for d in sorted({r["dzien"] for r in obrazy}):
        g = [r for r in obrazy if r["dzien"] == d]
        raport["dni"][d] = blok(g)
        raport["dni_x_czesci"][d] = {c: sum(1 for r in g if r["czesc"] == c) for c in ("train", "val", "test")}
    raport["klasy"] = dict(Counter(o["klasa"] for r in obrazy for o in r["obiekty"]).most_common())
    raport["rozmiary_obrazow"] = {f"{w}x{h}": n for (w, h), n in
                                  Counter((r["szer"], r["wys"]) for r in obrazy).most_common()}
    return raport


def _md(raport):
    """Raport w Markdown - czytelny bez zadnych narzedzi."""
    L = ["# Analiza zbioru", ""]
    r = raport["razem"]
    L += [f"Obrazow: {r['obrazow']}, obiektow: {r['obiektow']}, obrazow bez obiektow: {r['puste']}.", "",
          f"Bok obiektu [px]: mediana {r['bok_px']['mediana']}, 10% ponizej {r['bok_px']['p10']}, "
          f"90% ponizej {r['bok_px']['p90']}.", "",
          f"Udzial obiektow: male {r['udzial']['male']:.0%}, srednie {r['udzial']['srednie']:.0%}, "
          f"duze {r['udzial']['duze']:.0%} (progi COCO: {MALE} i {SREDNIE} px).", ""]
    for tytul, klucz in (("Czesci podzialu", "czesci"), ("Dni lotu", "dni")):
        L += [f"## {tytul}", "", "| | obrazow | obiektow | puste | obiektow na obraz | mediana boku [px] | male |",
              "|---|---|---|---|---|---|---|"]
        for n, b in raport[klucz].items():
            L.append(f"| {n} | {b['obrazow']} | {b['obiektow']} | {b['puste']} | "
                     f"{b['obiekty_na_obraz']['srednia']} | {b['bok_px']['mediana']} | {b['udzial']['male']:.0%} |")
        L.append("")
    L += ["## Dni w czesciach podzialu", "", "| dzien | train | val | test |", "|---|---|---|---|"]
    for d, c in raport["dni_x_czesci"].items():
        L.append(f"| {d} | {c['train']} | {c['val']} | {c['test']} |")
    L += ["", "## Klasy", ""] + [f"- {k}: {n}" for k, n in raport["klasy"].items()]
    L += ["", "## Rozmiary obrazow", ""] + [f"- {k}: {n}" for k, n in raport["rozmiary_obrazow"].items()]
    return "\n".join(L) + "\n"


def wykresy(obrazy, raport, katalog):
    """Cztery wykresy PNG. Zwraca liste zapisanych plikow (pusta bez matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("brak matplotlib - pomijam wykresy")
        return []
    katalog = Path(katalog)
    pliki = []

    # 1. rozmiar obiektow wedlug dnia
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for d in raport["dni"]:
        boki = [np.sqrt(o["w_px"] * o["h_px"]) for r in obrazy if r["dzien"] == d for o in r["obiekty"]]
        if boki:
            ax.hist(boki, bins=np.arange(0, 101, 2), histtype="step", linewidth=1.8, label=d)
    ax.axvline(MALE, color="gray", linestyle="--", linewidth=1)
    ax.text(MALE + 1, ax.get_ylim()[1] * 0.92, "granica obiektów małych (32 px)", color="gray", fontsize=9)
    ax.set_xlabel("bok obiektu [px]")
    ax.set_ylabel("liczba obiektów")
    ax.set_title("Rozmiar obiektów według dnia lotu")
    ax.legend()
    fig.tight_layout()
    pliki.append(katalog / "rozmiary_wedlug_dni.png")
    fig.savefig(pliki[-1], dpi=150)
    plt.close(fig)

    # 2. obiekty na obraz wedlug dnia
    fig, ax = plt.subplots(figsize=(9, 4.5))
    dni = list(raport["dni"])
    dane = [[len(r["obiekty"]) for r in obrazy if r["dzien"] == d] for d in dni]
    ax.boxplot(dane)
    ax.set_xticks(range(1, len(dni) + 1), dni)          # bez labels= - nazwa zmienia sie miedzy wersjami
    ax.set_ylabel("obiektów na obraz")
    ax.set_title("Ile obiektów jest na jednym obrazie")
    fig.tight_layout()
    pliki.append(katalog / "obiekty_na_obraz.png")
    fig.savefig(pliki[-1], dpi=150)
    plt.close(fig)

    # 3. dni w czesciach podzialu
    fig, ax = plt.subplots(figsize=(9, 4.5))
    dol = np.zeros(len(dni))
    for c, kolor in (("train", "#4C72B0"), ("val", "#DD8452"), ("test", "#55A868")):
        v = np.array([raport["dni_x_czesci"][d][c] for d in dni])
        ax.bar(dni, v, bottom=dol, label=c, color=kolor)
        dol += v
    ax.set_ylabel("obrazów")
    ax.set_title("Dni lotu w częściach podziału")
    ax.legend()
    fig.tight_layout()
    pliki.append(katalog / "dni_w_podziale.png")
    fig.savefig(pliki[-1], dpi=150)
    plt.close(fig)

    # 4. polozenie obiektow na obrazie
    cx = [o["cx"] for r in obrazy for o in r["obiekty"]]
    cy = [o["cy"] for r in obrazy for o in r["obiekty"]]
    if cx:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        h = ax.hist2d(cx, cy, bins=25, range=[[0, 1], [0, 1]], cmap="viridis")
        ax.invert_yaxis()
        fig.colorbar(h[3], ax=ax, label="obiektów")
        ax.set_xlabel("x (ułamek szerokości)")
        ax.set_ylabel("y (ułamek wysokości)")
        ax.set_title("Gdzie na obrazie leżą obiekty")
        fig.tight_layout()
        pliki.append(katalog / "polozenie_obiektow.png")
        fig.savefig(pliki[-1], dpi=150)
        plt.close(fig)
    return pliki


def analizuj(dane_yaml, wyjscie=None, wzorzec_dnia=WZORZEC_DNIA):
    """Cala analiza: zbieranie, podsumowanie, raport i wykresy."""
    obrazy = zbierz(dane_yaml, wzorzec_dnia)
    raport = podsumuj(obrazy)
    if wyjscie:
        wyjscie = Path(wyjscie)
        wyjscie.mkdir(parents=True, exist_ok=True)
        (wyjscie / "raport.json").write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding="utf-8")
        (wyjscie / "raport.md").write_text(_md(raport), encoding="utf-8")
        raport["wykresy"] = [str(p) for p in wykresy(obrazy, raport, wyjscie)]
    return raport
