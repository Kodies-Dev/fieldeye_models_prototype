"""Laczenie wykryc z wielu zdjec (multi-view fusion).

Przy pokryciu 70% wzdluz trasy i 50% miedzy liniami ten sam chwast trafia na
3-6 kolejnych zdjec, za kazdym razem w innym miejscu kadru. Model moze go
przeoczyc na jednym, a znalezc na innym - albo widziec go na kilku z niska
pewnoscia. Zamiast tylko usuwac powtorzenia (scal_ramki), laczymy dowody:

  wynik = 1 - (1 - p1)(1 - p2)...      (noisy-OR, po jednym p na zdjecie)

  - jedno wykrycie: wynik = jego pewnosc, wiec prog dziala jak dotad,
  - trzy wykrycia po 0,08: wynik 0,22 - przechodzi prog 0,15.

Wariant z glosowaniem dodatkowo odrzuca slabe wykrycie widziane tylko na jednym
z co najmniej trzech zdjec, ktore obejmuja to miejsce - to zwykle falszywy alarm.

Zasada jest ustalona z gory, a nie dobierana na danych testowych.
Wspolrzedne: wspolny uklad wszystkich zdjec (mozaika albo teren w metrach).
"""

import cv2
import numpy as np


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    w = (x1 - x0) * (y1 - y0)
    return w / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w)


def _srodek_w(a, b):
    """Czy srodek ramki a lezy w ramce b - pomaga przy malych, lekko przesunietych ramkach."""
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def grupuj(ramki, iou=0.3):
    """ramki: [(x0, y0, x1, y1, pewnosc, id_zdjecia)] -> lista grup.

    Od najpewniejszej: ramka dolacza do grupy, gdy pokrywa sie z jej ramka
    wiodaca (IoU >= iou) albo jej srodek lezy w ramce wiodacej.
    Grupa: {"ramka", "pewnosci": {id_zdjecia: najwyzsza pewnosc z tego zdjecia}}.
    """
    grupy = []
    for r in sorted(ramki, key=lambda r: -r[4]):
        for g in grupy:
            w = g["ramka"]
            if _iou(r, w) >= iou or _srodek_w(r, w):
                p = g["pewnosci"]
                p[r[5]] = max(p.get(r[5], 0.0), r[4])
                break
        else:
            grupy.append({"ramka": tuple(r[:4]), "pewnosci": {r[5]: r[4]}})
    return grupy


def widocznosc(grupy, wielokaty):
    """Ile zdjec obejmowalo miejsce kazdej grupy: wielokaty = {id_zdjecia: narozniki}."""
    wk = {k: np.asarray(v, np.float32).reshape(-1, 1, 2) for k, v in wielokaty.items()}
    for g in grupy:
        x0, y0, x1, y1 = g["ramka"]
        c = (float((x0 + x1) / 2), float((y0 + y1) / 2))
        g["widzialo"] = sum(1 for w in wk.values() if cv2.pointPolygonTest(w, c, False) >= 0)
    return grupy


def noisy_or(pewnosci):
    return 1.0 - float(np.prod([1.0 - p for p in pewnosci]))


def polacz(ramki, wielokaty=None, prog=0.15, glosowanie=False, iou=0.3,
           prog_niski=None, min_widokow=2):
    """Wykrycia ze wszystkich zdjec -> jedna lista chwastow [(x0, y0, x1, y1, wynik)].

    ramki powinny miec niski prog (np. 0,05) - slabe wykrycia to wlasnie
    dowody, ktore sie sumuja. Prog stosowany jest do wyniku polaczonego.

    prog_niski (np. 0,05): tryb "niski prog z glosowaniem". Chwast przyjety, gdy
    na ktoryms zdjeciu ma pewnosc >= prog (jak zwykle) ALBO gdy co najmniej
    min_widokow zdjec widzi go z pewnoscia >= prog_niski. Pojedyncze slabe
    wykrycie odpada - fałszywka zwykle jest na jednym zdjeciu, chwast na kilku.
    Odpowiedz na problem niskiego progu w maju: wiecej siewek bez lawiny alarmow.
    """
    grupy = grupuj(ramki, iou)
    if wielokaty:
        widocznosc(grupy, wielokaty)
    out = []
    for g in grupy:
        p = list(g["pewnosci"].values())
        wynik = noisy_or(p)
        if prog_niski is not None:
            silny = max(p) >= prog
            slabe_widoki = sum(1 for x in p if x >= prog_niski)
            if not (silny or slabe_widoki >= min_widokow):
                continue
            if silny and glosowanie and g.get("widzialo", 0) >= 3 and len(p) == 1 and max(p) < 2 * prog:
                continue
            out.append((*g["ramka"], max(wynik, prog)))
            continue
        if wynik < prog:
            continue
        if glosowanie and g.get("widzialo", 0) >= 3 and len(p) == 1 and max(p) < 2 * prog:
            continue
        out.append((*g["ramka"], wynik))
    return out