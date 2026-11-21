"""Symulowany lot nad polem z kafelkow - funkcje zakladki 4 interfejsu (app.py) i symulowanego drona.

    from mission.sim_flight import kafelki_wedlug_dni, pole_z_kafelkow
    pole, prawda = pole_z_kafelkow(kafelki_wedlug_dni(katalog)["2023-05-25"], z_etykietami=True)

Skad pole do lotu:
    pole_z_kafelkow()  - kafelki WeedsGalore z jednego dnia sklejone w siatke
    pole_ze_zdjecia()  - zdjecie EWIS1 odtworzone z jego kafelkow (..._k{x}_{y}), co do piksela

Po detekcji na kadrach: prawda_w_kadrze() (prawdziwe ramki w kadrze), scal_ramki()
(ten sam chwast z kilku kadrow -> jedna ramka), porownaj() (trafione / pominiete / falszywe),
rysuj_ramki() (podglad).

Uzywany w:
    app.py zakladka 4, mission/drone_sim.py (--ewis1), tests/test_sim_tiles.py, test_gui_flight.py
"""

import re
from pathlib import Path

import cv2
import numpy as np

from models.evaluation import pr_przy_progach
from models.plan import katalog_etykiet
from stitchlive.flight_sim import macierz_kadru

from .detector import iou_ramek

ZIELONY, ROZOWY = (60, 200, 60), (255, 0, 255)


def kafelki_wedlug_dni(katalog):
    """{dzien lotu: [pliki]} - dzien z nazwy pliku (RRRR-MM-DD), reszta jako "bez daty".

    Argumenty:
        katalog - folder kafelkow, np. "zbior_lotnicze/dataset/images/test"

    Zwraca:
        dict, np. {"2023-05-25": [Path(...), ...], "2023-06-06": [...]}

    Uzywana w:
        app.py zakladka 4, tests
    """
    dni = {}
    for p in sorted(Path(katalog).glob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            m = re.search(r"\d{4}-\d{2}-\d{2}", p.name)
            dni.setdefault(m.group(0) if m else "bez daty", []).append(p)
    return dni


WZOR_KAFELKA = re.compile(r"^(?P<zdjecie>.+)_k(?P<x>\d+)_(?P<y>\d+)$")


def zdjecia_z_kafelkow(pliki):
    """{zdjecie zrodlowe: [(x, y, plik)]} dla kafelkow z polozeniem w nazwie (..._k{x}_{y}).

    Tak nazywa kafelki normalizacja, gdy tnie wieksze zdjecie (np. EWIS1): x, y to
    lewy gorny rog kafelka na zdjeciu. Kafelki bez polozenia w nazwie (WeedsGalore:
    jedno zdjecie = jeden kafelek) sa pomijane.

    Argumenty:
        pliki - sciezki kafelkow, np. Path("zbior_ewis1_podzial/dataset/images/test").glob("*")

    Zwraca:
        dict, np. {"ewis1__Loew_20220516_Maize_003": [(0, 0, p1), (806, 0, p2), ...]}
        (kafelki posortowane wierszami; tylko zdjecia z co najmniej 2 kafelkami)

    Uzywana w:
        mission/drone_sim.py (--ewis1), tests
    """
    out = {}
    for p in pliki:
        p = Path(p)
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        m = WZOR_KAFELKA.match(p.stem)
        if m:
            out.setdefault(m["zdjecie"], []).append((int(m["x"]), int(m["y"]), p))
    return {k: sorted(v, key=lambda t: (t[1], t[0])) for k, v in sorted(out.items()) if len(v) >= 2}


def _scal_powtorzone(ramki, prog=0.6):
    """Ten sam chwast z dwoch zachodzacych kafelkow -> jedna ramka.

    Na kafelku chwast przy krawedzi bywa przyciety, wiec zostaje wieksza ramka,
    a mniejsza odpada, gdy co najmniej `prog` jej powierzchni lezy w wiekszej.

    Argumenty:
        ramki - lista (x0, y0, x1, y1) w px odtworzonego zdjecia
        prog  - jaka czesc mniejszej ramki musi lezec w wiekszej, np. 0.6

    Zwraca:
        lista ramek bez powtorzen

    Uzywana w:
        pole_ze_zdjecia() (ten plik)
    """
    pole = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])
    out = []
    for r in sorted(ramki, key=pole, reverse=True):
        a = pole(r)
        if a <= 0:
            continue
        if any(max(0.0, min(r[2], q[2]) - max(r[0], q[0])) * max(0.0, min(r[3], q[3]) - max(r[1], q[1]))
               >= prog * a for q in out):
            continue
        out.append(r)
    return out


def pole_ze_zdjecia(kafelki, z_etykietami=False):
    """Odtwarza zdjecie zrodlowe z jego kafelkow - ciagle pole bez powtorzen.

    Kafelki lezacego obok siebie w siatce (pole_z_kafelkow) zachodza na siebie,
    wiec te same rosliny bylyby w polu dwa razy i mozaika sie nie sklada. Tu kazdy
    kafelek trafia dokladnie tam, skad zostal wyciety, a obszary zakladki pokrywaja
    sie co do piksela. `kafelki` to lista (x, y, plik) z zdjecia_z_kafelkow().

    Argumenty:
        kafelki      - lista (x, y, plik) jednego zdjecia
        z_etykietami - True = zwroc tez prawdziwe ramki z plikow YOLO

    Zwraca:
        obraz BGR (h, w, 3) albo (obraz, ramki) przy z_etykietami; None, gdy nic sie nie wczytalo

    Uzywana w:
        mission/drone_sim.py (--ewis1), tests
    """
    obrazy = [(x, y, cv2.imread(str(p)), Path(p)) for x, y, p in kafelki]
    obrazy = [o for o in obrazy if o[2] is not None]
    if not obrazy:
        return None
    W = max(x + o.shape[1] for x, _, o, _ in obrazy)
    H = max(y + o.shape[0] for _, y, o, _ in obrazy)
    pole = np.zeros((H, W, 3), np.uint8)
    for x, y, o, _ in obrazy:
        pole[y:y + o.shape[0], x:x + o.shape[1]] = o
    if not z_etykietami:
        return pole
    ramki = []
    for x, y, o, p in obrazy:
        h, w = o.shape[:2]
        et = katalog_etykiet(p.parent) / (p.stem + ".txt")
        if not et.exists():
            continue
        for linia in et.read_text().split("\n"):
            cz = linia.split()
            if len(cz) >= 5:
                cx, cy, bw, bh = (float(v) for v in cz[1:5])
                ramki.append((x + (cx - bw / 2) * w, y + (cy - bh / 2) * h,
                              x + (cx + bw / 2) * w, y + (cy + bh / 2) * h))
    return pole, _scal_powtorzone(ramki)


def pole_z_kafelkow(pliki, ile=25, z_etykietami=False):
    """Skleja kafelki w prostokatna siatke - zrodlo do symulacji lotu.

    Kafelki musza pochodzic z JEDNEGO dnia lotu. WeedsGalore to jedno pole
    sfotografowane kilka razy - kafelki z roznych dni pokazuja te same miejsca,
    a to samo miejsce w dwoch punktach siatki to dla skladacza sprzecznosc
    i mozaika sie nie sklada.

    Argumenty:
        pliki        - kafelki jednego dnia
        ile          - najwyzej tyle kafelkow, np. 25 (siatka 5x5; tylko pelne wiersze)
        z_etykietami - True = zwroc tez prawdziwe ramki przeniesione na sklejke

    Zwraca:
        obraz BGR albo (obraz, ramki); None, gdy kafelkow mniej niz 4

    Uzywana w:
        app.py zakladka 4, tests
    """
    pliki = list(pliki)[:ile]
    if len(pliki) < 4:
        return None
    kol = int(np.ceil(np.sqrt(len(pliki))))
    wier = len(pliki) // kol                      # tylko pelne wiersze
    obrazy = [cv2.imread(str(p)) for p in pliki[:wier * kol]]
    h, w = obrazy[0].shape[:2]
    obrazy = [cv2.resize(o, (w, h)) for o in obrazy]
    pole = np.vstack([np.hstack(obrazy[r * kol:(r + 1) * kol]) for r in range(wier)])
    if not z_etykietami:
        return pole
    # prawdziwe ramki z plikow YOLO obok kafelkow, przeniesione na sklejke
    prawda = []
    for i, p in enumerate(pliki[:wier * kol]):
        et = katalog_etykiet(Path(p).parent) / (Path(p).stem + ".txt")
        if not et.exists():
            continue
        ox, oy = (i % kol) * w, (i // kol) * h
        for linia in et.read_text().split("\n"):
            cz = linia.split()
            if len(cz) >= 5:
                cx, cy, bw, bh = (float(v) for v in cz[1:5])
                prawda.append((ox + (cx - bw / 2) * w, oy + (cy - bh / 2) * h,
                               ox + (cx + bw / 2) * w, oy + (cy + bh / 2) * h))
    return pole, prawda


def prawda_w_kadrze(prawda, kadr_info, kadr, min_widocznosc=0.4, kadr_wys=None):
    """Prawdziwe ramki ze zrodla -> wspolrzedne kadru.

    Macierz symulatora przenosi piksel kadru na zrodlo, wiec do kadru idziemy
    jej odwrotnoscia. Ramka obcieta krawedzia kadru zostaje, gdy widac co
    najmniej min_widocznosc jej powierzchni - tak samo jak przy kafelkowaniu.

    Argumenty:
        prawda         - ramki (x0, y0, x1, y1) na zrodle (polu)
        kadr_info      - opis kadru z prawda.json symulatora: srodek, kat_st, zasieg_px, kadr_px
        kadr           - szerokosc kadru w px, np. 600
        min_widocznosc - jaka czesc ramki musi byc w kadrze, np. 0.4
        kadr_wys       - wysokosc kadru albo None (z kadr_info)

    Zwraca:
        lista (x0, y0, x1, y1) w px kadru

    Uzywana w:
        app.py zakladka 4, tests
    """
    kh = kadr_wys or (kadr_info.get("kadr_px") or [kadr, kadr])[1]
    M = macierz_kadru(*kadr_info["srodek"], kadr_info["kat_st"], kadr_info["zasieg_px"], kadr, kh)
    Mi = cv2.invertAffineTransform(M)
    out = []
    for x0, y0, x1, y1 in prawda:
        q = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], float) @ Mi.T
        (a0, b0), (a1, b1) = q.min(axis=0), q.max(axis=0)
        pole = (a1 - a0) * (b1 - b0)
        c0, d0, c1, d1 = max(a0, 0), max(b0, 0), min(a1, kadr), min(b1, kh)
        if c1 > c0 and d1 > d0 and pole > 0 and (c1 - c0) * (d1 - d0) >= min_widocznosc * pole:
            out.append((c0, d0, c1, d1))
    return out


def porownaj(wykryte, prawdziwe, iou=0.5):
    """-> (trafione, pominiete, falszywe) dla jednego obrazu.

    Argumenty:
        wykryte    - ramki (x0, y0, x1, y1, pewnosc)
        prawdziwe  - ramki (x0, y0, x1, y1)
        iou        - minimalne IoU trafienia, np. 0.5

    Zwraca:
        krotka, np. (12, 3, 2)

    Uzywana w:
        app.py zakladka 4
    """
    w = pr_przy_progach({"x": [(0, r[4], *r[:4]) for r in wykryte]},
                        {"x": [(0, *r[:4]) for r in prawdziwe]}, [0.0], iou)[0]
    return w["trafione"], w["pominiete"], w["falszywe"]


def rysuj_ramki(img, ramki, grubosc=2, kolor=ROZOWY, kopia=True):
    """Rysuje prostokaty na obrazie.

    Argumenty:
        img     - obraz BGR
        ramki   - lista (x0, y0, x1, y1, ...) - dalsze pola (np. pewnosc) pomijane
        grubosc - grubosc linii w px
        kolor   - BGR, np. ROZOWY (wykryte) albo ZIELONY (prawdziwe)
        kopia   - True = rysuj na kopii, False = na tym samym obrazie

    Zwraca:
        obraz z ramkami

    Uzywana w:
        app.py zakladka 4
    """
    out = img.copy() if kopia else img
    for x0, y0, x1, y1, *_ in ramki:
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), kolor, grubosc)
    return out


def scal_ramki(ramki, prog_iou=0.3):
    """Ten sam chwast z kilku kadrow -> jedna ramka (zostaje najpewniejsza).

    Argumenty:
        ramki    - lista (x0, y0, x1, y1, pewnosc) na mozaice
        prog_iou - od tego IoU dwie ramki to ten sam chwast, np. 0.3

    Zwraca:
        lista ramek bez powtorzen

    Uzywana w:
        app.py zakladka 4
    """
    zostaja = []
    for r in sorted(ramki, key=lambda r: -r[4]):
        if all(iou_ramek(r, z) < prog_iou for z in zostaja):
            zostaja.append(r)
    return zostaja


