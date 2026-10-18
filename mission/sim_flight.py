"""Symulowany lot z detekcja chwastow: wspolny dla zakladki 4 interfejsu i komendy.

    from mission.sim_flight import pole_z_kafelkow, lot_z_detekcja
    pole, prawda = pole_z_kafelkow(kafelki_dnia, z_etykietami=True)
    w = lot_z_detekcja(pole, prawda, detektor, SimSettings(kadr=600), "lot/")
    w["porownanie"], w["porownanie_fuzja"]       # (trafione, pominiete, falszywe) na mozaice

Kolejno: ciecie pola na kadry (symulator), detekcja na kazdym kadrze (z kafelkami,
jesli kadr jest duzy), skladanie mozaiki, przeniesienie ramek na mozaike i dwa
sposoby laczenia wykryc tego samego chwastu z kilku kadrow:
  - scalanie (dotychczas): zostaje najpewniejsza ramka z tych, ktore przeszly prog,
  - laczenie widokow (mission.fusion): slabe wykrycia z kilku kadrow sumuja sie.
"""

import re
from pathlib import Path

import cv2
import numpy as np

from models.evaluation import pr_przy_progach
from models.plan import katalog_etykiet
from stitchlive.flight_sim import macierz_kadru, rozloz
from stitchlive.global_stitcher import GlobalStitcher

from .detector import TiledDetector
from .fusion import polacz

ZIELONY, ROZOWY = (60, 200, 60), (255, 0, 255)


def kafelki_wedlug_dni(katalog):
    """{dzien lotu: [pliki]} - dzien z nazwy pliku (RRRR-MM-DD), reszta jako "bez daty"."""
    dni = {}
    for p in sorted(Path(katalog).glob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            m = re.search(r"\d{4}-\d{2}-\d{2}", p.name)
            dni.setdefault(m.group(0) if m else "bez daty", []).append(p)
    return dni


def pole_z_kafelkow(pliki, ile=25, z_etykietami=False):
    """Skleja kafelki w prostokatna siatke - zrodlo do symulacji lotu.

    Kafelki musza pochodzic z JEDNEGO dnia lotu. WeedsGalore to jedno pole
    sfotografowane kilka razy - kafelki z roznych dni pokazuja te same miejsca,
    a to samo miejsce w dwoch punktach siatki to dla skladacza sprzecznosc
    i mozaika sie nie sklada.
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
    """-> (trafione, pominiete, falszywe) dla jednego obrazu."""
    w = pr_przy_progach({"x": [(0, r[4], *r[:4]) for r in wykryte]},
                        {"x": [(0, *r[:4]) for r in prawdziwe]}, [0.0], iou)[0]
    return w["trafione"], w["pominiete"], w["falszywe"]


def rysuj_ramki(img, ramki, grubosc=2, kolor=ROZOWY, kopia=True):
    out = img.copy() if kopia else img
    for x0, y0, x1, y1, *_ in ramki:
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), kolor, grubosc)
    return out


def scal_ramki(ramki, prog_iou=0.3):
    """Ten sam chwast z kilku kadrow -> jedna ramka (zostaje najpewniejsza)."""
    def iou(a, b):
        x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        wsp = (x1 - x0) * (y1 - y0)
        return wsp / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - wsp)
    zostaja = []
    for r in sorted(ramki, key=lambda r: -r[4]):
        if all(iou(r, z) < prog_iou for z in zostaja):
            zostaja.append(r)
    return zostaja


def _metryki(wynik):
    t, p, f = wynik
    return {"trafione": t, "pominiete": p, "falszywe": f,
            "czulosc": round(t / (t + p), 4) if t + p else 0.0,
            "precyzja": round(t / (t + f), 4) if t + f else 0.0}


def lot_z_detekcja(zrodlo, prawda_zrodla, detektor, ustawienia, katalog, prog=0.15, prog_min=0.05,
                   kafelek=None, na_kadr=None):
    """Caly symulowany lot. Zwraca slownik z mozaika, ramkami i porownaniem z prawda.

    prawda_zrodla: prawdziwe ramki na zrodle (pusta lista = bez porownania).
    kafelek: bok kafelka w px kadru (kamera drona) - None = caly kadr naraz.
    na_kadr(i, n, img, oznaczony, opis, laczenie, suma): wywolywane po kazdym
    kadrze (podglad w interfejsie). Bledy symulatora (za male pole) to ValueError.
    """
    prawda_lotu = rozloz(zrodlo, katalog, ustawienia)
    kadry = prawda_lotu["kadry"]
    pliki = [Path(katalog) / "images" / k["plik"] for k in kadry]
    det = TiledDetector(detektor, kafelek=kafelek, zakladka=0.15) if kafelek else detektor
    # przy wielu kadrach porownujemy tylko sasiadow: ta sama i sasiednia linia trasy
    na_linii = max(sum(1 for k in kadry if k["linia"] == l) for l in {k["linia"] for k in kadry})
    skl = GlobalStitcher(renderuj_co=10 ** 6, zakres_par=2 * na_linii + 2 if len(pliki) > 40 else 0)
    jest_prawda = bool(prawda_zrodla)

    wszystkie, wykrycia, prawdy, rozmiary = [], [], [], []
    laczenie, suma = 0, [0, 0, 0]
    for i, (p, info_k) in enumerate(zip(pliki, kadry), 1):
        img = cv2.imread(str(p))
        surowe = [(d.x0, d.y0, d.x1, d.y1, d.pewnosc) for d in det.wykryj(img) if d.pewnosc >= prog_min]
        ramki = [r for r in surowe if r[4] >= prog]
        gt = prawda_w_kadrze(prawda_zrodla, info_k, img.shape[1], kadr_wys=img.shape[0]) if jest_prawda else []
        wszystkie.append(surowe)
        wykrycia.append(ramki)
        prawdy.append(gt)
        rozmiary.append(img.shape[:2])
        laczenie += len(ramki)
        if jest_prawda:
            t_, p_, f_ = porownaj(ramki, gt)
            suma = [suma[0] + t_, suma[1] + p_, suma[2] + f_]
            opis = f"trafione {t_} · pominięte {p_} · fałszywe {f_}"
        else:
            opis = f"Wykryte chwasty: {len(ramki)}"
        skl.dodaj(p)
        if na_kadr:
            oznaczony = rysuj_ramki(img, gt, 2, ZIELONY) if jest_prawda else img.copy()
            rysuj_ramki(oznaczony, ramki, 2, ROZOWY, kopia=False)
            na_kadr(i, len(pliki), img, oznaczony, opis, laczenie, suma)

    skl.zakoncz()
    if skl.mozaika is None:
        raise RuntimeError("Nie udało się złożyć mozaiki.")

    def na_mozaike(lista, z_id=False):
        out = []
        for k, (p, ramki, rozm) in enumerate(zip(pliki, lista, rozmiary)):
            for r in ramki:
                x0, y0, x1, y1 = r[:4]
                q = skl.na_mozaike(p, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], rozm)
                if q is not None:
                    (mx0, my0), (mx1, my1) = q.min(axis=0), q.max(axis=0)
                    c = r[4] if len(r) > 4 else 1.0
                    out.append((mx0, my0, mx1, my1, c, k) if z_id else (mx0, my0, mx1, my1, c))
        return out

    # obrys kazdego kadru na mozaice - ile kadrow widzialo dane miejsce
    wielokaty = {}
    for k, (p, (h, w)) in enumerate(zip(pliki, rozmiary)):
        q = skl.na_mozaike(p, [[0, 0], [w, 0], [w, h], [0, h]], (h, w))
        if q is not None:
            wielokaty[k] = q
    unikalne = scal_ramki(na_mozaike(wykrycia))
    surowe_m = na_mozaike(wszystkie, z_id=True)
    fuzja = polacz(surowe_m, wielokaty, prog=prog)
    fuzja_glos = polacz(surowe_m, wielokaty, prog=prog, glosowanie=True)
    niski_glos = polacz(surowe_m, wielokaty, prog=prog, glosowanie=True, prog_niski=prog_min)
    prawdziwe = scal_ramki(na_mozaike(prawdy)) if jest_prawda else None

    wynik = {"mozaika": skl.mozaika, "wykryte": unikalne, "wykryte_fuzja": fuzja,
             "wykryte_fuzja_glos": fuzja_glos, "wykryte_niski_glos": niski_glos, "prawdziwe": prawdziwe,
             "kadrow": len(pliki), "w_mozaice": len(skl.S or {}), "wykryc": laczenie,
             "unikalnych": len(unikalne), "wielokaty": wielokaty, "porownanie": None}
    if jest_prawda:
        wynik["porownanie"] = porownaj(unikalne, prawdziwe)
        wynik["porownanie_fuzja"] = porownaj(fuzja, prawdziwe)
        wynik["porownanie_fuzja_glos"] = porownaj(fuzja_glos, prawdziwe)
        wynik["porownanie_niski_glos"] = porownaj(niski_glos, prawdziwe)
        wynik["metryki"] = {"scalanie": _metryki(wynik["porownanie"]),
                            "laczenie_widokow": _metryki(wynik["porownanie_fuzja"]),
                            "laczenie_z_glosowaniem": _metryki(wynik["porownanie_fuzja_glos"]),
                            "niski_prog_z_glosowaniem": _metryki(wynik["porownanie_niski_glos"])}
    return wynik