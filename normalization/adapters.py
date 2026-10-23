"""Adaptery: kazdy zbior zrodlowy -> manifest.

Jeden adapter na zbior. Reszta potoku (deduplikacja, podzial, kafelkowanie,
eksport) nie wie juz, skad dane pochodza.

Formaty, z ktorymi trzeba sobie poradzic:

  WeedsGalore  kanaly w OSOBNYCH plikach (_R, _G, _B, _NIR, _RE),
               maski semantyczne + instancji
  CoFly        maski: labels_1d (liczby) albo labels (kolory)
  Sudars       Pascal VOC (XML), rozne rozdzielczosci
  WeedMaize    Pascal VOC (XML), kody EPPO z faza wzrostu
  wlasne       YOLO albo Pascal VOC
"""

import logging
import collections
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from .manifest import LabeledObject, Record

log = logging.getLogger(__name__)

OBRAZY = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


# --------------------------------------------------------------- narzedzia

def ramki_z_maski(maska, instancje=None, min_pole=15):
    """Maska -> lista (wartosc, x0, y0, x1, y1, pole).

    Z maska instancji kazda roslina ma wlasny numer, wiec dwie stykajace sie
    rosliny tego samego gatunku daja dwie ramki. Bez niej - spojne skladowe,
    czyli stykajace sie rosliny zleja sie w jedna ramke.
    """
    wynik = []
    if instancje is not None:
        for nr in np.unique(instancje):
            if nr == 0:
                continue
            ys, xs = np.nonzero(instancje == nr)
            if len(xs) < min_pole:
                continue
            wart = maska[ys, xs]
            wart = wart[wart > 0]
            if not len(wart):
                continue
            # klasa instancji = najczestsza wartosc maski pod jej pikselami
            wynik.append((int(np.bincount(wart).argmax()), int(xs.min()), int(ys.min()),
                          int(xs.max()) + 1, int(ys.max()) + 1, int(len(xs))))
        return wynik

    for w in np.unique(maska):
        if w == 0:
            continue
        n, _, st, _ = cv2.connectedComponentsWithStats((maska == w).astype(np.uint8), 8)
        for i in range(1, n):
            x, y, bw, bh, pole = st[i]
            if pole >= min_pole:
                wynik.append((int(w), int(x), int(y), int(x + bw), int(y + bh), int(pole)))
    return wynik


def _rozmiar(sciezka):
    img = cv2.imread(str(sciezka), cv2.IMREAD_REDUCED_COLOR_8)
    if img is None:
        return None
    return img.shape[1] * 8, img.shape[0] * 8


# --------------------------------------------------------------- WeedsGalore

def weedsgalore(katalog, tax, splity=None, uzyj_instancji=True):
    """WeedsGalore: kanaly w osobnych plikach, maski semantyczne + instancji.

    grupa = dzien lotu. UWAGA: to jedno pole sfotografowane cztery razy, wiec
    ten sam fragment terenu wystepuje we wszystkich dniach. Podzial po dniach
    NIE gwarantuje rozlacznosci przestrzennej - patrz README, krok 6.
    """
    katalog = Path(katalog)
    podzial = _splity_weedsgalore(katalog, splity)
    out = []

    for dzien in sorted(p for p in katalog.iterdir()
                        if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)):
        for r_band in sorted((dzien / "images").glob("*_R.png")):
            oid = r_band.name[:-len("_R.png")]
            sem_p = dzien / "semantics" / f"{oid}.png"
            if not sem_p.exists():
                continue

            sem = cv2.imread(str(sem_p), cv2.IMREAD_UNCHANGED)
            if sem is None:
                continue
            if sem.ndim == 3:
                sem = sem[:, :, 0]
            h, w = sem.shape[:2]

            ins = None
            if uzyj_instancji:
                ins_p = dzien / "instances" / f"{oid}.png"
                if ins_p.exists():
                    ins = cv2.imread(str(ins_p), cv2.IMREAD_UNCHANGED)
                    if ins is not None and ins.ndim == 3:
                        # numer instancji zakodowany w RGB
                        ins = ((ins[:, :, 0].astype(np.int64) << 16)
                               | (ins[:, :, 1].astype(np.int64) << 8) | ins[:, :, 2])

            obiekty = []
            for wart, x0, y0, x1, y1, pole in ramki_z_maski(sem, ins):
                klucz = tax.z_maski("weedsgalore", wart)
                if klucz is None:
                    tax.nieznane[f"weedsgalore:maska={wart}"] += 1
                    continue
                obiekty.append(LabeledObject(klasa=klucz, xyxy=[x0, y0, x1, y1],
                                      zrodlowa_nazwa=f"maska={wart}", pole_maski=pole))

            out.append(Record(
                id=f"weedsgalore/{dzien.name}/{oid}",
                obraz=str(r_band)[:-len("_R.png")],   # prefiks - kanaly dokladane przy eksporcie
                szer=w, wys=h, zrodlo="weedsgalore",
                grupa=f"weedsgalore_{dzien.name}", domena="uav",
                agl_m=5.0, gsd_mm=2.5, wyczerpujace=True,
                split=podzial.get(oid), obiekty=obiekty))
    return out


def _splity_weedsgalore(katalog, splity):
    """Oficjalny podzial autorow z plikow splits/*.txt, jesli dostepny."""
    for kand in (splity, katalog / "splits"):
        if kand and Path(kand).is_dir():
            wynik = {}
            for nazwa in ("train", "val", "test"):
                p = Path(kand) / f"{nazwa}.txt"
                if p.exists():
                    for linia in p.read_text().split():
                        wynik[linia.strip()] = nazwa
            if wynik:
                return wynik
    return {}


# --------------------------------------------------------------- CoFly

_GPS = re.compile(r"Lat=([-\d.]+),Lon=([-\d.]+),Alt=([-\d.]+)")
_KOLORY = {(255, 0, 0): 1, (0, 0, 255): 2, (255, 255, 0): 3}   # RGB -> wartosc labels_1d


def cofly(katalog, tax):
    """CoFly: 366 zdjec, 201 z maskami. Reszta to PRZYKLADY NEGATYWNE.

    wyczerpujace=False, bo zbior adnotuje tylko trzy gatunki chwastow,
    a bawelne i reszte roslinnosci traktuje jako tlo.
    """
    katalog = Path(katalog)
    out = []
    for p in sorted((katalog / "images").glob("*")):
        if p.suffix not in OBRAZY:
            continue
        maska = _maska_cofly(katalog, p.name)
        rozm = _rozmiar(p)
        if rozm is None:
            continue
        w, h = rozm
        if maska is not None:
            h, w = maska.shape[:2]

        meta = _GPS.search(p.name)
        alt = float(meta.group(3)) if meta else 5.0

        obiekty = []
        if maska is not None:
            for wart, x0, y0, x1, y1, pole in ramki_z_maski(maska):
                klucz = tax.z_maski("cofly", wart)
                if klucz is None:
                    tax.nieznane[f"cofly:maska={wart}"] += 1
                    continue
                obiekty.append(LabeledObject(klasa=klucz, xyxy=[x0, y0, x1, y1],
                                      zrodlowa_nazwa=f"maska={wart}", pole_maski=pole))

        oid = re.match(r"(ID_\d+)", p.name)
        out.append(Record(
            id=f"cofly/{oid.group(1) if oid else p.stem}",
            obraz=str(p), szer=w, wys=h, zrodlo="cofly",
            grupa="cofly_larisa", domena="uav", agl_m=alt,
            wyczerpujace=False, obiekty=obiekty))
    return out


def _maska_cofly(katalog, nazwa):
    p1 = katalog / "labels_1d" / nazwa
    if p1.exists():
        m = cv2.imread(str(p1), cv2.IMREAD_UNCHANGED)
        if m is not None:
            return m[:, :, 0] if m.ndim == 3 else m
    p2 = katalog / "labels" / nazwa
    if p2.exists():
        kol = cv2.imread(str(p2))
        if kol is not None:
            kol = cv2.cvtColor(kol, cv2.COLOR_BGR2RGB)
            m = np.zeros(kol.shape[:2], np.uint8)
            for rgb, wart in _KOLORY.items():
                # tolerancja - kompresja zmienia kolory na krawedziach
                m[np.all(np.abs(kol.astype(int) - rgb) < 40, axis=2)] = wart
            return m
    return None


# --------------------------------------------------------------- Pascal VOC

OGOLNE_KATALOGI = {"annotations", "annotation", "labels", "xml", "xmls", "anno", "bounding_boxes"}


def pascal_voc(katalog, tax, zrodlo, domena="ground", agl_m=None, grupa_z_katalogu=True):
    """Wspolny parser dla Sudarsa i WeedMaize (oba w Pascal VOC).

    Rozne rzeczy do obslugi: czesc XML-i ma zerowy rozmiar (bierzemy z pliku
    obrazu), zdjecia leza w roznych miejscach wzgledem XML-a.
    """
    katalog = Path(katalog)
    out = []
    for x in sorted(katalog.rglob("*.xml")):
        try:
            root = ET.parse(x).getroot()
        except ET.ParseError:
            log.warning("uszkodzony XML: %s", x)
            continue

        size = root.find("size")
        w = int(float(size.findtext("width", "0"))) if size is not None else 0
        h = int(float(size.findtext("height", "0"))) if size is not None else 0
        plik = root.findtext("filename") or x.stem + ".jpg"

        sciezka = next((c for c in (x.with_name(plik),
                                    x.parent.parent / "raw images" / plik,
                                    x.parent.parent / "images" / plik) if c.exists()), None)
        if sciezka is None:
            znal = list(katalog.rglob(plik))
            sciezka = znal[0] if znal else None
        if sciezka is None:
            # nazwa w XML z innym rozszerzeniem (np. .jpg zamiast .png) - szukamy po samej nazwie
            for kat in (x.parent, x.parent.parent / "images", x.parent.parent / "raw images"):
                znal = [c for c in kat.glob(Path(plik).stem + ".*") if c.suffix in OBRAZY] if kat.is_dir() else []
                znal += [c for c in kat.glob(x.stem + ".*") if c.suffix in OBRAZY] if kat.is_dir() else []
                if znal:
                    sciezka = znal[0]
                    break
        if (not w or not h) and sciezka is not None:
            rozm = _rozmiar(sciezka)
            if rozm:
                w, h = rozm
        if not w or not h or sciezka is None:
            log.warning("pomijam %s - brak rozmiaru albo zdjecia", x.name)
            continue

        obiekty = []
        for ob in root.findall("object"):
            bb = ob.find("bndbox")
            if bb is None:
                continue
            nazwa = ob.findtext("name", "")
            klucz = tax.klucz(nazwa)
            if klucz is None:
                continue                       # trafilo do raportu nieznanych
            obiekty.append(LabeledObject(
                klasa=klucz,
                xyxy=[float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                      float(bb.findtext("xmax")), float(bb.findtext("ymax"))],
                zrodlowa_nazwa=nazwa, faza=tax.faza(nazwa)))

        # grupa: katalog nadrzedny (train_set, validation_set, dzien sesji...)
        # Ogolna nazwa folderu ("annotations") nie niesie informacji o sesji -
        # wtedy kazde zdjecie jest osobna grupa. Inaczej caly zbior bylby jedna
        # grupa i w calosci trafilby do jednej czesci podzialu.
        if not grupa_z_katalogu:
            grupa = zrodlo
        elif x.parent.name.lower() in OGOLNE_KATALOGI:
            grupa = f"{zrodlo}_{x.stem}"
        else:
            grupa = f"{zrodlo}_{x.parent.name}"
        out.append(Record(id=f"{zrodlo}/{x.stem}", obraz=str(sciezka), szer=w, wys=h,
                          zrodlo=zrodlo, grupa=grupa, domena=domena, agl_m=agl_m,
                          wyczerpujace=True, obiekty=obiekty))
    return out


def sudars(katalog, tax):
    """Sudars: NIE jest zbiorem dronowym - zdjecia z 30 cm i z platformy kolowej."""
    return pascal_voc(katalog, tax, "sudars", domena="ground")


def weedmaize(katalog, tax):
    """WeedMaize: zdjecia zenitalne z 1,3 m, kody EPPO z faza wzrostu."""
    return pascal_voc(katalog, tax, "weedmaize", domena="ground", agl_m=1.3)


# --------------------------------------------------------------- dane wlasne

def _klasy_yolo_z_katalogu(katalog):
    """Nazwy klas dla etykiet YOLO: classes.txt albo data.yaml obok danych.

    W formacie YOLO etykieta to indeks liczbowy (0, 1, 2...), a nazwy sa w
    osobnym pliku. Bez niego kazda klasa trafialaby do raportu jako
    nierozpoznana "0", "1" itd.
    """
    for p in sorted(katalog.rglob("classes.txt")):
        nazwy = [l.strip() for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]
        if nazwy:
            return nazwy
    for p in sorted(katalog.rglob("data.yaml")):
        try:
            import yaml
            dane = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        n = dane.get("names")
        if isinstance(n, dict):
            return [n[k] for k in sorted(n, key=lambda x: int(x))]
        if isinstance(n, list) and n:
            return n
    return None


def wlasne(katalog, tax, klasy_yolo=None, domena="uav", agl_m=None, grupa=None):
    """Dane z wlasnej kapsuly: format YOLO (.txt) albo Pascal VOC (.xml).

    grupa domyslnie z nazwy podkatalogu - jeden podkatalog na misje.
    To wazne: podzial train/val/test idzie wlasnie po grupach.
    """
    katalog = Path(katalog)
    if klasy_yolo is None:
        klasy_yolo = _klasy_yolo_z_katalogu(katalog)
        if klasy_yolo:
            log.info("wlasne: nazwy klas z pliku obok etykiet: %s", klasy_yolo)
    if any(katalog.rglob("*.xml")):
        rek = pascal_voc(katalog, tax, "wlasne", domena=domena, agl_m=agl_m)
        if grupa:
            for r in rek:
                r.grupa = grupa
        return rek

    out = []
    for txt in sorted(katalog.rglob("*.txt")):
        if txt.name in ("classes.txt", "data.yaml"):
            continue
        obraz = next((txt.with_suffix(e) for e in (".jpg", ".png", ".jpeg")
                      if txt.with_suffix(e).exists()), None)
        if obraz is None:
            # uklad images/ + labels/
            kand = [p for p in katalog.rglob(txt.stem + ".*") if p.suffix in OBRAZY]
            obraz = kand[0] if kand else None
        if obraz is None:
            continue
        rozm = _rozmiar(obraz)
        if rozm is None:
            continue
        w, h = rozm

        obiekty = []
        for linia in txt.read_text().split("\n"):
            czesci = linia.split()
            if len(czesci) < 5:
                continue
            idx = int(float(czesci[0]))
            cx, cy, bw, bh = (float(v) for v in czesci[1:5])
            nazwa = klasy_yolo[idx] if klasy_yolo and idx < len(klasy_yolo) else str(idx)
            klucz = tax.klucz(nazwa)
            if klucz is None:
                continue
            # YOLO trzyma wartosci znormalizowane - manifest wymaga pikseli
            obiekty.append(LabeledObject(
                klasa=klucz, zrodlowa_nazwa=nazwa,
                xyxy=[(cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h]))

        out.append(Record(id=f"wlasne/{txt.stem}", obraz=str(obraz), szer=w, wys=h,
                          zrodlo="wlasne", grupa=grupa or f"wlasne_{txt.parent.name}",
                          domena=domena, agl_m=agl_m, wyczerpujace=True, obiekty=obiekty))
    return out


def ewis1(katalog, tax, z_sorgo=False, test_dni=None):
    """EWIS1 (Mendeley 2026): dron DJI Mavic 2 Pro, 5 m nad polem, kukurydza i sorgo.

    Uklad: bounding_boxes/*.xml (Pascal VOC, ramki z masek autorow), images/*.png,
    masks/*.png. Katalog additional_images/ to nieoznaczone loty - pomijany.

    Domyslnie TYLKO kukurydza (pliki z "Maize" w nazwie). Mlode sorgo wyglada jak
    trawa, a w WeedsGalore trawa (chwastnica) jest chwastem - sorgo jako tlo mogloby
    nauczyc model pomijac trawy. Pliki bez nazwy uprawy (img_2021...) pochodza
    najpewniej z pierwszej czesci projektu EWIS, poswieconej sorgo - traktowane jak sorgo.
    GSD ok. 1,18 mm/px (matryca 13,2 mm, 5472 px, ogniskowa 10,26 mm, 5 m).

    test_dni: dni lotu (RRRRMMDD) odlozone jako test - reszta do treningu. Grupa
    to pole + dzien lotu, wiec zdjecia jednego lotu nie trafiaja do dwoch czesci.
    """
    katalog = Path(katalog)
    bb = katalog / "bounding_boxes"
    rek = pascal_voc(bb if bb.is_dir() else katalog, tax, "ewis1", domena="uav", agl_m=5.0)
    if not z_sorgo:
        przed = len(rek)
        rek = [r for r in rek if "maize" in Path(r.obraz).name.lower()]
        log.info("ewis1: tylko kukurydza - %d zdjec, pominieto %d (sorgo i bez nazwy uprawy)",
                 len(rek), przed - len(rek))
    uklady = collections.Counter()
    test_dni = {str(d) for d in (test_dni or [])}
    for r in rek:
        r.gsd_mm = 1.18
        m = re.match(r"([A-Za-z]+)_(20\d{6})", Path(r.obraz).name)
        if m:
            r.grupa = f"ewis1_{m.group(1)}_{m.group(2)}"
        if test_dni:
            r.split = "test" if m and m.group(2) in test_dni else "train"
        uklady[_ewis1_uklad(r, katalog / "masks" / Path(r.obraz).name)] += 1
    log.info("ewis1: uklad ramek wzgledem zdjecia: %s", dict(uklady))
    return rek


def _ewis1_przeksztalc(xyxy, typ, w_xml, h_xml):
    """Ramka z ukladu XML do ukladu zdjecia: bez zmian, zamiana osi albo obrot o 90 st."""
    x0, y0, x1, y1 = xyxy
    if typ == "bez_zmian":
        return [x0, y0, x1, y1]
    if typ == "zamiana_osi":
        return [y0, x0, y1, x1]
    if typ == "obrot_prawo":            # (x, y) -> (h - y, x)
        return [h_xml - y1, x0, h_xml - y0, x1]
    if typ == "obrot_lewo":             # (x, y) -> (y, w - x)
        return [y0, w_xml - x1, y1, w_xml - x0]
    raise ValueError(typ)


def _ewis1_uklad(r, maska_p, probka=60):
    """Rozmiar ze zdjecia (w XML EWIS1 szerokosc i wysokosc sa zamienione), a uklad
    ramek ten, ktory trafia w rosliny na masce autorow (maski sa w ukladzie zdjecia).

    Zwraca nazwe wybranego ukladu. Ramki sa juz przeksztalcone w rekordzie.
    """
    try:
        from PIL import Image
        with Image.open(r.obraz) as im:              # dokladny rozmiar z naglowka, bez dekodowania
            rozm = im.size
    except Exception:
        rozm = _rozmiar(r.obraz)
    if not rozm:
        return "brak_zdjecia"
    W, H = rozm
    w_xml, h_xml = r.szer, r.wys
    r.szer, r.wys = W, H
    if (w_xml, h_xml) == (W, H):
        kandydaci = ["bez_zmian"]
    else:
        kandydaci = ["bez_zmian", "zamiana_osi", "obrot_prawo", "obrot_lewo"]
    maska = (cv2.imread(str(maska_p), cv2.IMREAD_REDUCED_COLOR_4)
             if len(kandydaci) > 1 and maska_p.exists() else None)
    if maska is None:
        wybor = "bez_zmian"
        if len(kandydaci) > 1:
            log.warning("ewis1 %s: rozmiar w XML inny niz zdjecia, a brak maski do sprawdzenia",
                        Path(r.obraz).name)
    else:
        zajete = maska.max(axis=2) > 0
        hm, wm = zajete.shape
        sk = wm / W
        obiekty = r.obiekty[:: max(1, len(r.obiekty) // probka)][:probka]

        def trafienia(typ):
            n = 0
            for o in obiekty:
                x0, y0, x1, y1 = _ewis1_przeksztalc(o.xyxy, typ, w_xml, h_xml)
                if x0 < 0 or y0 < 0 or x1 > W + 1 or y1 > H + 1:
                    continue
                a0, b0 = int(x0 * sk), int(y0 * sk)
                a1, b1 = max(a0 + 1, int(x1 * sk)), max(b0 + 1, int(y1 * sk))
                if zajete[b0:b1, a0:a1].mean() > 0.15:
                    n += 1
            return n
        wyniki = {t: trafienia(t) for t in kandydaci}
        wybor = max(wyniki, key=wyniki.get)
        if obiekty and wyniki[wybor] < 0.6 * len(obiekty):
            log.warning("ewis1 %s: tylko %d/%d ramek trafia w maske (%s)", Path(r.obraz).name,
                        wyniki[wybor], len(obiekty), wyniki)
    for o in r.obiekty:
        o.xyxy = _ewis1_przeksztalc(o.xyxy, wybor, w_xml, h_xml)
    r.obiekty = [o for o in r.obiekty
                 if o.xyxy[0] >= -1 and o.xyxy[1] >= -1 and o.xyxy[2] <= W + 1 and o.xyxy[3] <= H + 1]
    return wybor


def phenobench(katalog, tax, min_pole=15):
    """PhenoBench (Bonn): dron, burak cukrowy, trzy fazy wzrostu, maski kazdej rosliny.

    Uzywa train/ i val/ (test nie ma etykiet). Ramki z plant_instances (16-bit,
    kazda roslina osobno), klasa z semantics (1/3 burak, 2/4 chwast).
    Data zdjecia, jesli jest na poczatku nazwy (np. 05-15_...), trafia do id.
    """
    katalog = Path(katalog)
    out = []
    for czesc in ("train", "val"):
        obr_kat = katalog / czesc / "images"
        if not obr_kat.is_dir():
            continue
        for p in sorted(obr_kat.glob("*.png")):
            sem_p = katalog / czesc / "semantics" / p.name
            ins_p = katalog / czesc / "plant_instances" / p.name
            if not sem_p.exists() or not ins_p.exists():
                continue
            sem = cv2.imread(str(sem_p), cv2.IMREAD_UNCHANGED)
            ins = cv2.imread(str(ins_p), cv2.IMREAD_UNCHANGED)
            if sem is None or ins is None:
                continue
            if sem.ndim == 3:
                sem = sem[:, :, 0]
            if ins.ndim == 3:
                ins = ins[:, :, 0]
            h, w = sem.shape[:2]
            obiekty = []
            for wart, x0, y0, x1, y1, pole in ramki_z_maski(sem.astype(np.int64), ins, min_pole):
                klucz = tax.z_maski("phenobench", wart)
                if klucz is None:
                    tax.nieznane[f"phenobench:maska={wart}"] += 1
                    continue
                obiekty.append(LabeledObject(klasa=klucz, xyxy=[x0, y0, x1, y1],
                                             zrodlowa_nazwa=f"maska={wart}", pole_maski=pole))
            m = re.match(r"(\d{2}-\d{2})", p.name)
            out.append(Record(id=f"phenobench/{czesc}/{p.stem}", obraz=str(p), szer=w, wys=h,
                              zrodlo="phenobench", grupa=f"phenobench_{czesc}_{m.group(1) if m else p.stem}",
                              domena="uav", wyczerpujace=True, obiekty=obiekty))
    return out


ADAPTERY = {
    "weedsgalore": weedsgalore,
    "ewis1": ewis1,
    "phenobench": phenobench,
    "cofly": cofly,
    "sudars": sudars,
    "weedmaize": weedmaize,
    "wlasne": wlasne,
}