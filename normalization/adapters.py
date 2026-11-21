"""Adaptery: kazdy zbior zrodlowy -> manifest (lista Record).

Jeden adapter na zbior. Reszta potoku (deduplikacja, podzial, kafelkowanie,
eksport) nie wie juz, skad dane pochodza.

Formaty, z ktorymi trzeba sobie poradzic:

  WeedsGalore  kanaly w OSOBNYCH plikach (_R, _G, _B, _NIR, _RE),
               maski semantyczne + instancji
  EWIS1        Pascal VOC (XML) + maski; w XML zamienione wymiary, czasem obrocone ramki
  Sudars       Pascal VOC (XML), rozne rozdzielczosci
  wlasne       YOLO albo Pascal VOC

Model koncowy (scripts/final_model.sh) uzywa trzech:
    weedsgalore()  -> zbior_lotnicze
    sudars()       -> zbior_naziemne   (przez pascal_voc())
    ewis1()        -> zbior_ewis1_podzial (przez pascal_voc() + poprawki ukladu ramek)
wlasne() - na przyszle dane z kapsuly.

Kazdy adapter: (katalog, tax, opcje) -> lista Record z ramkami w pikselach zdjecia.

Uzywany w:
    build.py zbuduj() - przez slownik ADAPTERY (na dole pliku)
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

OBRAZY = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}   # rozszerzenia zdjec


# --------------------------------------------------------------- narzedzia

def ramki_z_maski(maska, instancje=None, min_pole=15):
    """Maska -> lista (wartosc, x0, y0, x1, y1, pole).

    Z maska instancji kazda roslina ma wlasny numer, wiec dwie stykajace sie
    rosliny tego samego gatunku daja dwie ramki. Bez niej - spojne skladowe,
    czyli stykajace sie rosliny zleja sie w jedna ramke.

    Po co: WeedsGalore nie ma ramek, tylko maski - ramki trzeba z nich wyliczyc.

    Argumenty:
        maska     - numpy (h, w), wartosc piksela = klasa (0 = tlo), np. 4 = zoltlica
        instancje - numpy (h, w), numer rosliny (0 = tlo) albo None
        min_pole  - minimalna liczba pikseli obiektu; mniejsze odpadaja (szum), np. 15

    Zwraca:
        lista krotek (wartosc_maski, x0, y0, x1, y1, pole_px), np. [(4, 10, 10, 30, 30, 400)]

    Uzywana w:
        weedsgalore() (ten plik)
    """
    wynik = []
    if instancje is not None:
        for nr in np.unique(instancje):          # kazdy numer rosliny osobno
            if nr == 0:
                continue
            ys, xs = np.nonzero(instancje == nr) # wspolrzedne pikseli tej rosliny
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

    # bez instancji: dla kazdej klasy osobno szukamy spojnych plam pikseli
    for w in np.unique(maska):
        if w == 0:
            continue
        n, _, st, _ = cv2.connectedComponentsWithStats((maska == w).astype(np.uint8), 8)
        for i in range(1, n):                    # 0 = tlo
            x, y, bw, bh, pole = st[i]
            if pole >= min_pole:
                wynik.append((int(w), int(x), int(y), int(x + bw), int(y + bh), int(pole)))
    return wynik


def _rozmiar(sciezka):
    """Szybki rozmiar zdjecia (szer, wys) - wczytuje je zmniejszone 8x.

    Po co: rozmiar jest potrzebny, gdy XML go nie podaje, a pelne wczytanie duzego
           zdjecia trwa dlugo. Wynik jest dokladny co do 8 px.

    Argumenty:
        sciezka - plik zdjecia, np. "dane/sudars/dataset/img1.jpg"

    Zwraca:
        (szer, wys) w px albo None, gdy nie da sie wczytac, np. (4000, 3000)

    Uzywana w:
        pascal_voc(), wlasne(), _ewis1_uklad() (ten plik)
    """
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

    Po co: glowny zbior lotniczy modelu koncowego (trening, walidacja i test).

    Argumenty:
        katalog        - folder zbioru z podfolderami dni RRRR-MM-DD, np. "dane/weedsgalore-dataset"
        tax            - Taxonomy (wartosc piksela maski -> gatunek)
        splity         - folder z train.txt/val.txt/test.txt autorow albo None (szuka katalog/splits)
        uzyj_instancji - True = ramka na kazda rosline osobno (maska instances/)

    Zwraca:
        lista Record: obraz = prefiks bez _R.png, split z oficjalnego podzialu,
        grupa np. "weedsgalore_2023-05-25", domena "uav", 2,5 mm/px

    Uzywana w:
        build.py zbuduj() przez ADAPTERY["weedsgalore"] (--weedsgalore)
    """
    katalog = Path(katalog)
    podzial = _splity_weedsgalore(katalog, splity)   # {nazwa zdjecia: "train"/"val"/"test"}
    out = []

    for dzien in sorted(p for p in katalog.iterdir()
                        if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)):
        for r_band in sorted((dzien / "images").glob("*_R.png")):   # jedno zdjecie = jeden plik _R
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
                    tax.nieznane[f"weedsgalore:maska={wart}"] += 1   # do raportu nieznanych
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
    """Oficjalny podzial autorow z plikow splits/*.txt, jesli dostepny.

    Argumenty:
        katalog - folder WeedsGalore (szuka w nim splits/)
        splity  - inny folder z plikami podzialu albo None

    Zwraca:
        dict nazwa_zdjecia -> czesc, np. {"2023-05-25_0012": "test"}; {} = brak plikow

    Uzywana w:
        weedsgalore() (ten plik)
    """
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


# --------------------------------------------------------------- Pascal VOC

# nazwy folderow z XML, ktore nic nie mowia o sesji/dniu - wtedy kazde zdjecie to osobna grupa
OGOLNE_KATALOGI = {"annotations", "annotation", "labels", "xml", "xmls", "anno", "bounding_boxes"}


def pascal_voc(katalog, tax, zrodlo, domena="ground", agl_m=None, grupa_z_katalogu=True):
    """Wspolny parser dla zbiorow w Pascal VOC (Sudars, EWIS1, wlasne).

    Rozne rzeczy do obslugi: czesc XML-i ma zerowy rozmiar (bierzemy z pliku
    obrazu), zdjecia leza w roznych miejscach wzgledem XML-a.

    Po co: jeden czytnik XML dla wszystkich zbiorow w formacie Pascal VOC
           (Sudars, EWIS1, wlasne).

    Argumenty:
        katalog  - folder, w ktorym (rekurencyjnie) leza pliki .xml
        tax      - Taxonomy (nazwa z <name> -> gatunek)
        zrodlo   - nazwa zbioru do id i grupy, np. "sudars"
        domena   - "ground" albo "uav"
        agl_m    - wysokosc w m albo None
        grupa_z_katalogu - True = grupa z nazwy folderu XML; False = caly zbior jedna grupa

    Zwraca:
        lista Record, np. Record(id="sudars/img1", grupa="sudars_train_set", obiekty=[...])
        (zdjecia bez rozmiaru albo bez pliku obrazu sa pomijane z ostrzezeniem)

    Uzywana w:
        sudars(), ewis1(), wlasne() (ten plik)
    """
    katalog = Path(katalog)
    out = []
    for x in sorted(katalog.rglob("*.xml")):            # wszystkie XML, tez w podfolderach
        try:
            root = ET.parse(x).getroot()
        except ET.ParseError:
            log.warning("uszkodzony XML: %s", x)
            continue

        # rozmiar i nazwa zdjecia z XML (rozmiar bywa 0 - wtedy z pliku nizej)
        size = root.find("size")
        w = int(float(size.findtext("width", "0"))) if size is not None else 0
        h = int(float(size.findtext("height", "0"))) if size is not None else 0
        plik = root.findtext("filename") or x.stem + ".jpg"

        # szukanie zdjecia: obok XML, w ../raw images, w ../images, potem w calym zbiorze
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

        # obiekty: <object><name>weed</name><bndbox>xmin ymin xmax ymax</bndbox></object>
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
    """Sudars: NIE jest zbiorem dronowym - zdjecia z 30 cm i z platformy kolowej.

    Po co: duzo przykladow chwastow z bliska - tylko do treningu modelu koncowego
           (walidacja i test sa z WeedsGalore).

    Argumenty:
        katalog - folder zbioru, np. "dane/sudars/dataset"
        tax     - Taxonomy

    Zwraca:
        lista Record (domena "ground"), bez splitu - dzieli go potem steps.podziel() 70/15/15

    Uzywana w:
        build.py zbuduj() przez ADAPTERY["sudars"] (--sudars)
    """
    return pascal_voc(katalog, tax, "sudars", domena="ground")


# --------------------------------------------------------------- dane wlasne

def _klasy_yolo_z_katalogu(katalog):
    """Nazwy klas dla etykiet YOLO: classes.txt albo data.yaml obok danych.

    W formacie YOLO etykieta to indeks liczbowy (0, 1, 2...), a nazwy sa w
    osobnym pliku. Bez niego kazda klasa trafialaby do raportu jako
    nierozpoznana "0", "1" itd.

    Argumenty:
        katalog - folder z danymi wlasnymi

    Zwraca:
        lista nazw klas, np. ["chwast", "kukurydza"], albo None

    Uzywana w:
        wlasne() (ten plik)
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

    Po co: gotowe wejscie na zdjecia z naszego drona, gdy beda oznaczone.

    Argumenty:
        katalog    - folder danych (podfolder na misje)
        tax        - Taxonomy
        klasy_yolo - nazwy klas YOLO albo None (szuka classes.txt / data.yaml)
        domena     - "uav" albo "ground"
        agl_m      - wysokosc lotu w m albo None
        grupa      - jedna grupa dla wszystkiego albo None (z nazwy podfolderu)

    Zwraca:
        lista Record (ramki YOLO zamienione na piksele)

    Uzywana w:
        build.py przez ADAPTERY["wlasne"] (--wlasne) - nie model koncowy
    """
    katalog = Path(katalog)
    if klasy_yolo is None:
        klasy_yolo = _klasy_yolo_z_katalogu(katalog)
        if klasy_yolo:
            log.info("wlasne: nazwy klas z pliku obok etykiet: %s", klasy_yolo)
    if any(katalog.rglob("*.xml")):                 # sa XML -> to Pascal VOC
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


# --------------------------------------------------------------- EWIS1

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

    Po co: drugi zbior lotniczy (inne pole, inna kamera) - trening modelu koncowego
           i niezalezny test na dniu 20220516.

    Argumenty:
        katalog  - folder EWIS1, np. "dane/EWIS1 Maize, Sorghum & Weed Dataset"
        tax      - Taxonomy
        z_sorgo  - True = tez sorgo i zdjecia bez nazwy uprawy
        test_dni - lista dni RRRRMMDD do testu, np. ["20220516"]; None = split ustawi podzial

    Zwraca:
        lista Record: grupa np. "ewis1_Loew_20220512", split "test"/"train" (gdy test_dni),
        rozmiar ze zdjecia, ramki w ukladzie zdjecia

    Uzywana w:
        build.py zbuduj() przez ADAPTERY["ewis1"] (--ewis1, --ewis1-test-dni, --ewis1-z-sorgo)
    """
    katalog = Path(katalog)
    bb = katalog / "bounding_boxes"
    rek = pascal_voc(bb if bb.is_dir() else katalog, tax, "ewis1", domena="uav", agl_m=5.0)
    if not z_sorgo:
        przed = len(rek)
        rek = [r for r in rek if "maize" in Path(r.obraz).name.lower()]
        log.info("ewis1: tylko kukurydza - %d zdjec, pominieto %d (sorgo i bez nazwy uprawy)",
                 len(rek), przed - len(rek))
    uklady = collections.Counter()               # ile zdjec mialo ktory uklad ramek (do logu)
    test_dni = {str(d) for d in (test_dni or [])}
    for r in rek:
        r.gsd_mm = 1.18
        # nazwa np. Loew_20220512_Maize_010.png -> pole "Loew", dzien "20220512"
        m = re.match(r"([A-Za-z]+)_(20\d{6})", Path(r.obraz).name)
        if m:
            r.grupa = f"ewis1_{m.group(1)}_{m.group(2)}"
        if test_dni:
            r.split = "test" if m and m.group(2) in test_dni else "train"
        # poprawka rozmiaru i ukladu ramek (blad w XML EWIS1) - zmienia r w miejscu
        uklady[_ewis1_uklad(r, katalog / "masks" / Path(r.obraz).name)] += 1
    log.info("ewis1: uklad ramek wzgledem zdjecia: %s", dict(uklady))
    return rek


def _ewis1_przeksztalc(xyxy, typ, w_xml, h_xml):
    """Ramka z ukladu XML do ukladu zdjecia: bez zmian, zamiana osi albo obrot o 90 st.

    Argumenty:
        xyxy  - ramka z XML [x0, y0, x1, y1], np. [10, 20, 40, 60]
        typ   - "bez_zmian", "zamiana_osi", "obrot_prawo" albo "obrot_lewo"
        w_xml, h_xml - rozmiar zapisany w XML, np. 3648, 5472

    Zwraca:
        lista [x0, y0, x1, y1] w ukladzie zdjecia

    Uzywana w:
        _ewis1_uklad() (ten plik) - sprawdzanie kandydatow i przeksztalcenie ramek
    """
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

    Jak dziala: gdy rozmiar w XML rozni sie od zdjecia, sprawdza 4 mozliwe uklady
    na probce ramek i wybiera ten, w ktorym najwiecej ramek trafia w rosliny na masce.

    Argumenty:
        r       - Record z adaptera (szer/wys i ramki jeszcze z XML) - zmieniany w miejscu
        maska_p - sciezka maski autorow, np. ".../masks/Loew_20220512_Maize_010.png"
        probka  - ile ramek sprawdzic, np. 60

    Zwraca:
        str - wybrany uklad, np. "obrot_prawo"; "brak_zdjecia", gdy nie da sie odczytac rozmiaru

    Uzywana w:
        ewis1() (ten plik)
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
    r.szer, r.wys = W, H                             # rozmiar zawsze ze zdjecia
    if (w_xml, h_xml) == (W, H):
        kandydaci = ["bez_zmian"]
    else:
        kandydaci = ["bez_zmian", "zamiana_osi", "obrot_prawo", "obrot_lewo"]
    maska = (cv2.imread(str(maska_p), cv2.IMREAD_REDUCED_COLOR_4)   # maska zmniejszona 4x - szybciej
             if len(kandydaci) > 1 and maska_p.exists() else None)
    if maska is None:
        wybor = "bez_zmian"
        if len(kandydaci) > 1:
            log.warning("ewis1 %s: rozmiar w XML inny niz zdjecia, a brak maski do sprawdzenia",
                        Path(r.obraz).name)
    else:
        zajete = maska.max(axis=2) > 0               # True tam, gdzie na masce jest roslina
        hm, wm = zajete.shape
        sk = wm / W                                  # skala zdjecie -> zmniejszona maska
        obiekty = r.obiekty[:: max(1, len(r.obiekty) // probka)][:probka]   # rownomierna probka ramek

        def trafienia(typ):
            """Ile ramek z probki po przeksztalceniu typ trafia w rosline (>15% pikseli ramki).

            Argumenty: typ - nazwa ukladu, np. "obrot_prawo"
            Zwraca: int, np. 58
            Uzywana w: _ewis1_uklad() - wybor najlepszego ukladu
            """
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
        wybor = max(wyniki, key=wyniki.get)          # uklad z najwieksza liczba trafien
        if obiekty and wyniki[wybor] < 0.6 * len(obiekty):
            log.warning("ewis1 %s: tylko %d/%d ramek trafia w maske (%s)", Path(r.obraz).name,
                        wyniki[wybor], len(obiekty), wyniki)
    for o in r.obiekty:
        o.xyxy = _ewis1_przeksztalc(o.xyxy, wybor, w_xml, h_xml)
    # ramki wyraznie poza zdjeciem (tolerancja 1 px) odpadaja
    r.obiekty = [o for o in r.obiekty
                 if o.xyxy[0] >= -1 and o.xyxy[1] >= -1 and o.xyxy[2] <= W + 1 and o.xyxy[3] <= H + 1]
    return wybor


# nazwa opcji w build.py (--weedsgalore, --ewis1, ...) -> funkcja adaptera.
# build.py robi z tego opcje terminala i wola ADAPTERY[nazwa](sciezka, tax, **opcje).
ADAPTERY = {
    "weedsgalore": weedsgalore,
    "ewis1": ewis1,
    "sudars": sudars,
    "wlasne": wlasne,
}
