"""Ocena modelu: metryki per klasa i osobno dla malych obiektow.

Dlaczego nie wystarczy samo mAP:

  Zbiory sa niezbalansowane (w Sudarsie chwastow jest wielokrotnie wiecej niz
  upraw), wiec srednia zaciera to, ze model moze zupelnie nie radzic sobie
  z rzadka klasa.

  Zadanie jest zdominowane przez obiekty male - chwast z pulapu ma kilkadziesiat
  pikseli. mAP liczone na wszystkich obiektach moze wygladac dobrze dzieki
  duzym roslinom, podczas gdy te najwazniejsze sa pomijane.

  Czulosc jest wazniejsza niz precyzja: pominiete ognisko to realna strata
  plonu, falszywy alarm to tylko zbedne sprawdzenie fragmentu pola.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from .device import wybierz

log = logging.getLogger(__name__)

# progi rozmiaru jak w COCO (pole ramki w pikselach)
MALE = 32 ** 2
SREDNIE = 96 ** 2


def _yolo():
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("Do oceny potrzebny pakiet ultralytics: pip install ultralytics") from e
    return YOLO


def ocen(model, dane, split="test", wejscie=640, urzadzenie=None, prog=0.001):
    """Standardowa ocena ultralytics - mAP ogolne i per klasa."""
    YOLO = _yolo()
    m = YOLO(str(model))
    wyniki = m.val(data=str(dane), split=split, imgsz=wejscie, conf=prog,
                   device=wybierz(urzadzenie), verbose=False)

    nazwy = list(getattr(wyniki, "names", {}).values()) or []
    per_klasa = {}
    try:
        for i, idx in enumerate(wyniki.box.ap_class_index):
            per_klasa[nazwy[int(idx)] if int(idx) < len(nazwy) else str(idx)] = {
                "precyzja": round(float(wyniki.box.p[i]), 4),
                "czulosc": round(float(wyniki.box.r[i]), 4),
                "mAP50": round(float(wyniki.box.ap50[i]), 4),
                "mAP50_95": round(float(wyniki.box.ap[i]), 4),
            }
    except Exception as e:                 # roznice miedzy wersjami ultralytics
        log.warning("nie udalo sie odczytac metryk per klasa: %s", e)

    return {
        "model": str(model),
        "dane": str(dane),
        "split": split,
        "wejscie": wejscie,
        "mAP50": round(float(wyniki.box.map50), 4),
        "mAP50_95": round(float(wyniki.box.map), 4),
        "precyzja": round(float(wyniki.box.mp), 4),
        "czulosc": round(float(wyniki.box.mr), 4),
        "per_klasa": per_klasa,
    }


# ---------------------------------------------------------------- male obiekty

def wczytaj_etykiety(katalog_etykiet, katalog_obrazow=None, rozmiar=None):
    """Etykiety YOLO -> {nazwa_pliku: [(klasa, x0, y0, x1, y1), ...]} w pikselach.

    rozmiar: (szer, wys), gdy wszystkie obrazy maja ten sam rozmiar (kafelki).
    Inaczej rozmiar odczytywany jest z plikow obrazow.
    """
    import cv2

    out = {}
    for txt in sorted(Path(katalog_etykiet).glob("*.txt")):
        if rozmiar:
            w, h = rozmiar
        else:
            obraz = None
            if katalog_obrazow:
                for e in (".jpg", ".jpeg", ".png"):
                    k = Path(katalog_obrazow) / (txt.stem + e)
                    if k.exists():
                        obraz = k
                        break
            if obraz is None:
                continue
            img = cv2.imread(str(obraz), cv2.IMREAD_REDUCED_COLOR_8)
            if img is None:
                continue
            h, w = img.shape[0] * 8, img.shape[1] * 8

        ramki = []
        for linia in txt.read_text(encoding="utf-8").split("\n"):
            cz = linia.split()
            if len(cz) < 5:
                continue
            k = int(float(cz[0]))
            cx, cy, bw, bh = (float(v) for v in cz[1:5])
            ramki.append((k, (cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h))
        out[txt.stem] = ramki
    return out


def rozklad_rozmiarow(etykiety, nazwy_klas=None):
    """Ile obiektow jest malych, srednich i duzych - oraz jak male naprawde sa.

    To jest liczba, ktora decyduje o tym, czy kafelkowanie jest konieczne.
    """
    import math

    kategorie = defaultdict(int)
    boki = []
    per_klasa = defaultdict(lambda: defaultdict(int))
    for ramki in etykiety.values():
        for k, x0, y0, x1, y1 in ramki:
            pole = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
            kat = "male" if pole < MALE else ("srednie" if pole < SREDNIE else "duze")
            kategorie[kat] += 1
            boki.append(math.sqrt(pole))
            nazwa = nazwy_klas[k] if nazwy_klas and k < len(nazwy_klas) else str(k)
            per_klasa[nazwa][kat] += 1

    boki.sort()
    n = len(boki)
    def pct(p):
        return round(boki[min(n - 1, int(p * n))], 1) if n else None

    return {
        "obiektow": n,
        "kategorie": dict(kategorie),
        "udzial_malych": round(kategorie["male"] / n, 3) if n else None,
        "bok_mediana_px": pct(0.5),
        "bok_p10_px": pct(0.1),
        "bok_p90_px": pct(0.9),
        "per_klasa": {k: dict(v) for k, v in per_klasa.items()},
    }


def ocen_male_obiekty(model, katalog_obrazow, katalog_etykiet, nazwy_klas,
                      wejscie=640, prog=0.25, iou=0.5, urzadzenie=None):
    """Czulosc osobno dla malych, srednich i duzych obiektow.

    Ultralytics podaje mAP ogolne; tutaj liczymy wprost, ile obiektow kazdej
    kategorii rozmiaru model znalazl. To jest metryka, ktora najlepiej mowi,
    czy kafelkowanie dziala.
    """
    YOLO = _yolo()
    m = YOLO(str(model))
    prawda = wczytaj_etykiety(katalog_etykiet, katalog_obrazow)

    znalezione = defaultdict(int)
    wszystkie = defaultdict(int)
    for stem, ramki in prawda.items():
        obraz = next((Path(katalog_obrazow) / (stem + e) for e in (".jpg", ".jpeg", ".png")
                      if (Path(katalog_obrazow) / (stem + e)).exists()), None)
        if obraz is None:
            continue
        wynik = m.predict(str(obraz), imgsz=wejscie, conf=prog,
                          device=wybierz(urzadzenie), verbose=False)[0]
        pred = [(int(b.cls), *[float(v) for v in b.xyxy[0]]) for b in wynik.boxes]

        for k, x0, y0, x1, y1 in ramki:
            pole = (x1 - x0) * (y1 - y0)
            kat = "male" if pole < MALE else ("srednie" if pole < SREDNIE else "duze")
            wszystkie[kat] += 1
            if any(pk == k and _iou((x0, y0, x1, y1), (px0, py0, px1, py1)) >= iou
                   for pk, px0, py0, px1, py1 in pred):
                znalezione[kat] += 1

    return {kat: {"obiektow": wszystkie[kat], "znalezionych": znalezione[kat],
                  "czulosc": round(znalezione[kat] / wszystkie[kat], 4) if wszystkie[kat] else None}
            for kat in ("male", "srednie", "duze")}


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    wsp = (x1 - x0) * (y1 - y0)
    pa = (a[2] - a[0]) * (a[3] - a[1])
    pb = (b[2] - b[0]) * (b[3] - b[1])
    return wsp / (pa + pb - wsp)


# ---------------------------------------------------------------- prog ufnosci

def _katalogi_splitu(dane, split):
    """data.yaml + nazwa czesci -> (katalog obrazow, katalog etykiet)."""
    import yaml
    from .plan import katalog_etykiet
    p = Path(dane)
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    baza = Path(d.get("path", p.parent))
    if not baza.is_absolute():
        baza = (p.parent / baza).resolve()
    obrazy = baza / d[split]
    return obrazy, katalog_etykiet(obrazy)


def _dopasuj(predykcje, prawda, iou=0.5):
    """Dopasowanie zachlanne od najpewniejszej predykcji na kazdym obrazie.

    Zwraca (lista (pewnosc, czy_trafienie), liczba obiektow prawdziwych).
    Wspolne dla pr_przy_progach() i ap50(), zeby obie liczby byly spojne.
    """
    oceny = []
    wszystkie_prawdziwe = 0
    for obraz, gt in prawda.items():
        wszystkie_prawdziwe += len(gt)
        uzyte = [False] * len(gt)
        for k, conf, *box in sorted(predykcje.get(obraz, []), key=lambda r: -r[1]):
            najlepszy, idx = 0.0, -1
            for j, (gk, *gbox) in enumerate(gt):
                if uzyte[j] or gk != k:
                    continue
                v = _iou(box, gbox)
                if v > najlepszy:
                    najlepszy, idx = v, j
            if najlepszy >= iou:
                uzyte[idx] = True
                oceny.append((conf, True))
            else:
                oceny.append((conf, False))
    return oceny, wszystkie_prawdziwe


def pr_przy_progach(predykcje, prawda, progi, iou=0.5):
    """Precyzja i czulosc liczone WPROST przy kazdym progu ufnosci.

    predykcje: {obraz: [(klasa, pewnosc, x0, y0, x1, y1), ...]}
    prawda:    {obraz: [(klasa, x0, y0, x1, y1), ...]}

    Dopasowanie zachlanne od najpewniejszej predykcji: kazdy obiekt prawdziwy
    moze byc trafiony tylko raz. Przy takiej kolejnosci to, czy predykcja jest
    trafieniem, nie zalezy od predykcji mniej pewnych - wystarczy wiec jedno
    dopasowanie, a dla kazdego progu tylko zliczamy predykcje powyzej niego.
    Dzieki temu czulosc jest gwarantowanie nierosnaca wraz z progiem.
    """
    oceny, wszystkie_prawdziwe = _dopasuj(predykcje, prawda, iou)
    wyniki = []
    for t in progi:
        tp = sum(1 for c, ok in oceny if c >= t and ok)
        fp = sum(1 for c, ok in oceny if c >= t and not ok)
        fn = wszystkie_prawdziwe - tp
        wyniki.append({"prog": t, "trafione": tp, "falszywe": fp, "pominiete": fn,
                       "precyzja": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                       "czulosc": round(tp / wszystkie_prawdziwe, 4) if wszystkie_prawdziwe else 0.0})
    return wyniki


def ap50(predykcje, prawda, iou=0.5):
    """AP przy IoU 0,5 liczone tak jak w ultralytics: pole pod wygladzona krzywa
    precyzja-czulosc w 101 punktach czulosci. Przy jednej klasie AP50 = mAP50.

    Wlasna wersja, bo dla modelu .hef nie ma walidacji ultralytics - a tak
    model .pt i .hef sa liczone tym samym kodem.
    """
    oceny, n = _dopasuj(predykcje, prawda, iou)
    if not n or not oceny:
        return 0.0
    oceny.sort(key=lambda r: -r[0])
    tp = np.cumsum([1 if ok else 0 for _, ok in oceny])
    fp = np.cumsum([0 if ok else 1 for _, ok in oceny])
    czulosc = tp / n
    precyzja = tp / (tp + fp)
    mrec = np.concatenate(([0.0], czulosc, [1.0]))
    mpre = np.concatenate(([1.0], precyzja, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))          # wygladzenie: nie rosnie w prawo
    x = np.linspace(0, 1, 101)
    calka = getattr(np, "trapezoid", None) or np.trapz
    return round(float(calka(np.interp(x, mrec, mpre), x)), 4)


def predykcje_splitu(model, dane, split, wejscie=640, prog_min=0.05, urzadzenie=None):
    """Jedno przejscie modelu po czesci zbioru -> (predykcje, prawda) per obraz."""
    YOLO = _yolo()
    m = YOLO(str(model))
    kat_obr, kat_et = _katalogi_splitu(dane, split)
    prawda = wczytaj_etykiety(kat_et, kat_obr)
    pred = {}
    for stem in prawda:
        obraz = next((kat_obr / (stem + e) for e in (".jpg", ".jpeg", ".png")
                      if (kat_obr / (stem + e)).exists()), None)
        if obraz is None:
            continue
        r = m.predict(str(obraz), imgsz=wejscie, conf=prog_min, device=wybierz(urzadzenie),
                      max_det=1000, verbose=False)[0]
        pred[stem] = [(int(b.cls), float(b.conf), *[float(v) for v in b.xyxy[0]]) for b in r.boxes]
    return pred, prawda


def predykcje_detektora(detektor, dane, split="test"):
    """Przejscie dowolnego detektora (mission.detector: YOLO, Hailo, atrapa) po czesci zbioru.

    Detektor powinien miec niski prog (np. 0,001), zeby dalo sie policzyc AP50.
    Wszystkie klasy detektora mapowane na 0 - nasze modele maja jedna klase.
    """
    import cv2
    kat_obr, kat_et = _katalogi_splitu(dane, split)
    prawda = wczytaj_etykiety(kat_et, kat_obr)
    pred = {}
    for stem in prawda:
        obraz = next((kat_obr / (stem + e) for e in (".jpg", ".jpeg", ".png")
                      if (kat_obr / (stem + e)).exists()), None)
        if obraz is None:
            continue
        img = cv2.imread(str(obraz))
        pred[stem] = [(0, d.pewnosc, d.x0, d.y0, d.x1, d.y1) for d in detektor.wykryj(img)]
    prawda = {k: [(0, *r[1:]) for r in v] for k, v in prawda.items() if k in pred}
    return pred, prawda


def ocen_detektor(detektor, dane, split="test", prog=0.15, iou=0.5):
    """AP50, precyzja i czulosc przy progu oraz wynik wedlug dni - dla dowolnego detektora."""
    pred, prawda = predykcje_detektora(detektor, dane, split)
    return {"detektor": getattr(detektor, "nazwa", str(detektor)), "split": split,
            "obrazow": len(prawda), "AP50": ap50(pred, prawda, iou),
            "przy_progu": pr_przy_progach(pred, prawda, [prog], iou)[0],
            "wedlug_dni": ocen_wedlug_grup(pred, prawda, prog, iou)}


def ocen_wedlug_grup(predykcje, prawda, prog=0.15, iou=0.5, wzorzec=r"\d{4}-\d{2}-\d{2}"):
    """Czulosc i precyzja osobno dla kazdej grupy obrazow (domyslnie: dzien lotu
    odczytany z nazwy pliku) oraz lacznie.

    Srednia z calego testu potrafi ukryc slaby dzien - w WeedsGalore najwczesniejszy
    dzien z najmniejszymi siewkami wypada kilka razy gorzej niz pozostale.
    """
    import re
    grupy = {}
    for stem in prawda:
        m = re.search(wzorzec, stem)
        grupy.setdefault(m.group(0) if m else "inne", []).append(stem)
    wynik = {}
    for nazwa, stemy in sorted(grupy.items()) + [("razem", list(prawda))]:
        w = pr_przy_progach({s: predykcje.get(s, []) for s in stemy},
                            {s: prawda[s] for s in stemy}, [prog], iou)[0]
        w["obrazow"] = len(stemy)
        wynik[nazwa] = w
    return wynik


def dobierz_prog(model, dane, split="val", progi=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5),
                 waga_czulosci=3.0, wejscie=640, urzadzenie=None, iou=0.5):
    """Prog ufnosci wazacy czulosc wyzej niz precyzje (miara F-beta).

    Domyslna waga 3.0 oznacza: pominiecie chwastu jest trzy razy gorsze niz
    falszywy alarm. To zalozenie wynika z zastosowania, nie z matematyki.

    UWAGA: dobieraj prog na czesci 'val', nigdy na 'test' - inaczej wynik
    na tescie przestaje byc niezalezny.

    Wczesniejsza wersja uzywala walidacji ultralytics z roznymi conf, ale ta
    raportuje precyzje i czulosc we wlasnym "najlepszym" punkcie krzywej, a nie
    przy podanym progu - czulosc potrafila rosnac razem z progiem.
    """
    YOLO = _yolo()
    m = YOLO(str(model))
    kat_obr, kat_et = _katalogi_splitu(dane, split)
    prawda = wczytaj_etykiety(kat_et, kat_obr)

    predykcje = {}
    for stem in prawda:
        obraz = next((kat_obr / (stem + e) for e in (".jpg", ".jpeg", ".png")
                      if (kat_obr / (stem + e)).exists()), None)
        if obraz is None:
            continue
        r = m.predict(str(obraz), imgsz=wejscie, conf=min(progi), device=wybierz(urzadzenie),
                      max_det=1000, verbose=False)[0]
        predykcje[stem] = [(int(b.cls), float(b.conf), *[float(v) for v in b.xyxy[0]])
                           for b in r.boxes]

    wyniki = pr_przy_progach(predykcje, prawda, progi, iou)
    beta2 = waga_czulosci ** 2
    for w in wyniki:
        p, r = w["precyzja"], w["czulosc"]
        w["f_beta"] = round((1 + beta2) * p * r / (beta2 * p + r), 4) if (p + r) else 0.0
        log.info("prog %.2f: precyzja %.3f czulosc %.3f (trafione %d, falszywe %d, pominiete %d)",
                 w["prog"], p, r, w["trafione"], w["falszywe"], w["pominiete"])
    najlepszy = max(wyniki, key=lambda x: x["f_beta"])
    return {"zalecany_prog": najlepszy["prog"], "waga_czulosci": waga_czulosci,
            "split": split, "iou": iou, "wyniki": wyniki}


def zapisz(wynik, sciezka):
    p = Path(sciezka)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(wynik, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
