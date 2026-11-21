"""Ocena modelu: AP50, czulosc i precyzja przy progu, wynik wedlug dni lotu.

Dlaczego nie wystarczy samo mAP:

  Zbiory sa niezbalansowane (w Sudarsie chwastow jest wielokrotnie wiecej niz
  upraw), wiec srednia zaciera to, ze model moze zupelnie nie radzic sobie
  z rzadka klasa.

  Zadanie jest zdominowane przez obiekty male - chwast z pulapu ma kilkadziesiat
  pikseli. mAP liczone na wszystkich obiektach moze wygladac dobrze dzieki
  duzym roslinom, podczas gdy te najwazniejsze sa pomijane.

  Czulosc jest wazniejsza niz precyzja: pominiete ognisko to realna strata
  plonu, falszywy alarm to tylko zbedne sprawdzenie fragmentu pola.

Model koncowy jest oceniany przez ocen_detektor() (komenda ocen-detektor, final_model.sh krok 3):
ten sam kod dla .pt, .onnx i .hef, wiec wyniki z Maca i z Raspberry Pi sa porownywalne.

Format danych w funkcjach (ramki w pikselach obrazu):
    predykcje = {nazwa_obrazu: [(klasa, pewnosc, x0, y0, x1, y1), ...]}
    prawda    = {nazwa_obrazu: [(klasa, x0, y0, x1, y1), ...]}

Pojecia:
    IoU       - pole czesci wspolnej dwoch ramek / pole ich sumy; trafienie = IoU >= 0,5
    czulosc   - jaka czesc prawdziwych chwastow model znalazl (recall)
    precyzja  - jaka czesc wykryc to naprawde chwasty (precision)
    AP50      - pole pod krzywa precyzja-czulosc przy IoU 0,5; nie zalezy od progu pewnosci
"""

import json
from pathlib import Path

import numpy as np

from mission.detector import iou_ramek


# ---------------------------------------------------------------- etykiety

def wczytaj_etykiety(katalog_etykiet, katalog_obrazow=None, rozmiar=None):
    """Etykiety YOLO -> {nazwa_pliku: [(klasa, x0, y0, x1, y1), ...]} w pikselach.

    rozmiar: (szer, wys), gdy wszystkie obrazy maja ten sam rozmiar (kafelki).
    Inaczej rozmiar odczytywany jest z plikow obrazow.

    Argumenty:
        katalog_etykiet - folder .txt YOLO, np. "zbior_lotnicze/dataset/labels/test"
        katalog_obrazow - folder obrazow (do odczytu rozmiaru) albo None
        rozmiar         - (szer, wys) wspolny dla wszystkich obrazow albo None

    Zwraca:
        dict, np. {"weedsgalore__2023-05-25__2023-05-25_0012": [(0, 120.0, 88.5, 150.2, 119.0)]}
        (obrazy bez pliku obrazu sa pomijane, gdy rozmiar nie jest podany)

    Uzywana w:
        predykcje_detektora() (ten plik)
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


# ---------------------------------------------------------------- dopasowanie i miary

def _katalogi_splitu(dane, split):
    """data.yaml + nazwa czesci -> (katalog obrazow, katalog etykiet).

    Argumenty:
        dane  - data.yaml, np. "zbior_lotnicze/dataset/data.yaml"
        split - "train" / "val" / "test"

    Zwraca:
        (Path obrazow, Path etykiet), np. (".../dataset/images/test", ".../dataset/labels/test")

    Uzywana w:
        predykcje_detektora() (ten plik)
    """
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

    Jak dziala: na kazdym obrazie predykcje od najpewniejszej; kazda dostaje najlepiej
    pasujacy, jeszcze nieuzyty prawdziwy obiekt tej samej klasy. IoU >= iou -> trafienie.

    Argumenty:
        predykcje - {obraz: [(klasa, pewnosc, x0, y0, x1, y1), ...]}
        prawda    - {obraz: [(klasa, x0, y0, x1, y1), ...]}
        iou       - minimalne IoU trafienia, np. 0.5

    Zwraca:
        (oceny, n), np. ([(0.91, True), (0.40, False), ...], 1628)

    Uzywana w:
        pr_przy_progach(), ap50() (ten plik)
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
                v = iou_ramek(box, gbox)
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

    Argumenty:
        predykcje, prawda - jak wyzej
        progi             - lista progow, np. [0.15]
        iou               - minimalne IoU trafienia, np. 0.5

    Zwraca:
        lista slownikow (jeden na prog), np. [{"prog": 0.15, "trafione": 80, "falszywe": 20,
        "pominiete": 30, "precyzja": 0.8, "czulosc": 0.7273}]

    Uzywana w:
        ocen_detektor(), ocen_wedlug_grup() (ten plik), mission/sim_flight.py porownaj()
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

    Argumenty:
        predykcje - {obraz: [(klasa, pewnosc, x0, y0, x1, y1), ...]} (z niskim progiem, np. 0.001)
        prawda    - {obraz: [(klasa, x0, y0, x1, y1), ...]}
        iou       - minimalne IoU trafienia, np. 0.5

    Zwraca:
        float 0-1, np. 0.817 (model koncowy na tescie EWIS1 z TTA)

    Uzywana w:
        ocen_detektor() (ten plik), tests/test_hailo.py
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


def predykcje_detektora(detektor, dane, split="test"):
    """Przejscie dowolnego detektora (mission.detector: YOLO, Hailo, atrapa) po czesci zbioru.

    Detektor powinien miec niski prog (np. 0,001), zeby dalo sie policzyc AP50.
    Wszystkie klasy detektora mapowane na 0 - nasze modele maja jedna klase.

    Argumenty:
        detektor - obiekt z metoda wykryj(obraz_bgr) -> lista Detection,
                   np. z mission/detector.py zbuduj_z_plikow()
        dane     - data.yaml
        split    - czesc zbioru, np. "test"

    Zwraca:
        (pred, prawda) - slowniki w formacie z naglowka pliku (klasa zawsze 0)

    Uzywana w:
        ocen_detektor() (ten plik)
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
    """AP50, precyzja i czulosc przy progu oraz wynik wedlug dni - dla dowolnego detektora.

    Po co: glowna ocena modelu koncowego - ten sam kod dla .pt (Mac), .onnx i .hef
           (Raspberry Pi + Hailo), wiec liczby sa porownywalne. AP50 nie zalezy od progu;
           czulosc i precyzja sa liczone przy progu prog.

    Argumenty:
        detektor - obiekt z wykryj(obraz), zbudowany z niskim progiem (np. 0.001)
        dane     - data.yaml, np. "zbior_ewis1_podzial/dataset/data.yaml"
        split    - czesc zbioru, np. "test"
        prog     - prog pewnosci dla czulosci i precyzji, np. 0.15
        iou      - minimalne IoU trafienia, np. 0.5

    Zwraca:
        dict, np. {"detektor": "TTA(YOLO(...))", "split": "test", "obrazow": 132, "AP50": 0.817,
                   "przy_progu": {"prog": 0.15, "trafione": ..., "czulosc": ..., "precyzja": ...},
                   "wedlug_dni": {"2022-05-16": {...}, "razem": {...}}}
        (to jest zawartosc plikow wyniki/model_koncowy/*.json)

    Uzywana w:
        models/cli.py cmd_ocen_detektor() - komenda "ocen-detektor" w scripts/final_model.sh
        (krok 3) i scripts/check_hef.sh
    """
    pred, prawda = predykcje_detektora(detektor, dane, split)
    return {"detektor": getattr(detektor, "nazwa", str(detektor)), "split": split,
            "obrazow": len(prawda), "AP50": ap50(pred, prawda, iou),
            "przy_progu": pr_przy_progach(pred, prawda, [prog], iou)[0],
            "wedlug_dni": ocen_wedlug_grup(pred, prawda, prog, iou)}


WZORZEC_DATY = r"\d{4}-\d{2}-\d{2}|(?<!\d)20\d{6}(?!\d)"     # WeedsGalore 2023-05-25, EWIS1 20220509


def ocen_wedlug_grup(predykcje, prawda, prog=0.15, iou=0.5, wzorzec=WZORZEC_DATY):
    """Czulosc i precyzja osobno dla kazdej grupy obrazow (domyslnie: dzien lotu
    odczytany z nazwy pliku) oraz lacznie.

    Srednia z calego testu potrafi ukryc slaby dzien - w WeedsGalore najwczesniejszy
    dzien z najmniejszymi siewkami wypada kilka razy gorzej niz pozostale.

    Argumenty:
        predykcje, prawda - slowniki w formacie z naglowka pliku
        prog              - prog pewnosci, np. 0.15
        iou               - minimalne IoU trafienia
        wzorzec           - data w nazwie: 2023-05-25 (WeedsGalore) albo 20220516 (EWIS1)

    Zwraca:
        dict dzien -> wynik pr_przy_progach() + "obrazow", plus "razem";
        daty EWIS1 zamienione na RRRR-MM-DD, np. {"2023-05-25": {...}, "2023-05-30": {...}, "razem": {...}}

    Uzywana w:
        ocen_detektor() (ten plik)
    """
    import re
    grupy = {}
    for stem in prawda:
        m = re.search(wzorzec, stem)
        d = m.group(0) if m else "inne"
        if len(d) == 8 and d.isdigit():                       # 20220509 -> 2022-05-09
            d = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        grupy.setdefault(d, []).append(stem)
    wynik = {}
    for nazwa, stemy in sorted(grupy.items()) + [("razem", list(prawda))]:
        w = pr_przy_progach({s: predykcje.get(s, []) for s in stemy},
                            {s: prawda[s] for s in stemy}, [prog], iou)[0]
        w["obrazow"] = len(stemy)
        wynik[nazwa] = w
    return wynik


def zapisz(wynik, sciezka):
    """Zapisuje wynik (slownik) do pliku JSON, tworzac brakujace foldery.

    Argumenty:
        wynik   - slownik z oceny
        sciezka - plik, np. "wyniki/model_koncowy/wg_tta.json"

    Zwraca:
        Path zapisanego pliku

    Uzywana w:
        models/cli.py cmd_ocen_detektor() (opcja -o)
    """
    p = Path(sciezka)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(wynik, indent=2, ensure_ascii=False), encoding="utf-8")
    return p