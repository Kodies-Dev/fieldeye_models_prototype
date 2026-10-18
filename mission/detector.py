"""Detekcja chwastow na pojedynczym zdjeciu.

Detektor jest wymienny - potok w locie nie wie, jakiego modelu uzywa.
Dzieki temu mozna podmienic model bez zmiany reszty programu.

Trzy implementacje:
  YoloDetector     - docelowy, wymaga ultralytics i wytrenowanego modelu (.pt)
  TiledDetector  - opakowanie: tnie duzy kadr na kafelki i scala wyniki
  DummyDetector   - NIE JEST DETEKTOREM CHWASTOW. Sluzy wylacznie do
                     sprawdzenia, czy potok dziala, zanim bedzie gotowy model.
"""

import logging

import cv2
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)


class Detection:
    """Jeden znaleziony obiekt: ramka w pikselach zdjecia + klasa + pewnosc."""

    __slots__ = ("x0", "y0", "x1", "y1", "klasa", "pewnosc")

    def __init__(self, x0, y0, x1, y1, klasa, pewnosc):
        self.x0, self.y0, self.x1, self.y1 = float(x0), float(y0), float(x1), float(y1)
        self.klasa = klasa
        self.pewnosc = float(pewnosc)

    @property
    def srodek(self):
        return (self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0

    def slownik(self):
        return {"bbox": [round(v, 1) for v in (self.x0, self.y0, self.x1, self.y1)],
                "klasa": self.klasa, "pewnosc": round(self.pewnosc, 3)}


# ---------------------------------------------------------------- YOLO

class YoloDetector:
    """Docelowy detektor. Wymaga: pip install ultralytics + plik modelu .pt."""

    def __init__(self, model, prog_pewnosci=0.25, wejscie=640, urzadzenie=None):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Brak biblioteki ultralytics. Zainstaluj: pip install ultralytics"
            ) from e
        self.model = YOLO(model)
        self.prog = prog_pewnosci
        self.wejscie = wejscie
        # bez jawnego urzadzenia ultralytics na Macu liczy na procesorze
        from models.device import wybierz
        self.urzadzenie = wybierz(urzadzenie)
        self.nazwa = f"YOLO({model})"

    def wykryj(self, obraz):
        wyniki = self.model.predict(obraz, conf=self.prog, imgsz=self.wejscie,
                                    device=self.urzadzenie, verbose=False)
        out = []
        for r in wyniki:
            for b in r.boxes:
                x0, y0, x1, y1 = (float(v) for v in b.xyxy[0])
                out.append(Detection(x0, y0, x1, y1,
                                    r.names[int(b.cls)], float(b.conf)))
        return out


# ---------------------------------------------------------------- kafelki

class TiledDetector:
    """Tnie kadr na kafelki, uruchamia detektor na kazdym i scala wyniki.

    Po co: kadr z kamery ma ~4000 px, wejscie sieci 640 px. Zmniejszenie
    calego kadru szescikrotnie sprawia, ze chwast o boku 40 px ma 6 px i
    jest niewykrywalny. Kafelkowanie zachowuje oryginalna rozdzielczosc -
    kosztem kilkunastu uruchomien sieci na kadr zamiast jednego.

    Zakladka miedzy kafelkami sprawia, ze obiekt na granicy nie zostanie
    przeciety na pol; duplikaty z zakladki usuwa _scal().
    """

    def __init__(self, detektor, kafelek=896, zakladka=0.15, prog_iou=0.45):
        self.detektor = detektor
        self.kafelek = kafelek
        self.zakladka = zakladka
        self.prog_iou = prog_iou
        self.nazwa = f"kafelki({detektor.nazwa})"

    def wykryj(self, obraz):
        h, w = obraz.shape[:2]
        if max(h, w) <= self.kafelek:
            return self.detektor.wykryj(obraz)

        krok = int(self.kafelek * (1 - self.zakladka))
        # pozycje kafelkow; ostatni dosuniety do krawedzi, zeby nie zgubic jej fragmentu
        poz_x = sorted({min(x, w - self.kafelek) for x in range(0, w, krok)} | {w - self.kafelek})
        poz_y = sorted({min(y, h - self.kafelek) for y in range(0, h, krok)} | {h - self.kafelek})
        poz_x = [max(0, x) for x in poz_x]
        poz_y = [max(0, y) for y in poz_y]

        zebrane = []
        for y in poz_y:
            for x in poz_x:
                wycinek = obraz[y:y + self.kafelek, x:x + self.kafelek]
                for d in self.detektor.wykryj(wycinek):
                    # wspolrzedne z kafelka z powrotem na cale zdjecie
                    zebrane.append(Detection(d.x0 + x, d.y0 + y, d.x1 + x, d.y1 + y,
                                            d.klasa, d.pewnosc))
        return self._scal(zebrane)

    def _scal(self, det):
        """Usuwa duplikaty z zakladki (klasyczne NMS - non-maximum suppression)."""
        det = sorted(det, key=lambda d: -d.pewnosc)
        zostaje = []
        for d in det:
            if all(self._iou(d, z) < self.prog_iou or z.klasa != d.klasa for z in zostaje):
                zostaje.append(d)
        return zostaje

    @staticmethod
    def _iou(a, b):
        """Stosunek czesci wspolnej do sumy pol dwoch ramek (0 = rozlaczne)."""
        x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
        x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        wsp = (x1 - x0) * (y1 - y0)
        pa = (a.x1 - a.x0) * (a.y1 - a.y0)
        pb = (b.x1 - b.x0) * (b.y1 - b.y0)
        return wsp / (pa + pb - wsp)


# ---------------------------------------------------------------- atrapa

class DummyDetector:
    """UWAGA: to NIE JEST detektor chwastow.

    Szuka po prostu zwartych plam zieleni odstajacych od otoczenia. Na
    prawdziwym polu zaznaczy tak samo uprawe jak chwast - nie odroznia
    gatunkow i nie ma z tym nic wspolnego.

    Istnieje po to, zeby dalo sie uruchomic i przetestowac caly potok
    (telemetria -> detekcja -> wspolrzedne -> mapa) ZANIM bedzie gotowy
    wytrenowany model. Kazda detekcja dostaje klase "atrapa", zeby
    przypadkiem nie trafila do wynikow jako prawdziwy chwast.
    """

    def __init__(self, min_pole=120, max_pole_udzial=0.02):
        self.min_pole = min_pole
        self.max_pole_udzial = max_pole_udzial
        self.nazwa = "ATRAPA (nie jest detektorem chwastow)"
        log.warning("Uzywasz atrapy detektora - wyniki NIE sa detekcja chwastow.")

    def wykryj(self, obraz):
        h, w = obraz.shape[:2]
        f = obraz.astype(np.float32) / 255.0
        # indeks nadmiaru zieleni (ExG) - klasyczny wskaznik roslinnosci
        exg = 2 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]
        prog = float(np.percentile(exg, 97))
        maska = ((exg > prog) * 255).astype(np.uint8)
        maska = cv2.morphologyEx(maska, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        n, _, staty, _ = cv2.connectedComponentsWithStats(maska, 8)
        maks = self.max_pole_udzial * h * w
        out = []
        for i in range(1, n):
            x, y, bw, bh, pole = staty[i]
            if self.min_pole <= pole <= maks:
                # To NIE jest pewnosc modelu - atrapa nie ma pojecia o chwastach.
                # Wartosc rosnie z wielkoscia plamy i mieści sie w 0.5-0.9, zeby
                # dawala jakies uporzadkowanie przy scalaniu i nie byla odcinana
                # przez domyslny prog mapy (0.30).
                pseudo = 0.5 + 0.4 * min(1.0, pole / (0.25 * maks))
                out.append(Detection(x, y, x + bw, y + bh, "atrapa", pseudo))
        return out


# ---------------------------------------------------------------- kilka przebiegow: modele i TTA

def _iou_det(a, b):
    x0, y0, x1, y1 = max(a.x0, b[0]), max(a.y0, b[1]), min(a.x1, b[2]), min(a.y1, b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    w = (x1 - x0) * (y1 - y0)
    return w / ((a.x1 - a.x0) * (a.y1 - a.y0) + (b[2] - b[0]) * (b[3] - b[1]) - w)


def polacz_wbf(listy, iou=0.55):
    """Weighted box fusion: wyniki kilku przebiegow tego samego zdjecia -> jedna lista.

    listy: [[Detection, ...] z przebiegu 1, [..] z przebiegu 2, ...]. Ramki,
    ktore sie pokrywaja, laczy w jedna: polozenie to srednia wazona pewnoscia,
    pewnosc to suma najwyzszych pewnosci z kazdego przebiegu / liczba przebiegow.
    Wykrycie widziane w jednym z dwoch przebiegow dostaje wiec polowe pewnosci -
    dlatego model laczony porownujemy glownie miara AP50, niezalezna od progu.
    """
    n = max(1, len(listy))
    grupy = []                                       # [suma_w, [x0, y0, x1, y1]*w, {przebieg: max}, klasa]
    for k, lista in enumerate(listy):
        for d in lista:
            for g in sorted(grupy, key=lambda g: -max(g[2].values())):
                ramka = [v / g[0] for v in g[1]]
                if g[3] == d.klasa and _iou_det(d, ramka) >= iou:
                    break
            else:
                g = [0.0, [0.0, 0.0, 0.0, 0.0], {}, d.klasa]
                grupy.append(g)
            w = d.pewnosc
            g[0] += w
            g[1] = [s_ + v * w for s_, v in zip(g[1], (d.x0, d.y0, d.x1, d.y1))]
            g[2][k] = max(g[2].get(k, 0.0), w)
    out = []
    for suma_w, ramka, per, klasa in grupy:
        if suma_w <= 0:
            continue
        x0, y0, x1, y1 = (v / suma_w for v in ramka)
        out.append(Detection(x0, y0, x1, y1, klasa, min(1.0, sum(per.values()) / n)))
    return out


class EnsembleDetector:
    """Kilka modeli na tym samym zdjeciu, wyniki laczone przez WBF.

    Modele z roznym ziarnem losowym myla sie troche inaczej - polaczenie zwykle
    poprawia AP o 1-3 pp. Na Hailo-8 oznacza to dwa pliki .hef i dwa przebiegi.
    """

    def __init__(self, detektory, iou=0.55):
        self.detektory = list(detektory)
        self.iou = iou
        self.nazwa = "Ensemble(" + " + ".join(getattr(d, "nazwa", "?") for d in self.detektory) + ")"

    def wykryj(self, obraz):
        return polacz_wbf([d.wykryj(obraz) for d in self.detektory], self.iou)


class FlipTTADetector:
    """Detekcja na oryginale i odbiciu poziomym (test-time augmentation), laczona przez WBF.

    Z gory chwast wyglada tak samo w odbiciu, ale model widzi go w innym
    ukladzie pikseli i czasem znajduje tylko w jednej wersji. pionowo=True
    dodaje trzeci przebieg z odbiciem pionowym.
    """

    def __init__(self, detektor, pionowo=False, iou=0.55):
        self.detektor = detektor
        self.pionowo = pionowo
        self.iou = iou
        self.nazwa = f"TTA({getattr(detektor, 'nazwa', '?')}{', +pion' if pionowo else ''})"

    def wykryj(self, obraz):
        h, w = obraz.shape[:2]
        listy = [self.detektor.wykryj(obraz)]
        listy.append([Detection(w - d.x1, d.y0, w - d.x0, d.y1, d.klasa, d.pewnosc)
                      for d in self.detektor.wykryj(np.ascontiguousarray(obraz[:, ::-1]))])
        if self.pionowo:
            listy.append([Detection(d.x0, h - d.y1, d.x1, h - d.y0, d.klasa, d.pewnosc)
                          for d in self.detektor.wykryj(np.ascontiguousarray(obraz[::-1]))])
        return polacz_wbf(listy, self.iou)


def zbuduj_z_plikow(modele, prog=0.15, wejscie=1280, urzadzenie=None, tta=False):
    """Jeden albo kilka modeli (.pt / .onnx / .hef) -> detektor, opcjonalnie z TTA."""
    modele = [modele] if isinstance(modele, (str, Path)) else list(modele)
    dets = [HailoDetector(m, prog) if str(m).lower().endswith(".hef")
            else YoloDetector(m, prog, wejscie, urzadzenie) for m in modele]
    d = dets[0] if len(dets) == 1 else EnsembleDetector(dets)
    return FlipTTADetector(d) if tta else d


# ---------------------------------------------------------------- Hailo (.hef)

def letterbox(obraz, wys, szer, wypelnienie=114):
    """Wpisuje obraz BGR w prostokat wys x szer z zachowaniem proporcji.

    Tak samo jak ultralytics przy treningu: zmniejszenie albo powiekszenie,
    a reszta wypelniona szaroscia (114). HEF ma staly rozmiar wejscia, wiec
    to przygotowanie musimy zrobic sami. Zwraca obraz RGB uint8 oraz skale
    i przesuniecie, potrzebne do przeliczenia ramek z powrotem.
    """
    import cv2
    h, w = obraz.shape[:2]
    s = min(szer / w, wys / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    zm = cv2.resize(obraz, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (szer - nw) // 2, (wys - nh) // 2
    plotno = np.full((wys, szer, 3), wypelnienie, np.uint8)
    plotno[py:py + nh, px:px + nw] = zm
    return cv2.cvtColor(plotno, cv2.COLOR_BGR2RGB), s, px, py


def nms_hailo_na_detekcje(wyjscie, wys, szer, skala, px, py, w0, h0, prog=0.0, nazwy=None):
    """Wynik HEF z wbudowanym NMS -> lista Detekcja w pikselach oryginalnego obrazu.

    Format Hailo ("NMS by class"): lista z jednym elementem na klase; element
    to tablica wierszy [y0, x0, y1, x1, pewnosc], wspolrzedne jako ulamek
    wejscia sieci (0-1). Kolejnosc y przed x - inaczej niz w ultralytics.
    """
    nazwy = nazwy or {0: "chwast"}
    out = []
    for k, tab in enumerate(wyjscie):
        tab = np.asarray(tab, dtype=np.float32).reshape(-1, 5)
        for y0, x0, y1, x1, c in tab:
            if c < prog:
                continue
            # ulamek wejscia -> piksel wejscia -> bez paddingu -> piksel oryginalu
            X0 = (x0 * szer - px) / skala
            X1 = (x1 * szer - px) / skala
            Y0 = (y0 * wys - py) / skala
            Y1 = (y1 * wys - py) / skala
            X0, X1 = np.clip([X0, X1], 0, w0)
            Y0, Y1 = np.clip([Y0, Y1], 0, h0)
            if X1 > X0 and Y1 > Y0:
                out.append(Detection(X0, Y0, X1, Y1, nazwy.get(k, str(k)), c))
    return out


class HailoDetector:
    """Model skompilowany do .hef, uruchamiany na akceleratorze Hailo (np. Hailo-8 w AI HAT+).

    Dziala tylko na urzadzeniu z Hailo i zainstalowanym HailoRT (pakiet
    hailo_platform; na Raspberry Pi: sudo apt install hailo-all). Zwraca te
    same obiekty Detekcja co YoloDetector, wiec reszta misji sie nie zmienia.

    HEF musi byc skompilowany z NMS (domyslnie w Hailo Model Zoo dla yolov8n).
    Prog pewnosci NMS ustalony przy kompilacji jest dolna granica; prog tutaj
    odsiewa wyzej, np. 0,15.
    """

    def __init__(self, hef, prog_pewnosci=0.15, nazwy=None):
        try:
            from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface,
                                        InferVStreams, InputVStreamParams, OutputVStreamParams,
                                        VDevice)
        except ImportError as e:
            raise ImportError("Do uruchomienia .hef potrzebny jest HailoRT (pakiet hailo_platform) "
                              "na urzadzeniu z Hailo, np.: sudo apt install hailo-all") from e
        self.prog = prog_pewnosci
        self.nazwy = nazwy
        self.nazwa = f"Hailo({Path(hef).name})"
        self._hef = HEF(str(hef))
        self._urz = VDevice()
        params = ConfigureParams.create_from_hef(self._hef, interface=HailoStreamInterface.PCIe)
        self._ng = self._urz.configure(self._hef, params)[0]
        we = InputVStreamParams.make(self._ng, format_type=FormatType.UINT8)
        wy = OutputVStreamParams.make(self._ng, format_type=FormatType.FLOAT32)
        info = self._hef.get_input_vstream_infos()[0]
        self._we_nazwa = info.name
        self.wys, self.szer = int(info.shape[0]), int(info.shape[1])
        self._wy_nazwa = self._hef.get_output_vstream_infos()[0].name
        # potok otwierany raz, a nie przy kazdym kadrze - otwarcie trwa kilkaset ms
        self._potok = InferVStreams(self._ng, we, wy)
        self._potok.__enter__()
        self._aktywacja = self._ng.activate(self._ng.create_params())
        self._aktywacja.__enter__()

    def wykryj(self, obraz):
        h0, w0 = obraz.shape[:2]
        we, s, px, py = letterbox(obraz, self.wys, self.szer)
        wynik = self._potok.infer({self._we_nazwa: we[None]})[self._wy_nazwa]
        return nms_hailo_na_detekcje(wynik[0], self.wys, self.szer, s, px, py, w0, h0,
                                     self.prog, self.nazwy)

    def zamknij(self):
        for kontekst in ("_aktywacja", "_potok"):
            k = getattr(self, kontekst, None)
            if k is not None:
                k.__exit__(None, None, None)
                setattr(self, kontekst, None)


def zbuduj_detektor(cfg):
    """Tworzy detektor wedlug konfiguracji. cfg.model = None -> atrapa; .hef -> Hailo."""
    if cfg.model and str(cfg.model).lower().endswith(".hef"):
        d = HailoDetector(cfg.model, cfg.prog_pewnosci)
    elif cfg.model:
        d = YoloDetector(cfg.model, cfg.prog_pewnosci, cfg.wejscie_sieci)
    else:
        d = DummyDetector()
    return TiledDetector(d, cfg.kafelek, cfg.zakladka) if cfg.kafelkowanie else d