"""Detekcja chwastow na pojedynczym zdjeciu.

Detektor jest wymienny - potok w locie nie wie, jakiego modelu uzywa.
Dzieki temu mozna podmienic model bez zmiany reszty programu. Kazdy detektor
ma metode wykryj(obraz_bgr) -> lista Detection (ramki w pikselach zdjecia).

Implementacje:
  YoloDetector     - model .pt przez ultralytics (Mac / PC)
  OnnxDetector     - model .onnx przez ONNX Runtime, bez PyTorcha (lekka binarka)
  HailoDetector    - model .hef na akceleratorze Hailo-8 (Raspberry Pi)
  TiledDetector    - opakowanie: tnie duzy kadr na kafelki i scala wyniki
  FlipTTADetector  - opakowanie: oryginal + odbicie (TTA), wyniki laczone przez WBF
  DummyDetector    - NIE JEST DETEKTOREM CHWASTOW. Sluzy wylacznie do
                     sprawdzenia, czy potok dziala, zanim bedzie gotowy model.

Budowanie:
  zbuduj_z_plikow()  - plik modelu (.pt / .onnx / .hef) -> detektor (+ TTA); uzywa go
                       ocen-detektor w scripts/final_model.sh i check_hef.sh oraz binarka

Wspolne: iou_ramek() - jedyne liczenie IoU w projekcie (uzywa go tez models/evaluation.py).

Model koncowy: models/weights/fieldeye_final.pt (YoloDetector) albo .hef (HailoDetector),
wejscie 1280, prog 0.15, z TTA (FlipTTADetector) przy ocenie.
"""

import importlib.util
import logging

import cv2
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)


class Detection:
    """Jeden znaleziony obiekt: ramka w pikselach zdjecia + klasa + pewnosc.

    Po co: wspolny wynik wszystkich detektorow - reszta programu nie wie, z jakiego modelu pochodzi.
           __slots__ = tylko te pola, mniej pamieci przy tysiacach wykryc.

    Pola:
        x0, y0, x1, y1 - rogi ramki w px zdjecia, np. 120.0, 88.5, 150.2, 119.0
        klasa          - nazwa klasy, np. "chwast" ("atrapa" z DummyDetector)
        pewnosc        - pewnosc 0-1, np. 0.42

    Uzywana w:
        wszystkie detektory (ten plik), evaluation.predykcje_detektora(), mission/detect_cli.py, app.py
    """

    __slots__ = ("x0", "y0", "x1", "y1", "klasa", "pewnosc")

    def __init__(self, x0, y0, x1, y1, klasa, pewnosc):
        """Zapisuje ramke, klase i pewnosc (liczby zamieniane na float).

        Argumenty:
            x0, y0, x1, y1 - rogi ramki w px
            klasa          - nazwa klasy, np. "chwast"
            pewnosc        - 0-1
        """
        self.x0, self.y0, self.x1, self.y1 = float(x0), float(y0), float(x1), float(y1)
        self.klasa = klasa
        self.pewnosc = float(pewnosc)

    @property
    def ramka(self):
        """Rogi ramki jako krotka (x0, y0, x1, y1) w px, np. (120.0, 88.5, 150.2, 119.0).

        Uzywana w:
            iou_ramek() przy TiledDetector._scal() i polacz_wbf() (ten plik)
        """
        return self.x0, self.y0, self.x1, self.y1


def iou_ramek(a, b):
    """IoU dwoch ramek: pole czesci wspolnej / pole sumy (0 = rozlaczne, 1 = identyczne).

    Po co: jedna wersja dla calego projektu - ocena modelu, NMS kafelkow, WBF i scalanie
           na mozaice licza IoU tym samym kodem.

    Argumenty:
        a, b - ramki (x0, y0, x1, y1, ...) w px; dalsze pola (np. pewnosc) sa pomijane

    Zwraca:
        float 0-1, np. 0.62

    Uzywana w:
        TiledDetector._scal(), polacz_wbf() (ten plik),
        mission/sim_flight.py scal_ramki(), models/evaluation.py _dopasuj()
    """
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    wsp = (x1 - x0) * (y1 - y0)
    return wsp / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - wsp)


# ---------------------------------------------------------------- YOLO

class YoloDetector:
    """Docelowy detektor. Wymaga: pip install ultralytics + plik modelu .pt.

    Po co: uruchamia model .pt (np. models/weights/fieldeye_final.pt) na Macu albo PC;
           ultralytics sam robi letterbox i NMS.

    Uzywana w:
        _detektor_z_pliku() (ten plik), app.py detektor_lotu()
    """

    def __init__(self, model, prog_pewnosci=0.25, wejscie=640, urzadzenie=None):
        """Wczytuje model .pt i wybiera urzadzenie (CUDA -> MPS -> CPU).

        Argumenty:
            model         - plik .pt (albo .onnx bez onnxruntime - przez ultralytics)
            prog_pewnosci - wykrycia ponizej odpadaja, np. 0.15
            wejscie       - bok wejscia sieci, np. 1280 (jak przy treningu)
            urzadzenie    - None = auto (models/device.py wybierz)
        """
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
        """Wykrywa obiekty na jednym obrazie.

        Argumenty:
            obraz - numpy (h, w, 3) BGR (jak z cv2.imread)

        Zwraca:
            lista Detection w pikselach tego obrazu
        """
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

    Uzywana w:
        mission/detect_cli.py (binarka),
        mission/sim_flight.py, app.py (zakladka 4)
    """

    def __init__(self, detektor, kafelek=896, zakladka=0.15, prog_iou=0.45):
        """Argumenty:
            detektor - detektor wewnetrzny (np. YoloDetector, HailoDetector)
            kafelek  - bok kafelka w px, np. 896 (jak przy treningu)
            zakladka - zakladka sasiednich kafelkow 0-1, np. 0.15
            prog_iou - powyzej tego IoU dwa wykrycia tej samej klasy to duplikat, np. 0.45
        """
        self.detektor = detektor
        self.kafelek = kafelek
        self.zakladka = zakladka
        self.prog_iou = prog_iou
        self.nazwa = f"kafelki({detektor.nazwa})"

    def wykryj(self, obraz):
        """Wykrywa na kafelkach i zwraca wynik w ukladzie calego zdjecia.

        Argumenty:
            obraz - numpy (h, w, 3) BGR; mniejszy niz kafelek idzie do detektora w calosci

        Zwraca:
            lista Detection w pikselach calego zdjecia (bez duplikatow z zakladki)
        """
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
        """Usuwa duplikaty z zakladki (klasyczne NMS - non-maximum suppression).

        Argumenty:
            det - lista Detection ze wszystkich kafelkow

        Zwraca:
            lista Detection - od najpewniejszej; odrzucone te, ktore pokrywaja sie
            (IoU >= prog_iou) z pewniejszym wykryciem tej samej klasy

        Uzywana w:
            wykryj() (ta klasa)
        """
        det = sorted(det, key=lambda d: -d.pewnosc)
        zostaje = []
        for d in det:
            if all(iou_ramek(d.ramka, z.ramka) < self.prog_iou or z.klasa != d.klasa for z in zostaje):
                zostaje.append(d)
        return zostaje


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

    Uzywana w:
        app.py (gdy model sie nie wczyta), mission/detect_cli.py
    """

    def __init__(self, min_pole=120, max_pole_udzial=0.02):
        """Argumenty:
            min_pole        - najmniejsza plama zieleni w px, np. 120
            max_pole_udzial - najwieksza plama jako czesc zdjecia, np. 0.02 (2%)
        """
        self.min_pole = min_pole
        self.max_pole_udzial = max_pole_udzial
        self.nazwa = "ATRAPA (nie jest detektorem chwastow)"
        log.warning("Uzywasz atrapy detektora - wyniki NIE sa detekcja chwastow.")

    def wykryj(self, obraz):
        """Plamy zieleni (indeks ExG, gorne 3% pikseli) -> ramki klasy "atrapa".

        Argumenty:
            obraz - numpy (h, w, 3) BGR

        Zwraca:
            lista Detection; pewnosc 0.5-0.9 to tylko wielkosc plamy, NIE pewnosc modelu
        """
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


# ---------------------------------------------------------------- TTA: kilka przebiegow jednego modelu


def polacz_wbf(listy, iou=0.55):
    """Weighted box fusion: wyniki kilku przebiegow (TTA) tego samego zdjecia -> jedna lista.

    listy: [[Detection, ...] z przebiegu 1, [..] z przebiegu 2, ...]. Ramki,
    ktore sie pokrywaja, laczy w jedna: polozenie to srednia wazona pewnoscia,
    pewnosc to suma najwyzszych pewnosci z kazdego przebiegu / liczba przebiegow.
    Wykrycie widziane w jednym z dwoch przebiegow dostaje wiec polowe pewnosci -
    dlatego wynik z TTA porownujemy glownie miara AP50, niezalezna od progu.

    Argumenty:
        listy - lista list Detection (jedna lista na przebieg)
        iou   - od tego IoU ramki uznaje sie za ten sam obiekt, np. 0.55

    Zwraca:
        lista Detection, np. przebiegi z pewnoscia 0.8 i 0.4 tej samej ramki -> jedna z pewnoscia 0.6

    Uzywana w:
        FlipTTADetector.wykryj() (ten plik)
    """
    n = max(1, len(listy))
    grupy = []                                       # [suma_w, [x0, y0, x1, y1]*w, {przebieg: max}, klasa]
    for k, lista in enumerate(listy):
        for d in lista:
            for g in sorted(grupy, key=lambda g: -max(g[2].values())):
                ramka = [v / g[0] for v in g[1]]
                if g[3] == d.klasa and iou_ramek(d.ramka, ramka) >= iou:
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


class FlipTTADetector:
    """Detekcja na oryginale i odbiciu poziomym (test-time augmentation), laczona przez WBF.

    Z gory chwast wyglada tak samo w odbiciu, ale model widzi go w innym
    ukladzie pikseli i czasem znajduje tylko w jednej wersji. pionowo=True
    dodaje trzeci przebieg z odbiciem pionowym.

    Model koncowy jest oceniany z TTA (ocen-detektor --tta): AP50 0.817 na EWIS1.

    Uzywana w:
        zbuduj_z_plikow() (tta=True), mission/detect_cli.py (--tta)
    """

    def __init__(self, detektor, pionowo=False, iou=0.55):
        """Argumenty:
            detektor - detektor wewnetrzny (np. YoloDetector)
            pionowo  - True = dodatkowo odbicie pionowe (3 przebiegi zamiast 2)
            iou      - prog laczenia ramek w WBF, np. 0.55
        """
        self.detektor = detektor
        self.pionowo = pionowo
        self.iou = iou
        self.nazwa = f"TTA({getattr(detektor, 'nazwa', '?')}{', +pion' if pionowo else ''})"

    def wykryj(self, obraz):
        """Wykrywa na oryginale i odbiciu, odbija ramki z powrotem i laczy przez WBF.

        Argumenty:
            obraz - numpy (h, w, 3) BGR

        Zwraca:
            lista Detection w ukladzie oryginalnego zdjecia
        """
        h, w = obraz.shape[:2]
        listy = [self.detektor.wykryj(obraz)]
        listy.append([Detection(w - d.x1, d.y0, w - d.x0, d.y1, d.klasa, d.pewnosc)
                      for d in self.detektor.wykryj(np.ascontiguousarray(obraz[:, ::-1]))])
        if self.pionowo:
            listy.append([Detection(d.x0, h - d.y1, d.x1, h - d.y0, d.klasa, d.pewnosc)
                          for d in self.detektor.wykryj(np.ascontiguousarray(obraz[::-1]))])
        return polacz_wbf(listy, self.iou)


class OnnxDetector:
    """Model .onnx przez ONNX Runtime - bez PyTorcha i ultralytics (lekka binarka).

    Ten sam model co .pt, wyeksportowany przez `models.cli onnx` (wyjscie
    (1, 4 + liczba klas, N): srodek, szerokosc i wysokosc ramki w pikselach
    wejscia oraz pewnosc kazdej klasy). Obraz przygotowany jak dla Hailo
    (letterbox, RGB, /255), NMS w OpenCV z tymi samymi ustawieniami co
    ultralytics przy przewidywaniu (IoU 0,7, do 300 ramek).

    Uzywana w:
        _detektor_z_pliku() (ten plik) - plik .onnx przy zainstalowanym onnxruntime
    """

    def __init__(self, model, prog_pewnosci=0.25, iou=0.7, maks=300, nazwy=None):
        """Otwiera sesje ONNX Runtime (CUDA, jesli jest, inaczej CPU) i czyta rozmiar wejscia z modelu.

        Argumenty:
            model         - plik .onnx
            prog_pewnosci - wykrycia ponizej odpadaja, np. 0.15
            iou           - prog NMS, np. 0.7
            maks          - najwyzej tyle wykryc, np. 300
            nazwy         - numer klasy -> nazwa, domyslnie {0: "chwast"}
        """
        import onnxruntime as ort
        dostepne = ort.get_available_providers()
        dostawcy = [d for d in ("CUDAExecutionProvider", "CPUExecutionProvider") if d in dostepne]
        self._sesja = ort.InferenceSession(str(model), providers=dostawcy or dostepne)
        we = self._sesja.get_inputs()[0]
        self._we = we.name
        self.wys, self.szer = int(we.shape[2]), int(we.shape[3])
        self.prog, self.iou, self.maks = prog_pewnosci, iou, maks
        self.nazwy = nazwy or {0: "chwast"}
        self.nazwa = f"ONNX({Path(model).name})"

    def wykryj(self, obraz):
        """letterbox -> siec -> prog -> NMS -> ramki z powrotem w pikselach zdjecia.

        Argumenty:
            obraz - numpy (h, w, 3) BGR

        Zwraca:
            lista Detection
        """
        h0, w0 = obraz.shape[:2]
        we, s, px, py = letterbox(obraz, self.wys, self.szer)
        x = np.ascontiguousarray(we.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        p = self._sesja.run(None, {self._we: x})[0][0].T          # (N, 4 + klasy)
        if p.shape[1] > 5:
            klasy = p[:, 4:].argmax(1)
            pewn = p[:, 4:].max(1)
        else:
            klasy, pewn = np.zeros(len(p), int), p[:, 4]
        m = pewn >= self.prog
        if not m.any():
            return []
        cx, cy, w, h = p[m, 0], p[m, 1], p[m, 2], p[m, 3]
        pewn, klasy = pewn[m], klasy[m]
        ramki = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
        idx = cv2.dnn.NMSBoxes(ramki.tolist(), pewn.tolist(), float(self.prog), float(self.iou))
        idx = sorted(np.array(idx).reshape(-1).tolist(), key=lambda i: -pewn[i])[: self.maks]
        out = []
        for i in idx:
            x0, y0, bw, bh = ramki[i]
            X0, X1 = np.clip([(x0 - px) / s, (x0 + bw - px) / s], 0, w0)
            Y0, Y1 = np.clip([(y0 - py) / s, (y0 + bh - py) / s], 0, h0)
            if X1 > X0 and Y1 > Y0:
                out.append(Detection(X0, Y0, X1, Y1, self.nazwy.get(int(klasy[i]), str(klasy[i])), float(pewn[i])))
        return out


def _detektor_z_pliku(m, prog, wejscie, urzadzenie):
    """Wybiera detektor po rozszerzeniu pliku modelu.

    Argumenty:
        m          - plik modelu: .hef -> HailoDetector, .onnx (z onnxruntime) -> OnnxDetector,
                     inne (.pt) -> YoloDetector
        prog       - prog pewnosci
        wejscie    - bok wejscia sieci (dla .pt)
        urzadzenie - None = auto (dla .pt)

    Zwraca:
        detektor

    Uzywana w:
        zbuduj_z_plikow() (ten plik)
    """
    m = str(m)
    if m.lower().endswith(".hef"):
        return HailoDetector(m, prog)
    if m.lower().endswith(".onnx") and importlib.util.find_spec("onnxruntime") is not None:
        return OnnxDetector(m, prog)               # bez onnxruntime - przez ultralytics (YoloDetector)
    return YoloDetector(m, prog, wejscie, urzadzenie)


def zbuduj_z_plikow(model, prog=0.15, wejscie=1280, urzadzenie=None, tta=False):
    """Plik modelu (.pt / .onnx / .hef) -> detektor, opcjonalnie z TTA.

    Argumenty:
        model      - plik modelu, np. "models/weights/fieldeye_final.pt"
        prog       - prog pewnosci, np. 0.15 (do AP50: 0.001)
        wejscie    - bok wejscia sieci dla .pt, np. 1280
        urzadzenie - None = auto
        tta        - True = FlipTTADetector (oryginal + odbicie poziome)

    Zwraca:
        detektor, np. FlipTTADetector(YoloDetector(...))

    Uzywana w:
        models/cli.py cmd_ocen_detektor() (ocen-detektor w final_model.sh i check_hef.sh),
        mission/detect_cli.py (binarka), tests
    """
    d = _detektor_z_pliku(model, prog, wejscie, urzadzenie)
    return FlipTTADetector(d) if tta else d


# ---------------------------------------------------------------- Hailo (.hef)

def letterbox(obraz, wys, szer, wypelnienie=114):
    """Wpisuje obraz BGR w prostokat wys x szer z zachowaniem proporcji.

    Tak samo jak ultralytics przy treningu: zmniejszenie albo powiekszenie,
    a reszta wypelniona szaroscia (114). HEF ma staly rozmiar wejscia, wiec
    to przygotowanie musimy zrobic sami. Zwraca obraz RGB uint8 oraz skale
    i przesuniecie, potrzebne do przeliczenia ramek z powrotem.

    Argumenty:
        obraz       - numpy (h, w, 3) BGR
        wys, szer   - rozmiar wejscia sieci, np. 1280, 1280
        wypelnienie - kolor pasow, 114 (jak w ultralytics)

    Zwraca:
        (obraz_rgb, s, px, py) - np. obraz 600x300 w 640x640: s = 1.067, px = 0, py = 160

    Uzywana w:
        OnnxDetector.wykryj(), HailoDetector.wykryj() (ten plik)
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

    Argumenty:
        wyjscie        - wynik HEF dla jednego obrazu (lista tablic, jedna na klase)
        wys, szer      - rozmiar wejscia sieci
        skala, px, py  - z letterbox()
        w0, h0         - rozmiar oryginalnego zdjecia
        prog           - wykrycia ponizej odpadaja, np. 0.15
        nazwy          - numer klasy -> nazwa, domyslnie {0: "chwast"}

    Zwraca:
        lista Detection (ramki w calosci w szarym marginesie odpadaja)

    Uzywana w:
        HailoDetector.wykryj() (ten plik), tests/test_hailo.py
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

    Uzywana w:
        _detektor_z_pliku() (ten plik), app.py detektor_lotu()
    """

    def __init__(self, hef, prog_pewnosci=0.15, nazwy=None):
        """Wczytuje .hef, konfiguruje akcelerator i otwiera potok (raz, nie przy kazdym kadrze).

        Argumenty:
            hef           - plik .hef, np. "fieldeye_final_1280_h8.hef"
            prog_pewnosci - wykrycia ponizej odpadaja, np. 0.15
            nazwy         - numer klasy -> nazwa albo None ({0: "chwast"})
        """
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
        """letterbox -> akcelerator Hailo -> nms_hailo_na_detekcje().

        Argumenty:
            obraz - numpy (h, w, 3) BGR

        Zwraca:
            lista Detection w pikselach zdjecia
        """
        h0, w0 = obraz.shape[:2]
        we, s, px, py = letterbox(obraz, self.wys, self.szer)
        wynik = self._potok.infer({self._we_nazwa: we[None]})[self._wy_nazwa]
        return nms_hailo_na_detekcje(wynik[0], self.wys, self.szer, s, px, py, w0, h0,
                                     self.prog, self.nazwy)

    def zamknij(self):
        """Zamyka potok i zwalnia akcelerator (bez tego kolejny program nie dostanie Hailo).

        Uzywana w:
            models/cli.py cmd_ocen_detektor(), mission/detect_cli.py
        """
        for kontekst in ("_aktywacja", "_potok"):
            k = getattr(self, kontekst, None)
            if k is not None:
                k.__exit__(None, None, None)
                setattr(self, kontekst, None)

