"""Testy detektorow bez prawdziwego modelu: IoU, TTA z odbiciem (WBF), kafelki, atrapa.

Po co: model koncowy dziala z TTA (FlipTTADetector) - testy pilnuja, ze wykrycia
z odbitego obrazu wracaja na swoje miejsce, a laczenie ramek (WBF) dobrze usrednia
pewnosc. Kafelki (TiledDetector) musza oddac ramki we wspolrzednych calego zdjecia.

Testowane: mission/detector.py (iou_ramek, polacz_wbf, FlipTTADetector, TiledDetector,
           DummyDetector, zbuduj_z_plikow), parser models/cli.py (ocen-detektor).

    python -m pytest tests/test_inference.py -q
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mission.detector import (Detection, DummyDetector, FlipTTADetector,  # noqa: E402
                              TiledDetector, iou_ramek, polacz_wbf)


def D(x0, y0, x1, y1, c):
    """Skrot: tworzy wykrycie chwastu.

    Po co: krotszy zapis ramek w testach.

    Argumenty:
        x0, y0, x1, y1 - rogi ramki w px, np. 10, 10, 20, 20
        c              - pewnosc 0-1, np. 0.8

    Zwraca:
        Detection, np. Detection(10, 10, 20, 20, "chwast", 0.8)

    Uzywana w:
        testach WBF, _Staly.wykryj(), _Magenta.wykryj()
    """
    return Detection(x0, y0, x1, y1, "chwast", c)


def test_wbf_laczy_pokrywajace_sie_ramki_i_usrednia_pewnosc():
    """Dwie nakladajace sie ramki z 2 przebiegow -> jedna, pewnosc srednia, polozenie wazone.

    Po co: tak TTA i ensemble lacza wyniki - zla srednia zmienilaby AP50.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    w = polacz_wbf([[D(10, 10, 20, 20, 0.8)], [D(12, 10, 22, 20, 0.4)]])
    assert len(w) == 1
    d = w[0]
    assert d.pewnosc == pytest.approx(0.6)                       # (0,8 + 0,4) / 2 przebiegi
    assert d.x0 == pytest.approx((10 * 0.8 + 12 * 0.4) / 1.2)    # srednia wazona pewnoscia


def test_wbf_wykrycie_z_jednego_przebiegu_dostaje_polowe():
    """Ramka znaleziona tylko w 1 z 2 przebiegow dostaje polowe pewnosci.

    Po co: wykrycie potwierdzone tylko raz jest mniej pewne - to tlumi falszywki.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    w = polacz_wbf([[D(10, 10, 20, 20, 0.8)], []])
    assert len(w) == 1 and w[0].pewnosc == pytest.approx(0.4)


def test_wbf_rozne_miejsca_osobno_i_powtorzenie_w_jednym_przebiegu():
    """Ramki w roznych miejscach zostaja osobno; z dubli w 1 przebiegu liczy sie najwyzsza.

    Po co: WBF nie moze zlewac roznych chwastow ani podbijac pewnosci dublami z jednego przebiegu.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    w = polacz_wbf([[D(10, 10, 20, 20, 0.8), D(11, 10, 21, 20, 0.7), D(100, 100, 110, 110, 0.5)],
                    [D(10, 11, 20, 21, 0.6)]])
    assert len(w) == 2
    duza = max(w, key=lambda d: d.pewnosc)
    assert duza.pewnosc == pytest.approx((0.8 + 0.6) / 2)        # z przebiegu 1 liczy sie najwyzsza


class _Staly:
    """Atrapa detektora: zawsze zwraca te same ramki.

    Uzywana w:
        test_zbuduj_z_plikow(), testy kafelkow
    """

    def __init__(self, ramki, nazwa="m"):
        """Argumenty: ramki - lista (x0, y0, x1, y1, c), nazwa - nazwa modelu. Zwraca: nic."""
        self.ramki, self.nazwa = ramki, nazwa

    def wykryj(self, img):
        """Argumenty: img - obraz (ignorowany). Zwraca: lista Detection z self.ramki."""
        return [D(*r) for r in self.ramki]


class _Magenta:
    """Znajduje magentowe kwadraty - po odbiciu wyniki musza wrocic na to samo miejsce.

    Uzywana w:
        test_tta_odbicie_wraca_na_to_samo_miejsce()
    """
    nazwa = "magenta"

    def wykryj(self, img):
        """Argumenty: img - obraz BGR. Zwraca: lista Detection, jedna na kazdy magentowy kwadrat."""
        m = cv2.inRange(img, (200, 0, 200), (255, 80, 255))
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        return [D(x, y, x + w, y + h, 0.5) for x, y, w, h, a in st[1:] if a > 5]


def test_tta_odbicie_wraca_na_to_samo_miejsce():
    """TTA z odbiciem (poziomo + pionowo): ramka po odbiciu wraca dokladnie na swoje miejsce.

    Po co: model koncowy dziala z TTA - zle odwrocenie odbicia przesuneloby wszystkie wykrycia.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    img = np.zeros((100, 160, 3), np.uint8)
    cv2.rectangle(img, (20, 30), (35, 44), (255, 0, 255), -1)
    w = FlipTTADetector(_Magenta(), pionowo=True).wykryj(img)
    assert len(w) == 1
    d = w[0]
    assert (d.x0, d.y0, d.x1, d.y1) == pytest.approx((20, 30, 36, 45), abs=0.01)
    assert d.pewnosc == pytest.approx(0.5)                       # znaleziony w kazdym przebiegu


def test_ocen_detektor_przyjmuje_model_i_tta():
    """ocen-detektor: jeden plik modelu, data.yaml i opcja --tta.

    Po co: pilnuje opcji, ktorych uzywa final_model.sh (ocen-detektor ... --tta).

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    from models.cli import parser
    a = parser().parse_args(["ocen-detektor", "a.pt", "d.yaml", "--tta"])
    assert a.model == "a.pt" and a.dane == "d.yaml" and a.tta


def test_zbuduj_z_plikow(monkeypatch):
    """zbuduj_z_plikow: tta=True -> FlipTTA, bez TTA -> sam model.

    Po co: ta funkcja buduje detektor dla ocen-detektor i binarki - musi wybrac dobry typ.

    Argumenty:
        monkeypatch - fixture pytest: podmienia YoloDetector na atrape _Staly

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    from mission import detector as dm
    monkeypatch.setattr(dm, "YoloDetector", lambda m, *a, **k: _Staly([], str(m)))
    assert isinstance(dm.zbuduj_z_plikow("a.pt", tta=True), FlipTTADetector)
    assert dm.zbuduj_z_plikow("a.pt").nazwa == "a.pt"


def test_iou_ramek():
    """IoU: identyczne ramki 1, rozlaczne 0, polowa pokrycia 1/3.

    Po co: jedno IoU liczy ocene modelu, NMS kafelkow, WBF i scalanie na mozaice.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    assert iou_ramek((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou_ramek((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou_ramek((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iou_ramek((0, 0, 10, 10, 0.9), (0, 0, 10, 10)) == pytest.approx(1.0)   # pewnosc pomijana


def test_atrapa_zwraca_ramki_w_obrazie():
    """Atrapa (plamy zieleni) zwraca ramki lezace w obrazie i z klasa "atrapa".

    Po co: atrapa uruchamia caly potok bez modelu - jej wynik nie moze udawac chwastu.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    img = np.full((300, 400, 3), (60, 70, 90), np.uint8)
    cv2.circle(img, (200, 150), 20, (60, 240, 110), -1)
    det = DummyDetector().wykryj(img)
    assert len(det) >= 1
    for d in det:
        assert 0 <= d.x0 < d.x1 <= 400 and 0 <= d.y0 < d.y1 <= 300
        assert d.klasa == "atrapa"


def test_kafelki_przeliczaja_wspolrzedne_na_caly_kadr():
    """Detekcja z kafelka trafia we wlasciwe miejsce calego zdjecia.

    Po co: binarka i zakladka 4 tna duze zdjecie na kafelki - zle przesuniecie = zle namiary.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    d = TiledDetector(_Staly([(10, 10, 30, 30, 0.9)]), kafelek=100, zakladka=0.0)
    wynik = d.wykryj(np.zeros((300, 300, 3), np.uint8))
    assert len(wynik) == 9                        # siatka 3x3 kafelkow
    rogi = {(int(r.x0), int(r.y0)) for r in wynik}
    assert (10, 10) in rogi and (110, 110) in rogi and (210, 210) in rogi


def test_kafelki_usuwaja_duplikaty_z_zakladki():
    """Te same ramki z zachodzacych kafelkow sa scalane (NMS), maly kadr idzie w calosci.

    Po co: bez scalania jeden chwast z zakladki bylby liczony kilka razy.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    d = TiledDetector(_Staly([(0, 0, 100, 100, 0.9)]), kafelek=100, zakladka=0.5, prog_iou=0.3)
    assert len(d.wykryj(np.zeros((150, 150, 3), np.uint8))) < 4
    maly = TiledDetector(_Staly([(1, 1, 5, 5, 0.5)]), kafelek=896)
    assert len(maly.wykryj(np.zeros((640, 640, 3), np.uint8))) == 1
