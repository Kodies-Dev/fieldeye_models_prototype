"""Testy popraw bez treningu: ensemble (WBF), TTA z odbiciem, niski prog z glosowaniem.

Po co: to sposoby na lepsza wykrywalnosc bez nowego treningu. Model koncowy dziala
z TTA (FlipTTADetector) - testy pilnuja, ze wykrycia z odbitego obrazu wracaja na
swoje miejsce, a laczenie ramek (WBF) dobrze usrednia pewnosc.

Testowane: mission/detector.py (polacz_wbf, EnsembleDetector, FlipTTADetector,
           zbuduj_z_plikow), mission/fusion.py (polacz), parsery models/cli.py i mission/cli.py.

    python -m pytest tests/test_inference.py -q
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mission.detector import (Detection, EnsembleDetector, FlipTTADetector,  # noqa: E402
                              polacz_wbf)
from mission.fusion import polacz  # noqa: E402


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
        test_ensemble_dwoch_modeli(), test_zbuduj_z_plikow()
    """

    def __init__(self, ramki, nazwa="m"):
        """Argumenty: ramki - lista (x0, y0, x1, y1, c), nazwa - nazwa modelu. Zwraca: nic."""
        self.ramki, self.nazwa = ramki, nazwa

    def wykryj(self, img):
        """Argumenty: img - obraz (ignorowany). Zwraca: lista Detection z self.ramki."""
        return [D(*r) for r in self.ramki]


def test_ensemble_dwoch_modeli():
    """Ensemble 2 modeli: ta sama ramka laczona w jedna, pewnosc srednia, nazwa "Ensemble(a + b)".

    Po co: sprawdza EnsembleDetector bez prawdziwych modeli.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    e = EnsembleDetector([_Staly([(10, 10, 20, 20, 0.6)], "a"), _Staly([(10, 10, 20, 20, 0.2)], "b")])
    w = e.wykryj(np.zeros((50, 50, 3), np.uint8))
    assert len(w) == 1 and w[0].pewnosc == pytest.approx(0.4)
    assert e.nazwa == "Ensemble(a + b)"


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


def test_niski_prog_z_glosowaniem():
    """Slabe wykrycia (ponizej 0.15) przechodza tylko, gdy widac je na 2 kadrach.

    Po co: laczenie wykryc z wielu zdjec w locie (fusion.polacz) - niski prog z glosowaniem
           lapie male chwasty bez zalewu falszywek.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    dwa_slabe = [(10, 10, 20, 20, 0.07, 0), (10, 10, 20, 20, 0.06, 1)]
    assert len(polacz(dwa_slabe, prog=0.15, prog_niski=0.05)) == 1   # 2 kadry -> przyjety
    assert polacz(dwa_slabe, prog=0.15) == []                        # noisy-OR 0,13 < 0,15
    jedno_slabe = [(10, 10, 20, 20, 0.12, 0)]
    assert polacz(jedno_slabe, prog=0.15, prog_niski=0.05) == []     # 1 kadr -> odpada
    silne = [(10, 10, 20, 20, 0.5, 0)]
    assert len(polacz(silne, prog=0.15, prog_niski=0.05)) == 1       # silne jak zwykle


def test_komendy_przyjmuja_kilka_modeli_i_tta():
    """ocen-detektor i symulacja przyjmuja kilka modeli (ensemble) i opcje --tta.

    Po co: pilnuje opcji, ktorych uzywa final_model.sh (ocen-detektor ... --tta).

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    from mission.cli import parser as mp
    from models.cli import parser
    a = parser().parse_args(["ocen-detektor", "a.pt", "b.pt", "d.yaml", "--tta"])
    assert a.model == ["a.pt", "b.pt"] and a.dane == "d.yaml" and a.tta
    b = mp().parse_args(["symulacja", "--model", "a.pt", "b.pt", "--tta"])
    assert b.model == ["a.pt", "b.pt"] and b.tta


def test_zbuduj_z_plikow(monkeypatch):
    """zbuduj_z_plikow: kilka modeli -> Ensemble, tta=True -> FlipTTA, jeden -> sam model.

    Po co: ta funkcja buduje detektor dla ocen-detektor i aplikacji - musi wybrac dobry typ.

    Argumenty:
        monkeypatch - fixture pytest: podmienia YoloDetector na atrape _Staly

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    from mission import detector as dm
    monkeypatch.setattr(dm, "YoloDetector", lambda m, *a, **k: _Staly([], str(m)))
    assert isinstance(dm.zbuduj_z_plikow(["a.pt", "b.pt"]), EnsembleDetector)
    assert isinstance(dm.zbuduj_z_plikow("a.pt", tta=True), FlipTTADetector)
    assert dm.zbuduj_z_plikow("a.pt").nazwa == "a.pt"
