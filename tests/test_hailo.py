"""Testy Hailo (.hef) bez sprzetu: przygotowanie obrazu, odczyt wyniku NMS, wybor
detektora, komunikat bez HailoRT oraz ocena dowolnego detektora (AP50).

Po co: akcelerator Hailo-8 jest tylko na Raspberry Pi, a wiekszosc logiki wokol niego
(letterbox, przeliczanie ramek z formatu Hailo, wybor detektora po rozszerzeniu pliku,
ocena AP50) da sie sprawdzic na zwyklym komputerze z atrapami.

Testowane: mission/detector.py (letterbox, nms_hailo_na_detekcje, HailoDetector,
           zbuduj_detektor), models/evaluation.py (ap50, ocen_detektor),
           models/export.py (przygotuj_kalibracje), models/cli.py (parser),
           mission/config.py (MissionConfig + configs/*.yaml),
           skladnia scripts/final_model.sh i scripts/check_hef.sh.

    python -m pytest tests/test_hailo.py -q
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mission.detector import Detection, letterbox, nms_hailo_na_detekcje  # noqa: E402


def _hailo_z_ramek(ramki_px, w0, h0, wej):
    """Odwrotnosc nms_hailo_na_detekcje: ramki w pikselach oryginalu -> format Hailo.

    Po co: robi sztuczne wyjscie Hailo, zeby sprawdzic, czy przeliczenie wraca do tych samych ramek.

    Argumenty:
        ramki_px - lista (x0, y0, x1, y1, pewnosc) w pikselach oryginalu, np. [(100, 200, 160, 260, 0.9)]
        w0, h0   - rozmiar oryginalu w px, np. 600, 600
        wej      - bok wejscia sieci (kwadrat), np. 640

    Zwraca:
        (wyjscie, s, px, py) - wyjscie: lista z jedna tablica [y0, x0, y1, x1, c] w ulamkach 0-1,
        s: skala letterboxa, px/py: margines w px, np. ([array(...)], 1.067, 0, 0)

    Uzywana w:
        test_ramki_wracaja_na_swoje_miejsce()
    """
    s = min(wej / w0, wej / h0)
    px, py = (wej - round(w0 * s)) // 2, (wej - round(h0 * s)) // 2
    wiersze = [[(y0 * s + py) / wej, (x0 * s + px) / wej, (y1 * s + py) / wej, (x1 * s + px) / wej, c]
               for x0, y0, x1, y1, c in ramki_px]
    return [np.array(wiersze, np.float32)], s, px, py


@pytest.mark.parametrize("w0,h0,wej", [(600, 600, 640), (600, 600, 1280), (4056, 3040, 640), (700, 500, 1280)])
def test_ramki_wracaja_na_swoje_miejsce(w0, h0, wej):
    """Ramka przeliczona do formatu Hailo i z powrotem trafia w to samo miejsce.

    Po co: blad w przeliczeniu (skala, margines) przesunalby wszystkie wykrycia na Pi.

    Argumenty:
        w0, h0 - rozmiar oryginalu w px (z parametrize), np. 4056, 3040
        wej    - bok wejscia sieci, np. 640 albo 1280

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    ramki = [(100, 200, 160, 260, 0.9), (10, 20, 30, 45, 0.3)]
    wyjscie, s, px, py = _hailo_z_ramek(ramki, w0, h0, wej)
    det = nms_hailo_na_detekcje(wyjscie, wej, wej, s, px, py, w0, h0)
    assert len(det) == 2
    for d, (x0, y0, x1, y1, c) in zip(det, ramki):
        assert abs(d.x0 - x0) < 0.6 and abs(d.y0 - y0) < 0.6
        assert abs(d.x1 - x1) < 0.6 and abs(d.y1 - y1) < 0.6
        assert d.klasa == "chwast" and abs(d.pewnosc - c) < 1e-6


def test_kolejnosc_y_przed_x():
    """Hailo zapisuje [y0, x0, y1, x1] - zamiana kolejnosci dalaby ramke obrocona.

    Po co: pilnuje kolejnosci wspolrzednych w odczycie wyniku NMS z Hailo.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    det = nms_hailo_na_detekcje([np.array([[0.1, 0.5, 0.2, 0.9, 0.8]], np.float32)],
                                100, 100, 1.0, 0, 0, 100, 100)
    d = det[0]
    assert (d.x0, d.y0, d.x1, d.y1) == (50, 10, 90, 20)


def test_prog_i_puste_klasy():
    """Wykrycia ponizej progu odpadaja, a pusta tablica klasy niczego nie psuje.

    Po co: Hailo zwraca osobna tablice na kazda klase, czesto pusta; prog 0.15 ma dzialac.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    wyjscie = [np.array([[0.1, 0.1, 0.2, 0.2, 0.10], [0.3, 0.3, 0.4, 0.4, 0.50]], np.float32),
               np.zeros((0, 5), np.float32)]
    det = nms_hailo_na_detekcje(wyjscie, 640, 640, 1.0, 0, 0, 640, 640, prog=0.15)
    assert len(det) == 1 and det[0].pewnosc == pytest.approx(0.5)


def test_ramka_w_paddingu_odpada():
    """Wykrycie w szarym marginesie letterboxa nie istnieje na prawdziwym zdjeciu.

    Po co: margines nie jest czescia pola - ramka w calosci w nim musi zostac odrzucona.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    # obraz 640x320 w wejsciu 640: margines 160 px na gorze i na dole
    det = nms_hailo_na_detekcje([np.array([[0.0, 0.1, 0.2, 0.2, 0.9]], np.float32)],
                                640, 640, 1.0, 0, 160, 640, 320)
    assert det == []


def test_letterbox_rozmiar_kolory_margines():
    """Letterbox: wejscie 640x640, kolory BGR -> RGB, szary margines (114, 114, 114).

    Po co: Hailo dostaje obraz w RGB i w rozmiarze sieci - zle kolory albo skala
           obnizylyby wykrywalnosc na Pi.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    img = np.zeros((300, 600, 3), np.uint8)
    img[:, :, 0] = 255                                   # niebieski w BGR
    we, s, px, py = letterbox(img, 640, 640)
    assert we.shape == (640, 640, 3) and we.dtype == np.uint8
    assert s == pytest.approx(640 / 600) and px == 0 and py == 160
    assert tuple(we[320, 320]) == (0, 0, 255)            # RGB: niebieski na koncu
    assert tuple(we[10, 320]) == (114, 114, 114)         # margines


def test_detektor_hailo_bez_hailort(monkeypatch, tmp_path):
    """Bez pakietu hailo_platform HailoDetector daje czytelny blad z nazwa HailoRT.

    Po co: na komputerze bez Hailo uzytkownik ma zobaczyc, czego brakuje, a nie dziwny traceback.

    Argumenty:
        monkeypatch - fixture pytest: podmienia builtins.__import__ na czas testu
        tmp_path    - fixture pytest: folder na nieistniejacy plik m.hef

    Zwraca:
        nic - pytest sprawdza, ze poleci ImportError

    Uzywana w:
        pytest (python -m pytest tests)
    """
    import builtins
    from mission.detector import HailoDetector
    prawdziwy = builtins.__import__

    def bez_hailo(name, *a, **k):
        """Import jak zwykle, tylko hailo_platform udaje, ze go nie ma.

        Argumenty: name - nazwa modulu, a/k - reszta argumentow __import__
        Zwraca: zaimportowany modul albo ImportError dla hailo_platform
        Uzywana w: test_detektor_hailo_bez_hailort() przez monkeypatch
        """
        if name == "hailo_platform":
            raise ImportError("brak")
        return prawdziwy(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", bez_hailo)
    with pytest.raises(ImportError, match="HailoRT"):
        HailoDetector(tmp_path / "m.hef")


def test_zbuduj_detektor_wybiera_hailo_dla_hef(monkeypatch):
    """Model z rozszerzeniem .hef (tez .HEF) -> HailoDetector, rowniez z kafelkowaniem.

    Po co: na Pi ten sam kod misji ma sam wybrac Hailo po rozszerzeniu pliku modelu.

    Argumenty:
        monkeypatch - fixture pytest: podmienia HailoDetector na atrape

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    from mission import detector as dm

    class _FakeHailo:
        """Atrapa HailoDetector - tylko zapamietuje model i prog."""

        def __init__(self, model, prog):
            """Argumenty: model - sciezka modelu, prog - prog pewnosci. Uzywana w: zbuduj_detektor()."""
            self.model, self.prog, self.nazwa = model, prog, "atrapa-hailo"
    monkeypatch.setattr(dm, "HailoDetector", _FakeHailo)

    class Cfg:
        model, prog_pewnosci, wejscie_sieci = "model.HEF", 0.15, 640
        kafelkowanie, kafelek, zakladka = False, 896, 0.15
    d = dm.zbuduj_detektor(Cfg)
    assert isinstance(d, _FakeHailo) and d.prog == 0.15
    Cfg.kafelkowanie = True
    assert isinstance(dm.zbuduj_detektor(Cfg).detektor, _FakeHailo)   # kafelkowanie dziala tez z Hailo


# ---------------------------------------------------------------- ocena dowolnego detektora

def test_ap50_idealny_i_bez_wykryc():
    """AP50 = 1 dla idealnych wykryc i 0, gdy nic nie wykryto.

    Po co: punkty brzegowe miary, ktora porownujemy .pt, .onnx i .hef.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    from models.evaluation import ap50
    prawda = {"a": [(0, 0, 0, 10, 10), (0, 20, 20, 30, 30)]}
    assert ap50({"a": [(0, 0.9, 0, 0, 10, 10), (0, 0.8, 20, 20, 30, 30)]}, prawda) == pytest.approx(1.0, abs=0.01)
    assert ap50({"a": []}, prawda) == 0.0


def test_ap50_polowa_znaleziona():
    """Jeden trafiony i jeden falszywy -> AP50 0.625, tak samo jak liczy ultralytics.

    Po co: nasza ocena (tez dla Hailo) ma dawac te same liczby co ultralytics dla .pt.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    from models.evaluation import ap50
    prawda = {"a": [(0, 0, 0, 10, 10), (0, 20, 20, 30, 30)]}
    # jeden trafiony, jeden fałszywy z niższą pewnością. Tak jak w ultralytics krzywa
    # kończy się punktem (czułość 1, precyzja 0), połączonym linią z ostatnim punktem
    # modelu: pole = 0,5 · 1 (czułość 0–0,5) + trójkąt 0,5 · 0,5 / 2 = 0,625
    assert ap50({"a": [(0, 0.9, 0, 0, 10, 10), (0, 0.5, 50, 50, 60, 60)]}, prawda) == pytest.approx(0.625, abs=0.01)


class _GroundTruthDetector:
    """Detektor, ktory 'widzi' dokladnie ramki z etykiet - do sprawdzenia calej oceny.

    Po co: z idealnym detektorem ocena musi dac AP50 = 1 i czulosc 1 - inaczej blad jest w ocenie.

    Uzywana w:
        test_ocen_detektor_cala_sciezka()
    """
    nazwa = "prawda"

    def __init__(self, etykiety):
        """Zapamietuje etykiety wszystkich obrazow.

        Argumenty:
            etykiety - slownik numer_obrazu -> lista (cx, cy, w, h) w ulamkach 0-1,
                       np. {0: [(0.5, 0.5, 0.25, 0.25)]}

        Zwraca:
            nic

        Uzywana w:
            test_ocen_detektor_cala_sciezka()
        """
        self.etykiety = etykiety

    def wykryj(self, obraz):
        """Zwraca ramki z etykiet obrazu (numer obrazu zapisany w pikselu [0, 0]).

        Argumenty:
            obraz - numpy (h, w, 3) uint8

        Zwraca:
            lista Detection w pikselach, pewnosc 0.9, np. [Detection(24, 24, 40, 40, "chwast", 0.9)]

        Uzywana w:
            models/evaluation.py ocen_detektor() - tak jak prawdziwy detektor
        """
        h, w = obraz.shape[:2]
        klucz = int(obraz[0, 0, 0])                      # numer obrazu zapisany w pikselu
        out = []
        for cx, cy, bw, bh in self.etykiety[klucz]:
            out.append(Detection((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h,
                                "chwast", 0.9))
        return out


def test_ocen_detektor_cala_sciezka(tmp_path):
    """Cala ocena na malym zbiorze: 2 obrazy z 2 dni, idealny detektor -> AP50 1, czulosc 1.

    Po co: sprawdza droge, ktora idzie ocen-detektor w final_model.sh - wczytanie data.yaml,
           etykiet, wykrycia, AP50, wynik przy progu i podzial wedlug dni.

    Argumenty:
        tmp_path - fixture pytest: folder na sztuczny zbior YOLO

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    from models.evaluation import ocen_detektor
    ds = tmp_path / "ds"
    (ds / "images" / "test").mkdir(parents=True)
    (ds / "labels" / "test").mkdir(parents=True)
    etykiety = {}
    for i, dzien in enumerate(["2023-05-25", "2023-06-06"]):
        img = np.full((64, 64, 3), 50, np.uint8)
        img[0, 0, 0] = i
        nazwa = f"weedsgalore__{dzien}__{dzien}_{i:04d}"
        cv2.imwrite(str(ds / "images" / "test" / f"{nazwa}.png"), img)
        etykiety[i] = [(0.5, 0.5, 0.25, 0.25), (0.2, 0.2, 0.2, 0.2)]
        (ds / "labels" / "test" / f"{nazwa}.txt").write_text(
            "\n".join(f"0 {cx} {cy} {bw} {bh}" for cx, cy, bw, bh in etykiety[i]))
    (ds / "data.yaml").write_text(yaml.safe_dump({"path": str(ds), "train": "images/test", "val": "images/test",
                                                  "test": "images/test", "names": {0: "chwast"}}))
    w = ocen_detektor(_GroundTruthDetector(etykiety), ds / "data.yaml", "test", prog=0.15)
    assert w["obrazow"] == 2
    assert w["AP50"] == pytest.approx(1.0, abs=0.01)
    assert w["przy_progu"]["czulosc"] == 1.0 and w["przy_progu"]["falszywe"] == 0
    assert set(w["wedlug_dni"]) == {"2023-05-25", "2023-06-06", "razem"}


def test_cli_ocen_detektor_i_kalibracja():
    """Domyslne opcje komend ocen-detektor (split test, wejscie 1280) i kalibracja.

    Po co: final_model.sh polega na tych domyslnych - ich zmiana zmienilaby wyniki.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    from models.cli import parser
    a = parser().parse_args(["ocen-detektor", "m.hef", "d.yaml", "--prog", "0.15"])
    assert a.model == ["m.hef"] and a.split == "test" and a.wejscie == 1280
    a = parser().parse_args(["kalibracja", "obrazy", "--rozmiar", "640"])
    assert a.rozmiar == 640


def test_kalibracja_w_rozmiarze_wejscia(tmp_path):
    """przygotuj_kalibracje: zdjecia kalibracyjne w rozmiarze wejscia sieci + lista plikow.

    Po co: kompilator Hailo kwantyzuje model na tych zdjeciach - zly rozmiar psuje HEF.

    Argumenty:
        tmp_path - fixture pytest: folder na zdjecia zrodlowe i wynik

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    from models.export import przygotuj_kalibracje
    (tmp_path / "src").mkdir()
    for i in range(3):
        cv2.imwrite(str(tmp_path / "src" / f"k{i}.jpg"), np.full((600, 600, 3), 80, np.uint8))
    kat, lista = przygotuj_kalibracje(tmp_path / "src", tmp_path / "kal", liczba=3, rozmiar=640)
    pliki = sorted(Path(kat).glob("*.jpg"))
    assert len(pliki) == 3 and cv2.imread(str(pliki[0])).shape == (640, 640, 3)
    assert len(Path(lista).read_text().split()) == 3


# ---------------------------------------------------------------- konfiguracje i skrypty

def test_konfiguracje_misji_i_domyslne():
    """Konfiguracje misji: domyslne (prog 0.15, wejscie 1280), wersja .pt i wersja .hef.

    Po co: configs/mission_wg_sudars.yaml i mission_hailo.yaml maja wskazywac model koncowy
           z tymi samymi ustawieniami co ocena.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    from mission.config import MissionConfig
    k = MissionConfig()
    assert k.detekcja.prog_pewnosci == 0.15 and k.detekcja.wejscie_sieci == 1280
    pt = MissionConfig.wczytaj(ROOT / "configs" / "mission_wg_sudars.yaml")
    assert pt.detekcja.model.endswith(".pt") and pt.detekcja.wejscie_sieci == 1280
    hef = MissionConfig.wczytaj(ROOT / "configs" / "mission_hailo.yaml")
    assert hef.detekcja.model.endswith(".hef") and hef.detekcja.prog_pewnosci == 0.15


@pytest.mark.parametrize("skrypt", ["final_model.sh", "check_hef.sh"])
def test_skrypty_bez_bledow_skladni(skrypt):
    """bash -n: skrypt modelu koncowego i skrypt sprawdzenia HEF bez bledow skladni.

    Po co: oba odpala sie rzadko (trening 4 h, Pi) - literowka wyszlaby dopiero tam.

    Argumenty:
        skrypt - nazwa pliku w scripts/ (z parametrize), np. "final_model.sh"

    Zwraca:
        nic - pytest sprawdza kod wyjscia bash -n

    Uzywana w:
        pytest (python -m pytest tests)
    """
    import subprocess
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / skrypt)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
