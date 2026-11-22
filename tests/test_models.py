"""Testy modulu modele.

Sam trening wymaga pakietu ultralytics i karty graficznej, wiec nie jest tu
uruchamiany. Testowane jest wszystko wokol: plan, kontrola danych przed
treningiem, wczytywanie etykiet, miary oceny, dobor zbioru kalibracyjnego
i komunikaty bledow przy braku ultralytics.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.evaluation import wczytaj_etykiety  # noqa: E402
from models.plan import Stage, TrainingPlan, katalog_etykiet, sprawdz_dane  # noqa: E402

yaml = pytest.importorskip("yaml")


def _zbior(katalog, ile_train=6, ile_val=2, ile_test=2, rozmiar=(640, 640),
           klasy=("chwast", "uprawa"), ramki_na_obraz=3):
    """Minimalny zbior w formacie YOLO."""
    katalog = Path(katalog)
    rng = np.random.default_rng(0)
    for split, ile in (("train", ile_train), ("val", ile_val), ("test", ile_test)):
        (katalog / "images" / split).mkdir(parents=True, exist_ok=True)
        (katalog / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(ile):
            nazwa = f"{split}_{i:03d}"
            cv2.imwrite(str(katalog / "images" / split / f"{nazwa}.jpg"),
                        rng.integers(0, 255, (rozmiar[1], rozmiar[0], 3), dtype=np.uint8))
            linie = []
            for j in range(ramki_na_obraz):
                bok = 0.03 + 0.05 * j            # male, srednie, wieksze
                linie.append(f"{j % len(klasy)} {0.2 + 0.2 * j:.6f} 0.5 {bok:.6f} {bok:.6f}")
            (katalog / "labels" / split / f"{nazwa}.txt").write_text("\n".join(linie))

    (katalog / "data.yaml").write_text(yaml.safe_dump({
        "path": str(katalog), "train": "images/train", "val": "images/val",
        "test": "images/test", "names": {i: n for i, n in enumerate(klasy)},
    }, allow_unicode=True))
    return katalog / "data.yaml"


# ---------------------------------------------------------------- plan


def _plan(dane):
    """Plan z jednym etapem na podanym data.yaml (jak plan_dane_ewis1_bt.json)."""
    return TrainingPlan(etapy=[Stage("etap", str(dane), epoki=1, lr0=0.01)])


def test_plan_zapis_i_odczyt(tmp_path):
    p = TrainingPlan(etapy=[Stage("a", "a.yaml", epoki=10, lr0=0.01),
                            Stage("b", "b.yaml", epoki=5, lr0=0.001)])
    p.batch = 8
    p.zapisz(tmp_path / "plan.json")
    w = TrainingPlan.wczytaj(tmp_path / "plan.json")
    assert w.batch == 8
    assert [e.nazwa for e in w.etapy] == [e.nazwa for e in p.etapy]
    assert all(isinstance(e, Stage) for e in w.etapy)


# ---------------------------------------------------------------- kontrola danych

def test_katalog_etykiet_windows():
    """Sciezka z backslashami - na Windowsie tekstowa zamiana nie dzialala."""
    from pathlib import PureWindowsPath
    p = PureWindowsPath(r"C:\Users\uzytkownik\projekt\zbior\dataset\images\train")
    assert katalog_etykiet(p) == PureWindowsPath(r"C:\Users\uzytkownik\projekt\zbior\dataset\labels\train")


def test_katalog_etykiet_posix():
    from pathlib import PurePosixPath
    p = PurePosixPath("/home/x/images/data/images/val")      # ostatnie "images" sie liczy
    assert katalog_etykiet(p) == PurePosixPath("/home/x/images/data/labels/val")


def test_poprawny_zbior_przechodzi(tmp_path):
    ok, problemy, info = sprawdz_dane(_zbior(tmp_path / "ds"))
    assert ok, problemy
    assert info["klasy"] == ["chwast", "uprawa"]
    assert info["train"]["obrazow"] == 6


def test_brak_pliku(tmp_path):
    ok, problemy, _ = sprawdz_dane(tmp_path / "nie_ma.yaml")
    assert not ok and "brak pliku" in problemy[0]


def test_wykrywa_obraz_bez_etykiet(tmp_path):
    y = _zbior(tmp_path / "ds")
    cv2.imwrite(str(tmp_path / "ds" / "images" / "train" / "sierota.jpg"),
                np.zeros((64, 64, 3), np.uint8))
    ok, problemy, _ = sprawdz_dane(y)
    assert not ok
    assert any("bez pliku etykiet" in p for p in problemy)


def test_wykrywa_przeciek_miedzy_splitami(tmp_path):
    """Najwazniejsza kontrola: ten sam plik w train i test."""
    y = _zbior(tmp_path / "ds")
    zrodlo = tmp_path / "ds" / "images" / "train" / "train_000.jpg"
    (tmp_path / "ds" / "images" / "test" / "train_000.jpg").write_bytes(zrodlo.read_bytes())
    ok, problemy, _ = sprawdz_dane(y)
    assert not ok
    assert any("PRZECIEK" in p for p in problemy)


def test_wykrywa_pusty_katalog(tmp_path):
    y = _zbior(tmp_path / "ds", ile_val=0)
    ok, problemy, _ = sprawdz_dane(y)
    assert not ok
    assert any("nie zawiera zadnych obrazow" in p for p in problemy)


def test_brak_klas_w_yaml(tmp_path):
    y = _zbior(tmp_path / "ds")
    dane = yaml.safe_load(y.read_text())
    del dane["names"]
    y.write_text(yaml.safe_dump(dane))
    ok, problemy, _ = sprawdz_dane(y)
    assert not ok
    assert any("names" in p for p in problemy)


# ---------------------------------------------------------------- etykiety

def test_wczytanie_etykiet_przelicza_na_piksele(tmp_path):
    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    cv2.imwrite(str(ds / "images" / "a.jpg"), np.zeros((1000, 1000, 3), np.uint8))
    (ds / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.1\n")
    e = wczytaj_etykiety(ds / "labels", ds / "images")
    (k, x0, y0, x1, y1), = e["a"]
    assert k == 0
    assert (x0, y0, x1, y1) == pytest.approx((400, 450, 600, 550), abs=1)


# ---------------------------------------------------------------- kalibracja

def test_zbior_kalibracyjny(tmp_path):
    from models.export import przygotuj_kalibracje

    zrodlo = tmp_path / "obrazy"
    zrodlo.mkdir()
    for i in range(50):
        cv2.imwrite(str(zrodlo / f"{i}.jpg"), np.zeros((32, 32, 3), np.uint8))
    kat, lista = przygotuj_kalibracje(zrodlo, tmp_path / "kal", liczba=20)
    assert len(list(kat.glob("*.jpg"))) == 20
    assert len(lista.read_text().strip().split("\n")) == 20


def test_kalibracja_jest_powtarzalna(tmp_path):
    from models.export import przygotuj_kalibracje

    zrodlo = tmp_path / "o"
    zrodlo.mkdir()
    for i in range(30):
        cv2.imwrite(str(zrodlo / f"{i}.jpg"), np.zeros((32, 32, 3), np.uint8))
    a, _ = przygotuj_kalibracje(zrodlo, tmp_path / "k1", liczba=10, seed=7)
    b, _ = przygotuj_kalibracje(zrodlo, tmp_path / "k2", liczba=10, seed=7)
    assert sorted(p.name for p in a.glob("*")) == sorted(p.name for p in b.glob("*"))


def test_kalibracja_bez_obrazow(tmp_path):
    from models.export import przygotuj_kalibracje

    (tmp_path / "puste").mkdir()
    with pytest.raises(FileNotFoundError):
        przygotuj_kalibracje(tmp_path / "puste", tmp_path / "k")


def test_rozmiary_modelu(tmp_path):
    from models.export import rozmiary

    p = tmp_path / "model.pt"
    p.write_bytes(b"x" * 12_800_000)
    r = rozmiary(p)
    assert r["rozmiar_MB"] == pytest.approx(12.8, abs=0.1)
    assert r["szacowany_INT8_MB"] == pytest.approx(3.2, abs=0.1)


# ---------------------------------------------------------------- brak ultralytics

@pytest.mark.skipif(
    __import__("importlib").util.find_spec("ultralytics") is not None,
    reason="ultralytics jest zainstalowany")
def test_czytelny_komunikat_bez_ultralytics(tmp_path):
    """Bez ultralytics ma byc jasna instrukcja, nie ModuleNotFoundError."""
    from models.training import _yolo

    with pytest.raises(ImportError, match="pip install ultralytics"):
        _yolo()


# ---------------------------------------------------------------- CLI

def test_cli_sprawdz_poprawny(tmp_path):
    y = _zbior(tmp_path / "ds")
    r = subprocess.run([sys.executable, "-m", "models.cli", "sprawdz", str(y)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "poprawnie" in r.stdout


def test_cli_sprawdz_zwraca_blad_przy_przecieku(tmp_path):
    y = _zbior(tmp_path / "ds")
    zrodlo = tmp_path / "ds" / "images" / "train" / "train_000.jpg"
    (tmp_path / "ds" / "images" / "test" / "train_000.jpg").write_bytes(zrodlo.read_bytes())
    r = subprocess.run([sys.executable, "-m", "models.cli", "sprawdz", str(y)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1                   # nadaje sie do uzycia w skrypcie
    assert "PRZECIEK" in r.stdout


# ---------------------------------------------------------------- trening (bez prawdziwego ultralytics)

class _FakeYOLO:
    """Udaje ultralytics.YOLO: zapisuje wagi tam, gdzie zrobilyby to nowsze
    wersje (runs/detect/...), a nie tam, gdzie wskazuje parametr project."""
    ostatnie = {}

    def __init__(self, wagi):
        self.wagi = wagi

    def train(self, **kw):
        _FakeYOLO.ostatnie = kw
        katalog = Path(kw["project"]).parent / "runs" / "detect" / kw["name"]
        (katalog / "weights").mkdir(parents=True, exist_ok=True)
        (katalog / "weights" / "best.pt").write_bytes(b"x")
        self.trainer = type("T", (), {"save_dir": katalog})()


def test_wagi_szukane_tam_gdzie_zapisal_je_ultralytics(tmp_path, monkeypatch):
    from models import training as trening
    monkeypatch.setattr(trening, "_yolo", lambda: _FakeYOLO)
    plan = _plan(_zbior(tmp_path / "ds"))
    plan.wyjscie = str(tmp_path / "out")
    best, _ = trening.trenuj_etap(plan.etapy[0], "yolov8n.pt", plan)
    assert best.exists()


def test_lr0_nie_jest_ignorowane(tmp_path, monkeypatch):
    """Bez jawnego optymalizatora ultralytics ignoruje lr0 z planu."""
    from models import training as trening
    monkeypatch.setattr(trening, "_yolo", lambda: _FakeYOLO)
    plan = _plan(_zbior(tmp_path / "ds"))
    plan.wyjscie = str(tmp_path / "out")
    trening.trenuj_etap(plan.etapy[0], "yolov8n.pt", plan)
    assert _FakeYOLO.ostatnie["optimizer"] not in (None, "auto")
    assert _FakeYOLO.ostatnie["lr0"] == plan.etapy[0].lr0


# ---------------------------------------------------------------- miary oceny

def test_czulosc_nie_rosnie_z_progiem():
    """Blad wczesniejszej wersji: czulosc przy progu 0.2 wyszla wyzsza niz przy 0.1."""
    from models.evaluation import pr_przy_progach
    prawda = {"a": [(0, 0, 0, 10, 10), (0, 20, 20, 30, 30), (0, 40, 40, 50, 50)]}
    pred = {"a": [(0, 0.9, 0, 0, 10, 10), (0, 0.3, 20, 20, 30, 30), (0, 0.15, 40, 40, 50, 50),
                  (0, 0.5, 70, 70, 80, 80)]}                       # ostatnia = falszywy alarm
    w = pr_przy_progach(pred, prawda, [0.1, 0.2, 0.4, 0.6])
    czul = [x["czulosc"] for x in w]
    assert czul == sorted(czul, reverse=True)
    assert [x["trafione"] for x in w] == [3, 2, 1, 1]
    assert [x["falszywe"] for x in w] == [1, 1, 1, 0]


def test_obiekt_trafiony_tylko_raz():
    """Dwie predykcje na tym samym obiekcie: jedno trafienie, jeden falszywy."""
    from models.evaluation import pr_przy_progach
    prawda = {"a": [(0, 0, 0, 10, 10)]}
    pred = {"a": [(0, 0.9, 0, 0, 10, 10), (0, 0.8, 1, 1, 10, 10)]}
    (w,) = pr_przy_progach(pred, prawda, [0.5])
    assert (w["trafione"], w["falszywe"], w["pominiete"]) == (1, 1, 0)


def test_pominiete_gdy_brak_predykcji():
    from models.evaluation import pr_przy_progach
    (w,) = pr_przy_progach({}, {"a": [(0, 0, 0, 10, 10)]}, [0.25])
    assert (w["czulosc"], w["pominiete"]) == (0.0, 1)


# ---------------------------------------------------------------- urzadzenie (Windows / Mac)

def test_jawne_urzadzenie_przechodzi_bez_zmian():
    from models.device import wybierz
    assert wybierz("cpu") == "cpu"
    assert wybierz("mps") == "mps"
    assert wybierz("0") == "0"


def test_auto_wybiera_cos_sensownego():
    from models.device import wybierz
    for wartosc in (None, "", "auto"):
        assert wybierz(wartosc) in ("0", "mps", "cpu")


def test_mps_fallback_ustawiony():
    """Bez tego operacje nieobslugiwane przez MPS przerywaja trening na Macu."""
    import os
    import models.device  # noqa: F401
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_wybor_kolejnosc_cuda_mps_cpu(monkeypatch):
    import sys
    import types
    from models import device as urzadzenie

    def atrapa(cuda, mps):
        t = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: cuda),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)))
        monkeypatch.setitem(sys.modules, "torch", t)

    atrapa(True, True)
    assert urzadzenie.wybierz() == "0"          # NVIDIA ma pierwszenstwo
    atrapa(False, True)
    assert urzadzenie.wybierz() == "mps"        # Mac z Apple Silicon
    atrapa(False, False)
    assert urzadzenie.wybierz() == "cpu"


def test_cli_srodowisko():
    r = subprocess.run([sys.executable, "-m", "models.cli", "srodowisko"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "wybrane_urzadzenie" in r.stdout


def test_train_jako_lista_katalogow(tmp_path):
    """data.yaml z kilkoma katalogami treningowymi (np. WeedsGalore + Sudars)."""
    a = _zbior(tmp_path / "a", ile_train=4)
    b = _zbior(tmp_path / "b", ile_train=3)
    dane = yaml.safe_load(a.read_text())
    dane["train"] = ["images/train", str(tmp_path / "b" / "images" / "train")]
    a.write_text(yaml.safe_dump(dane))
    ok, problemy, info = sprawdz_dane(a)
    # nazwy plikow w obu zbiorach sa takie same (train_000...), wiec to nie przeciek,
    # tylko powtorzenie w obrebie train - kontrola miedzy czesciami ma przejsc
    assert ok, problemy
    assert info["train"]["obrazow"] == 7


def test_lista_wykrywa_brakujacy_katalog(tmp_path):
    a = _zbior(tmp_path / "a")
    dane = yaml.safe_load(a.read_text())
    dane["train"] = ["images/train", str(tmp_path / "nie_ma")]
    a.write_text(yaml.safe_dump(dane))
    ok, problemy, _ = sprawdz_dane(a)
    assert not ok and any("nie istnieje" in p for p in problemy)


# ---------------------------------------------------------------- augmentacje i ocena wedlug dni

def test_augmentacje_z_planu_trafiaja_do_treningu(tmp_path, monkeypatch):
    from models import training as trening
    monkeypatch.setattr(trening, "_yolo", lambda: _FakeYOLO)
    y = _zbior(tmp_path / "ds")
    (tmp_path / "plan.json").write_text(json.dumps({"etapy": [
        {"nazwa": "aug", "dane": str(y), "epoki": 1, "lr0": 0.01,
         "augmentacje": {"degrees": 15, "flipud": 0.5}}]}))
    plan = TrainingPlan.wczytaj(tmp_path / "plan.json")
    plan.wyjscie = str(tmp_path / "out")
    trening.trenuj_etap(plan.etapy[0], "yolov8n.pt", plan)
    assert _FakeYOLO.ostatnie["degrees"] == 15
    assert _FakeYOLO.ostatnie["flipud"] == 0.5


def test_bez_augmentacji_nic_nie_dochodzi(tmp_path, monkeypatch):
    from models import training as trening
    monkeypatch.setattr(trening, "_yolo", lambda: _FakeYOLO)
    plan = _plan(_zbior(tmp_path / "ds"))
    plan.wyjscie = str(tmp_path / "out")
    trening.trenuj_etap(plan.etapy[0], "yolov8n.pt", plan)
    assert "degrees" not in _FakeYOLO.ostatnie


def test_ocena_wedlug_dni():
    from models.evaluation import ocen_wedlug_grup
    gt = (0, 0, 0, 10, 10)
    prawda = {"wg__2023-05-25_1": [gt, (0, 20, 20, 30, 30)],
              "wg__2023-06-06_1": [gt]}
    pred = {"wg__2023-05-25_1": [(0, 0.9, 0, 0, 10, 10)],          # 1 z 2
            "wg__2023-06-06_1": [(0, 0.9, 0, 0, 10, 10), (0, 0.9, 50, 50, 60, 60)]}
    w = ocen_wedlug_grup(pred, prawda, prog=0.15)
    assert w["2023-05-25"]["czulosc"] == 0.5
    assert w["2023-06-06"]["czulosc"] == 1.0 and w["2023-06-06"]["falszywe"] == 1
    assert w["razem"]["trafione"] == 2 and w["razem"]["obrazow"] == 2
