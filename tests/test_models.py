"""Testy modulu modele.

Sam trening wymaga pakietu ultralytics i karty graficznej, wiec nie jest tu
uruchamiany. Testowane jest wszystko wokol: plan, kontrola danych przed
treningiem, rozklad rozmiarow obiektow, liczenie IoU, dobor zbioru
kalibracyjnego i komunikaty bledow przy braku ultralytics.
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

from models.evaluation import (MALE, _iou, rozklad_rozmiarow,          # noqa: E402
                          wczytaj_etykiety)
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

def test_domyslny_plan_ma_trzy_etapy():
    p = TrainingPlan.domyslny("a.yaml", "b.yaml", "c.yaml")
    assert [e.nazwa for e in p.etapy] == ["1-naziemne", "2-lotnicze", "3-wlasne"]


def test_lr_maleje_z_etapami():
    """Kazdy kolejny etap uczy sie wolniej - inaczej zapomni poprzedni."""
    p = TrainingPlan.domyslny("a.yaml", "b.yaml", "c.yaml")
    lr = [e.lr0 for e in p.etapy]
    assert lr == sorted(lr, reverse=True)


def test_epoki_maleja_z_etapami():
    p = TrainingPlan.domyslny("a.yaml", "b.yaml", "c.yaml")
    ep = [e.epoki for e in p.etapy]
    assert ep == sorted(ep, reverse=True)


def test_brakujace_etapy_pomijane():
    p = TrainingPlan.domyslny(None, None, "c.yaml")
    assert len(p.etapy) == 1 and p.etapy[0].nazwa == "3-wlasne"


def test_plan_bez_danych_to_blad():
    with pytest.raises(ValueError):
        TrainingPlan.domyslny(None, None, None)


def test_plan_zapis_i_odczyt(tmp_path):
    p = TrainingPlan.domyslny("a.yaml", "b.yaml", "c.yaml")
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
    p = PureWindowsPath(r"C:\Users\kam05\pelny pbl\zbior\dataset\images\train")
    assert katalog_etykiet(p) == PureWindowsPath(r"C:\Users\kam05\pelny pbl\zbior\dataset\labels\train")


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


def test_rozklad_rozmiarow_liczy_kategorie():
    etykiety = {"a": [(0, 0, 0, 20, 20),        # pole 400 -> male
                      (0, 0, 0, 50, 50),        # 2500 -> srednie
                      (1, 0, 0, 200, 200)]}     # 40000 -> duze
    r = rozklad_rozmiarow(etykiety, ["chwast", "uprawa"])
    assert r["kategorie"] == {"male": 1, "srednie": 1, "duze": 1}
    assert r["obiektow"] == 3
    assert r["per_klasa"]["uprawa"] == {"duze": 1}


def test_prog_malego_obiektu_zgodny_z_coco():
    assert MALE == 32 ** 2


def test_rozklad_na_prawdziwym_zbiorze(tmp_path):
    y = _zbior(tmp_path / "ds")
    ds = tmp_path / "ds"
    r = rozklad_rozmiarow(wczytaj_etykiety(ds / "labels" / "train",
                                           ds / "images" / "train"))
    assert r["obiektow"] == 18                 # 6 obrazow x 3 ramki
    assert 0 <= r["udzial_malych"] <= 1


# ---------------------------------------------------------------- IoU

def test_iou_identyczne_ramki():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_rozlaczne():
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_polowa():
    # dwie ramki 10x10 nachodzace na sie w polowie: 50 / 150
    assert _iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


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


def test_wniosek_o_kwantyzacji():
    from models.export import _wniosek

    assert "pomijalna" in _wniosek(0.005)
    assert "akceptowalna" in _wniosek(0.02)
    assert "QAT" in _wniosek(0.08)


# ---------------------------------------------------------------- brak ultralytics

@pytest.mark.skipif(
    __import__("importlib").util.find_spec("ultralytics") is not None,
    reason="ultralytics jest zainstalowany")
def test_czytelny_komunikat_bez_ultralytics(tmp_path):
    """Bez ultralytics ma byc jasna instrukcja, nie ModuleNotFoundError."""
    from models.evaluation import _yolo

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


def test_cli_rozmiary(tmp_path):
    _zbior(tmp_path / "ds")
    r = subprocess.run([sys.executable, "-m", "models.cli", "rozmiary",
                        str(tmp_path / "ds"), "--split", "train"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout[r.stdout.index("{"):r.stdout.rindex("}") + 1])["obiektow"] == 18


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
    plan = TrainingPlan.domyslny(None, None, str(_zbior(tmp_path / "ds")))
    plan.wyjscie = str(tmp_path / "out")
    best, _ = trening.trenuj_etap(plan.etapy[0], "yolov8n.pt", plan)
    assert best.exists()


def test_lr0_nie_jest_ignorowane(tmp_path, monkeypatch):
    """Bez jawnego optymalizatora ultralytics ignoruje lr0 z planu."""
    from models import training as trening
    monkeypatch.setattr(trening, "_yolo", lambda: _FakeYOLO)
    plan = TrainingPlan.domyslny(None, None, str(_zbior(tmp_path / "ds")))
    plan.wyjscie = str(tmp_path / "out")
    trening.trenuj_etap(plan.etapy[0], "yolov8n.pt", plan)
    assert _FakeYOLO.ostatnie["optimizer"] not in (None, "auto")
    assert _FakeYOLO.ostatnie["lr0"] == plan.etapy[0].lr0


# ---------------------------------------------------------------- dobor progu

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


class _FakeBox:
    def __init__(self, k, c, box):
        self.cls, self.conf, self.xyxy = k, c, [box]


class _FakePrediction:
    """Udaje YOLO.predict: zwraca prawdziwe ramki z etykiet z pewnoscia 0.6."""
    def __init__(self, wagi):
        pass

    def predict(self, sciezka, **kw):
        import cv2
        from pathlib import Path as P
        p = P(sciezka)
        h, w = cv2.imread(str(p)).shape[:2]
        txt = p.parent.parent.parent / "labels" / p.parent.name / (p.stem + ".txt")
        boxes = []
        for l in txt.read_text().split("\n"):
            if l.strip():
                k, cx, cy, bw, bh = map(float, l.split())
                boxes.append(_FakeBox(int(k), 0.6, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                                                    (cx + bw / 2) * w, (cy + bh / 2) * h]))
        return [type("R", (), {"boxes": boxes})()]


def test_dobierz_prog_na_atrapie(tmp_path, monkeypatch):
    from models import evaluation as ocena
    monkeypatch.setattr(ocena, "_yolo", lambda: _FakePrediction)
    y = _zbior(tmp_path / "ds")
    w = ocena.dobierz_prog("x.pt", y, split="val", progi=(0.5, 0.7))
    p05, p07 = w["wyniki"]
    assert p05["czulosc"] == 1.0 and p05["precyzja"] == 1.0      # wszystko trafione
    assert p07["czulosc"] == 0.0                                  # wszystko odciete
    assert w["zalecany_prog"] == 0.5 and w["split"] == "val"


def test_cli_prog_domyslnie_na_walidacji():
    """Dobor progu na tescie bylby ukrytym przeciekiem."""
    from models.cli import parser
    a = parser().parse_args(["prog", "m.pt", "d.yaml"])
    assert a.split == "val"


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


def test_etap2_lr_z_ablacji():
    """Ablacja: lr0 0.001 w etapie 2 dalo wyraznie gorszy wynik niz 0.01."""
    p = TrainingPlan.domyslny("a.yaml", "b.yaml", "c.yaml")
    assert p.etapy[1].lr0 == 0.01


def test_cli_ocen_ma_prog():
    from models.cli import parser
    a = parser().parse_args(["ocen", "m.pt", "d.yaml", "--male-obiekty", "--prog", "0.15"])
    assert a.prog == 0.15
    assert parser().parse_args(["ocen", "m.pt", "d.yaml"]).prog == 0.25


def test_train_jako_lista_katalogow(tmp_path):
    """data.yaml z kilkoma katalogami treningowymi (np. WeedsGalore + CoFly)."""
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
    plan = TrainingPlan.domyslny(None, None, str(_zbior(tmp_path / "ds")))
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


def test_cli_ocen_wedlug_dni():
    from models.cli import parser
    a = parser().parse_args(["ocen", "m.pt", "d.yaml", "--wedlug-dni", "--prog", "0.15"])
    assert a.wedlug_dni and a.prog == 0.15


# ---------------------------------------------------------------- podzial calymi dniami

def _zbior_dni(katalog, dni=("2023-05-25", "2023-05-30", "2023-06-06", "2023-06-15"), n=20):
    """Jak zbior_lotnicze: kafelki z kilku dni rozrzucone po train/val/test."""
    czesci = ["train", "val", "test"]
    for c in czesci:
        (katalog / "images" / c).mkdir(parents=True, exist_ok=True)
        (katalog / "labels" / c).mkdir(parents=True, exist_ok=True)
    i = 0
    for d in dni:
        for k in range(n):
            c = czesci[i % 3]
            nazwa = f"weedsgalore__{d}__{d}_{k:04d}"
            cv2.imwrite(str(katalog / "images" / c / f"{nazwa}.jpg"), np.zeros((32, 32, 3), np.uint8))
            (katalog / "labels" / c / f"{nazwa}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            i += 1
    (katalog / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(katalog), "train": "images/train", "val": "images/val",
         "test": "images/test", "names": {0: "chwast"}}))
    return katalog


def test_podzial_dniami_caly_dzien_w_tescie(tmp_path):
    from models.splits import podziel_dniami
    zb = _zbior_dni(tmp_path / "zb")
    info = podziel_dniami(zb, tmp_path / "out", "2023-05-25", udzial_val=0.15)
    ds = tmp_path / "out" / "dataset"
    test = list((ds / "images" / "test").iterdir())
    assert len(test) == 20 and all("2023-05-25" in p.name for p in test)
    for c in ("train", "val"):
        assert not any("2023-05-25" in p.name for p in (ds / "images" / c).iterdir())
    # brak przecieku, wszystko ma etykiety, kontrola danych przechodzi
    ok, problemy, _ = sprawdz_dane(ds / "data.yaml")
    assert ok, problemy
    assert info["obrazow"]["train"] + info["obrazow"]["val"] == 60
    # dokladnie: 0.15 * 80 = 12 kafelkow walidacji, po 4 z kazdego z 3 dni
    assert info["obrazow"]["val"] == 12
    val = list((ds / "images" / "val").iterdir())
    for d in ("2023-05-30", "2023-06-06", "2023-06-15"):
        assert sum(d in p.name for p in val) == 4


def test_podzial_dniami_powtarzalny(tmp_path):
    from models.splits import podziel_dniami
    zb = _zbior_dni(tmp_path / "zb")
    podziel_dniami(zb, tmp_path / "a", "2023-06-06")
    podziel_dniami(zb, tmp_path / "b", "2023-06-06")
    nazwy = lambda k: sorted(p.name for p in (k / "dataset" / "images" / "val").iterdir())
    assert nazwy(tmp_path / "a") == nazwy(tmp_path / "b")


def test_podzial_dniami_z_dodatkowym_zbiorem(tmp_path):
    from models.splits import podziel_dniami
    zb = _zbior_dni(tmp_path / "zb")
    inny = _zbior(tmp_path / "sudars")              # zbior bez dat w nazwach
    podziel_dniami(zb, tmp_path / "out", "2023-05-25", dodatkowe_treningowe=[inny.parent])
    y = tmp_path / "out" / "dataset" / "data_z_dodatkowymi.yaml"
    d = yaml.safe_load(y.read_text())
    assert d["train"][0] == "images/train" and len(d["train"]) == 2
    assert d["val"] == "images/val" and d["test"] == "images/test"   # tylko zbior glowny
    ok, problemy, info = sprawdz_dane(y)
    assert ok, problemy


def test_podzial_dniami_nieznany_dzien(tmp_path):
    from models.splits import podziel_dniami
    zb = _zbior_dni(tmp_path / "zb")
    with pytest.raises(ValueError, match="dostepne"):
        podziel_dniami(zb, tmp_path / "out", "2099-01-01")


# ---------------------------------------------------------------- analiza zbioru

def test_analiza_zbioru(tmp_path):
    from models.analysis import analizuj
    zb = _zbior_dni(tmp_path / "zb", n=6)            # 4 dni x 6 obrazow 32x32, po 1 obiekcie
    r = analizuj(zb / "data.yaml", tmp_path / "raport")
    assert r["razem"]["obrazow"] == 24 and r["razem"]["obiektow"] == 24
    assert set(r["dni"]) == {"2023-05-25", "2023-05-30", "2023-06-06", "2023-06-15"}
    assert r["dni"]["2023-05-25"]["obrazow"] == 6
    # ramka 0.2 x 0.2 obrazu 32 px -> bok 6.4 px, wszystkie obiekty male
    assert r["razem"]["bok_px"]["mediana"] == pytest.approx(6.4, abs=0.01)
    assert r["razem"]["udzial"]["male"] == 1.0
    assert sum(r["czesci"][c]["obrazow"] for c in r["czesci"]) == 24
    assert (tmp_path / "raport" / "raport.json").exists()
    assert "Dni lotu" in (tmp_path / "raport" / "raport.md").read_text()


def test_analiza_wykresy(tmp_path):
    pytest.importorskip("matplotlib")
    from models.analysis import analizuj
    r = analizuj(_zbior_dni(tmp_path / "zb", n=4) / "data.yaml", tmp_path / "raport")
    assert len(r["wykresy"]) == 4
    assert all(Path(p).stat().st_size > 1000 for p in r["wykresy"])


def test_analiza_puste_obrazy_i_lista_katalogow(tmp_path):
    from models.analysis import analizuj
    zb = _zbior_dni(tmp_path / "zb", n=3)
    pusty = next((zb / "labels" / "train").iterdir())
    pusty.write_text("")                              # obraz bez obiektow
    inny = _zbior(tmp_path / "inny", ile_train=2)
    d = yaml.safe_load((zb / "data.yaml").read_text())
    d["train"] = ["images/train", str(inny.parent / "images" / "train")]
    (zb / "data.yaml").write_text(yaml.safe_dump(d))
    r = analizuj(zb / "data.yaml")
    assert r["razem"]["puste"] >= 1
    assert r["czesci"]["train"]["obrazow"] == 4 + 2  # 12 obrazow / 3 czesci + 2 z drugiego zbioru


def test_cli_analiza():
    from models.cli import parser
    a = parser().parse_args(["analiza", "d.yaml", "-o", "wyniki/an"])
    assert a.dane == "d.yaml" and a.output == "wyniki/an"
