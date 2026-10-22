"""Hailo (.hef) bez sprzetu: przygotowanie obrazu, odczyt wyniku NMS, wybor
detektora, komunikat bez HailoRT oraz ocena dowolnego detektora (AP50)."""
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
    """Odwrotnosc nms_hailo_na_detekcje: ramki w pikselach oryginalu -> format Hailo."""
    s = min(wej / w0, wej / h0)
    px, py = (wej - round(w0 * s)) // 2, (wej - round(h0 * s)) // 2
    wiersze = [[(y0 * s + py) / wej, (x0 * s + px) / wej, (y1 * s + py) / wej, (x1 * s + px) / wej, c]
               for x0, y0, x1, y1, c in ramki_px]
    return [np.array(wiersze, np.float32)], s, px, py


@pytest.mark.parametrize("w0,h0,wej", [(600, 600, 640), (600, 600, 1280), (4056, 3040, 640), (700, 500, 1280)])
def test_ramki_wracaja_na_swoje_miejsce(w0, h0, wej):
    """Ramka przeliczona do formatu Hailo i z powrotem trafia w to samo miejsce."""
    ramki = [(100, 200, 160, 260, 0.9), (10, 20, 30, 45, 0.3)]
    wyjscie, s, px, py = _hailo_z_ramek(ramki, w0, h0, wej)
    det = nms_hailo_na_detekcje(wyjscie, wej, wej, s, px, py, w0, h0)
    assert len(det) == 2
    for d, (x0, y0, x1, y1, c) in zip(det, ramki):
        assert abs(d.x0 - x0) < 0.6 and abs(d.y0 - y0) < 0.6
        assert abs(d.x1 - x1) < 0.6 and abs(d.y1 - y1) < 0.6
        assert d.klasa == "chwast" and abs(d.pewnosc - c) < 1e-6


def test_kolejnosc_y_przed_x():
    """Hailo zapisuje [y0, x0, y1, x1] - zamiana kolejnosci dalaby ramke obrocona."""
    det = nms_hailo_na_detekcje([np.array([[0.1, 0.5, 0.2, 0.9, 0.8]], np.float32)],
                                100, 100, 1.0, 0, 0, 100, 100)
    d = det[0]
    assert (d.x0, d.y0, d.x1, d.y1) == (50, 10, 90, 20)


def test_prog_i_puste_klasy():
    wyjscie = [np.array([[0.1, 0.1, 0.2, 0.2, 0.10], [0.3, 0.3, 0.4, 0.4, 0.50]], np.float32),
               np.zeros((0, 5), np.float32)]
    det = nms_hailo_na_detekcje(wyjscie, 640, 640, 1.0, 0, 0, 640, 640, prog=0.15)
    assert len(det) == 1 and det[0].pewnosc == pytest.approx(0.5)


def test_ramka_w_paddingu_odpada():
    """Wykrycie w szarym marginesie letterboxa nie istnieje na prawdziwym zdjeciu."""
    # obraz 640x320 w wejsciu 640: margines 160 px na gorze i na dole
    det = nms_hailo_na_detekcje([np.array([[0.0, 0.1, 0.2, 0.2, 0.9]], np.float32)],
                                640, 640, 1.0, 0, 160, 640, 320)
    assert det == []


def test_letterbox_rozmiar_kolory_margines():
    img = np.zeros((300, 600, 3), np.uint8)
    img[:, :, 0] = 255                                   # niebieski w BGR
    we, s, px, py = letterbox(img, 640, 640)
    assert we.shape == (640, 640, 3) and we.dtype == np.uint8
    assert s == pytest.approx(640 / 600) and px == 0 and py == 160
    assert tuple(we[320, 320]) == (0, 0, 255)            # RGB: niebieski na koncu
    assert tuple(we[10, 320]) == (114, 114, 114)         # margines


def test_detektor_hailo_bez_hailort(monkeypatch, tmp_path):
    import builtins
    from mission.detector import HailoDetector
    prawdziwy = builtins.__import__

    def bez_hailo(name, *a, **k):
        if name == "hailo_platform":
            raise ImportError("brak")
        return prawdziwy(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", bez_hailo)
    with pytest.raises(ImportError, match="HailoRT"):
        HailoDetector(tmp_path / "m.hef")


def test_zbuduj_detektor_wybiera_hailo_dla_hef(monkeypatch):
    from mission import detector as dm

    class _FakeHailo:
        def __init__(self, model, prog):
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
    from models.evaluation import ap50
    prawda = {"a": [(0, 0, 0, 10, 10), (0, 20, 20, 30, 30)]}
    assert ap50({"a": [(0, 0.9, 0, 0, 10, 10), (0, 0.8, 20, 20, 30, 30)]}, prawda) == pytest.approx(1.0, abs=0.01)
    assert ap50({"a": []}, prawda) == 0.0


def test_ap50_polowa_znaleziona():
    from models.evaluation import ap50
    prawda = {"a": [(0, 0, 0, 10, 10), (0, 20, 20, 30, 30)]}
    # jeden trafiony, jeden fałszywy z niższą pewnością. Tak jak w ultralytics krzywa
    # kończy się punktem (czułość 1, precyzja 0), połączonym linią z ostatnim punktem
    # modelu: pole = 0,5 · 1 (czułość 0–0,5) + trójkąt 0,5 · 0,5 / 2 = 0,625
    assert ap50({"a": [(0, 0.9, 0, 0, 10, 10), (0, 0.5, 50, 50, 60, 60)]}, prawda) == pytest.approx(0.625, abs=0.01)


class _GroundTruthDetector:
    """Detektor, który 'widzi' dokładnie ramki z etykiet - do sprawdzenia całej oceny."""
    nazwa = "prawda"

    def __init__(self, etykiety):
        self.etykiety = etykiety

    def wykryj(self, obraz):
        h, w = obraz.shape[:2]
        klucz = int(obraz[0, 0, 0])                      # numer obrazu zapisany w pikselu
        out = []
        for cx, cy, bw, bh in self.etykiety[klucz]:
            out.append(Detection((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h,
                                "chwast", 0.9))
        return out


def test_ocen_detektor_cala_sciezka(tmp_path):
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
    from models.cli import parser
    a = parser().parse_args(["ocen-detektor", "m.hef", "d.yaml", "--prog", "0.15"])
    assert a.model == ["m.hef"] and a.split == "test" and a.wejscie == 1280
    a = parser().parse_args(["kalibracja", "obrazy", "--rozmiar", "640"])
    assert a.rozmiar == 640


def test_kalibracja_w_rozmiarze_wejscia(tmp_path):
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
    from mission.config import MissionConfig
    k = MissionConfig()
    assert k.detekcja.prog_pewnosci == 0.15 and k.detekcja.wejscie_sieci == 1280
    pt = MissionConfig.wczytaj(ROOT / "configs" / "mission_wg_sudars.yaml")
    assert pt.detekcja.model.endswith(".pt") and pt.detekcja.wejscie_sieci == 1280
    hef = MissionConfig.wczytaj(ROOT / "configs" / "mission_hailo.yaml")
    assert hef.detekcja.model.endswith(".hef") and hef.detekcja.prog_pewnosci == 0.15


def test_konfiguracja_hailo_wskazuje_model_koncowy():
    """Model z konfiguracji jest w model_fin.zip, a skrypt sprawdzajacy uzywa tej samej sciezki."""
    import zipfile
    from mission.config import MissionConfig
    model = MissionConfig.wczytaj(ROOT / "configs" / "mission_hailo.yaml").detekcja.model
    with zipfile.ZipFile(ROOT / "model_fin.zip") as z:
        assert f"model_fin/{Path(model).name}" in z.namelist()
    assert model in (ROOT / "scripts" / "check_hef.sh").read_text(encoding="utf-8")


def test_kafelek_hailo_obejmuje_pole_kafelka_treningowego():
    """Kafelek w konfiguracji to ok. 1,5 m pola dla Arducam 64MP (16 MP) z 5 m, jak przy treningu."""
    from mission.config import MissionConfig
    from stitchlive.camera import ARDUCAM_64MP_16MP as kam, KAFELEK_TRENINGOWY_M
    kafelek = MissionConfig.wczytaj(ROOT / "configs" / "mission_hailo.yaml").detekcja.kafelek
    pole_m = kafelek * kam.gsd_mm(5.0) / 1000
    assert abs(pole_m - KAFELEK_TRENINGOWY_M) < 0.05 * KAFELEK_TRENINGOWY_M


@pytest.mark.parametrize("skrypt", ["finish_models.sh", "check_hef.sh"])
def test_skrypty_bez_bledow_skladni(skrypt):
    import subprocess
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / skrypt)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr