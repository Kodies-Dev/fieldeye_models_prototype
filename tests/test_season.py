"""Sezonowosc: nadprobkowanie wczesnych dni, zbior z wybranych dni, progi wedlug dni."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.plan import sprawdz_dane  # noqa: E402
from models.season import (nadprobkuj, ocen_z_progami_dni, progi_wedlug_dni,  # noqa: E402
                           wybierz_dni)

DNI = ["2023-05-25", "2023-05-30", "2023-06-06"]


@pytest.fixture
def zbior(tmp_path):
    """3 dni x (4 train, 2 val, 2 test), po jednym obiekcie na kafelek."""
    ds = tmp_path / "ds"
    for czesc, n in (("train", 4), ("val", 2), ("test", 2)):
        (ds / "images" / czesc).mkdir(parents=True)
        (ds / "labels" / czesc).mkdir(parents=True)
        for d in DNI:
            for i in range(n):
                nazwa = f"weedsgalore__{d}__{d}_{czesc}{i}"
                cv2.imwrite(str(ds / "images" / czesc / f"{nazwa}.jpg"), np.full((32, 32, 3), 90, np.uint8))
                (ds / "labels" / czesc / f"{nazwa}.txt").write_text("0 0.5 0.5 0.2 0.2")
    (ds / "data.yaml").write_text(yaml.safe_dump({"path": str(ds), "train": "images/train",
                                                  "val": "images/val", "test": "images/test",
                                                  "names": {0: "chwast"}}))
    return ds


def test_nadprobkowanie_liczby_i_pliki(zbior, tmp_path):
    out = zbior / "data_sezon.yaml"
    licz = nadprobkuj(zbior, {"2023-05-25": 3, "2023-05-30": 2}, out)
    assert licz["2023-05-25"] == {"przed": 4, "po": 12}
    assert licz["2023-05-30"] == {"przed": 4, "po": 8}
    assert licz["2023-06-06"] == {"przed": 4, "po": 4}
    kopie = list((zbior / "images" / "train_nadprob").glob("*.jpg"))
    assert len(kopie) == 4 * 2 + 4 * 1                          # 25.05: 2 kopie, 30.05: 1 kopia
    assert all((zbior / "labels" / "train_nadprob" / (k.stem + ".txt")).exists() for k in kopie)
    d = yaml.safe_load(out.read_text())
    assert d["train"] == ["images/train", "images/train_nadprob"] and d["val"] == "images/val"


def test_nadprobkowanie_bez_przecieku_i_z_dodatkowym_zbiorem(zbior, tmp_path):
    inny = tmp_path / "inny"
    (inny / "images" / "train").mkdir(parents=True)
    (inny / "labels" / "train").mkdir(parents=True)
    cv2.imwrite(str(inny / "images" / "train" / "s1.jpg"), np.zeros((32, 32, 3), np.uint8))
    (inny / "labels" / "train" / "s1.txt").write_text("")
    out = zbior / "data_sezon.yaml"
    nadprobkuj(zbior, {"2023-05-25": 2}, out, dodatkowe_train=[inny])
    ok, problemy, info = sprawdz_dane(out)
    assert ok, problemy
    assert info["train"]["obrazow"] == 12 + 4 + 1               # train + kopie + dodatkowy
    assert info["val"]["obrazow"] == 6 and info["test"]["obrazow"] == 6   # walidacja i test bez zmian


def test_ponowne_nadprobkowanie_nie_dokleja_kopii(zbior):
    nadprobkuj(zbior, {"2023-05-25": 3})
    nadprobkuj(zbior, {"2023-05-25": 2})
    assert len(list((zbior / "images" / "train_nadprob").glob("*.jpg"))) == 4


def test_wybierz_dni(zbior, tmp_path):
    ile = wybierz_dni(zbior, ["2023-05-25", "2023-05-30"], tmp_path / "maj")
    assert ile == {"train": 8, "val": 4, "test": 4}
    nowy = tmp_path / "maj" / "dataset" / "data.yaml"
    ok, problemy, _ = sprawdz_dane(nowy)
    assert ok, problemy
    assert not list((tmp_path / "maj" / "dataset" / "images" / "test").glob("*2023-06-06*"))


def _pred_prawda():
    """Dzien 'a': siewki z niska pewnoscia (0,1); dzien 'b': rosliny z wysoka (0,8) i falszywki 0,1."""
    prawda, pred = {}, {}
    for i in range(4):
        prawda[f"x__2023-05-25_{i}"] = [(0, 0, 0, 10, 10)]
        pred[f"x__2023-05-25_{i}"] = [(0, 0.10, 0, 0, 10, 10)]
        prawda[f"x__2023-06-06_{i}"] = [(0, 0, 0, 10, 10)]
        pred[f"x__2023-06-06_{i}"] = [(0, 0.80, 0, 0, 10, 10), (0, 0.10, 50, 50, 60, 60)]
    return pred, prawda


def test_progi_wedlug_dni_nizszy_w_maju():
    pred, prawda = _pred_prawda()
    p = progi_wedlug_dni(pred, prawda)
    assert p["2023-05-25"]["prog"] <= 0.10 and p["2023-05-25"]["czulosc"] == 1.0
    assert p["2023-06-06"]["prog"] > 0.10 and p["2023-06-06"]["precyzja"] == 1.0


def test_ocen_z_progami_dni_lepiej_niz_jeden_prog():
    from models.evaluation import ocen_wedlug_grup
    pred, prawda = _pred_prawda()
    jeden = ocen_wedlug_grup(pred, prawda, 0.15)
    progi = progi_wedlug_dni(pred, prawda)
    dni = ocen_z_progami_dni(pred, prawda, progi)
    assert jeden["2023-05-25"]["czulosc"] == 0.0 and dni["2023-05-25"]["czulosc"] == 1.0
    assert dni["2023-06-06"]["falszywe"] == 0
    r = dni["razem"]
    assert r["trafione"] == 8 and r["pominiete"] == 0 and r["czulosc"] == 1.0
    # dzien bez progu z walidacji dostaje prog domyslny
    assert ocen_z_progami_dni(pred, prawda, {}, 0.15)["2023-05-25"]["prog"] == 0.15


def test_cli_sezon():
    from models.cli import parser
    a = parser().parse_args(["nadprobkuj", "zb", "--waga", "2023-05-25=3", "--waga", "2023-05-30=2",
                             "--dodaj-trening", "sud", "-o", "x.yaml"])
    assert a.waga == ["2023-05-25=3", "2023-05-30=2"] and a.dodaj_trening == ["sud"]
    a = parser().parse_args(["ocen-sezon", "m.pt", "d.yaml", "--progi", "p.json"])
    assert a.wejscie == 1280 and a.prog == 0.15
    assert parser().parse_args(["prog-dni", "m.pt", "d.yaml"]).split == "val"


def test_skrypt_eksperymentu_bez_bledow_skladni():
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / "season_experiments.sh")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr