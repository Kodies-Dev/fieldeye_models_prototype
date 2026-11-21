"""Testy sezonowosci: nadprobkowanie wczesnych dni lotu.

Po co: male chwasty z poczatku sezonu (25.05, 30.05) sa najtrudniejsze, wiec w treningu
modelu koncowego ich kafelki sa powielane (krok 2 w final_model.sh: 25.05 x3, 30.05 x2).
Testy pilnuja, ze kopie trafiaja tylko do treningu (bez przecieku do walidacji i testu)
i ze ponowne uruchomienie nie dokleja kolejnych kopii.

Testowane: models/season.py (nadprobkuj), models/plan.py (sprawdz_dane), parser models/cli.py.

    python -m pytest tests/test_season.py -q
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.plan import sprawdz_dane  # noqa: E402
from models.season import nadprobkuj  # noqa: E402

DNI = ["2023-05-25", "2023-05-30", "2023-06-06"]


@pytest.fixture
def zbior(tmp_path):
    """3 dni x (4 train, 2 val, 2 test), po jednym obiekcie na kafelek.

    Po co: maly zbior YOLO w ukladzie jak zbior_lotnicze/dataset (nazwy z data w srodku).

    Argumenty:
        tmp_path - fixture pytest: pusty folder tymczasowy

    Zwraca:
        Path - folder zbioru z data.yaml, np. tmp_path / "ds"

    Uzywana w:
        testach nadprobkowania
    """
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
    """Wagi 25.05=3 i 30.05=2: liczby kafelkow przed/po, kopie z etykietami, nowy data.yaml.

    Po co: dokladnie te wagi sa w treningu modelu koncowego.

    Argumenty:
        zbior    - fixture: maly zbior YOLO
        tmp_path - fixture pytest: folder tymczasowy

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
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
    """Dodatkowy zbior treningowy (jak Sudars / EWIS1) trafia tylko do treningu.

    Po co: walidacja i test musza zostac bez zmian - inaczej wyniki bylyby zawyzone.

    Argumenty:
        zbior    - fixture: maly zbior YOLO
        tmp_path - fixture pytest: folder na dodatkowy zbior

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
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
    """Drugie nadprobkowanie zastepuje kopie z pierwszego, zamiast dokladac nowe.

    Po co: final_model.sh mozna uruchomic ponownie - kopie nie moga sie mnozyc.

    Argumenty:
        zbior - fixture: maly zbior YOLO

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    nadprobkuj(zbior, {"2023-05-25": 3})
    nadprobkuj(zbior, {"2023-05-25": 2})
    assert len(list((zbior / "images" / "train_nadprob").glob("*.jpg"))) == 4


def test_cli_sezon():
    """Opcje komendy nadprobkuj (--waga, --dodaj-trening).

    Po co: final_model.sh wola nadprobkuj z tymi opcjami - zmiana ich nazw zepsulaby trening.

    Argumenty:
        brak

    Zwraca:
        nic - pytest sprawdza asserty

    Uzywana w:
        pytest (python -m pytest tests)
    """
    from models.cli import parser
    a = parser().parse_args(["nadprobkuj", "zb", "--waga", "2023-05-25=3", "--waga", "2023-05-30=2",
                             "--dodaj-trening", "sud", "-o", "x.yaml"])
    assert a.waga == ["2023-05-25=3", "2023-05-30=2"] and a.dodaj_trening == ["sud"]
