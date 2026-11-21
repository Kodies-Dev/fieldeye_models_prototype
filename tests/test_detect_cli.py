"""fieldeye-detect: zdjecia z katalogu -> detektor -> wynik w terminalu (atrapa zamiast modelu)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mission import detect_cli as D  # noqa: E402


def _zdjecie(p, plamy=3):
    img = np.full((600, 800, 3), (60, 100, 140), np.uint8)
    for k in range(plamy):
        cv2.circle(img, (100 + k * 150, 300), 25, (40, 170, 40), -1)
    cv2.imwrite(str(p), img)


def test_kompletny_jpeg_i_ucieta_wersja(tmp_path):
    _zdjecie(tmp_path / "a.jpg")
    d = (tmp_path / "a.jpg").read_bytes()
    (tmp_path / "b.jpg").write_bytes(d[: len(d) // 2])
    assert D.kompletny(tmp_path / "a.jpg") and not D.kompletny(tmp_path / "b.jpg")


def test_gotowy_dopiero_po_ustabilizowaniu(tmp_path):
    p = tmp_path / "a.jpg"
    _zdjecie(p)
    stan = {}
    assert not D.gotowy(p, stan, 1.0, teraz=100.0)        # pierwsze spojrzenie
    assert not D.gotowy(p, stan, 1.0, teraz=100.5)        # stabilny za krotko
    assert D.gotowy(p, stan, 1.0, teraz=101.2)


def test_pomija_ukryte_i_tymczasowe(tmp_path):
    for n in ("IMG_1.jpg", ".ukryty.jpg", "~tmp.jpg", "IMG_2.jpg.part", "notatka.txt"):
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in D.kandydaci(tmp_path, set())] == ["IMG_1.jpg"]


def test_raz_zdjecia_z_ramkami_do_folderu_wynikow(tmp_path, capsys):
    kat = tmp_path / "zdjecia"; kat.mkdir()
    _zdjecie(kat / "IMG_0001.jpg", 2); _zdjecie(kat / "IMG_0002.jpg", 4)
    (kat / "IMG_0003.jpg").write_text("zepsuty")
    wyniki = tmp_path / "nowy" / "wyniki"                          # nie istnieje - ma powstac
    assert D.main([str(kat), str(wyniki), "--raz", "--atrapa"]) == 0
    wyj = capsys.readouterr().out
    assert "IMG_0001.jpg" in wyj and "POMINIETE" in wyj and "Podsumowanie: 2 zdjec" in wyj
    assert sorted(p.name for p in wyniki.iterdir()) == ["IMG_0001.jpg", "IMG_0002.jpg"]   # same zdjecia
    assert cv2.imread(str(wyniki / "IMG_0002.jpg")).shape == (600, 800, 3)


def test_jsonl_na_zyczenie(tmp_path, capsys):
    kat = tmp_path / "z"; kat.mkdir()
    _zdjecie(kat / "a.jpg", 2)
    assert D.main([str(kat), str(tmp_path / "w"), "--raz", "--atrapa", "--jsonl"]) == 0
    rek = [json.loads(l) for l in (tmp_path / "w" / "wyniki.jsonl").read_text().splitlines()]
    assert [r["plik"] for r in rek] == ["a.jpg"] and set(rek[0]) >= {"chwasty", "wykrycia", "szer", "wys"}


def test_json_na_stdout(tmp_path, capsys):
    kat = tmp_path / "z"; kat.mkdir()
    _zdjecie(kat / "a.jpg")
    assert D.main([str(kat), "--raz", "--atrapa", "--json"]) == 0
    linie = capsys.readouterr().out.strip().splitlines()
    assert len(linie) == 1 and json.loads(linie[0])["plik"] == "a.jpg"     # na stdout tylko JSON


def test_bledne_argumenty(tmp_path):
    assert D.main([str(tmp_path / "nie_ma"), "--atrapa", "--raz"]) == 2
    assert D.main([str(tmp_path), "--model", str(tmp_path / "brak.pt")]) == 2
    assert D.main([str(tmp_path), str(tmp_path), "--atrapa"]) == 2           # wyniki = zdjecia
    assert D.main([str(tmp_path), "--atrapa", "--jsonl"]) == 2               # --jsonl bez folderu


def test_fifo_wedlug_czasu_zapisu_a_nie_nazwy(tmp_path):
    import os
    for nazwa, t in (("c.jpg", 100), ("a.jpg", 300), ("b.jpg", 200), ("d.jpg", 200)):
        p = tmp_path / nazwa
        p.write_bytes(b"x")
        os.utime(p, (t, t))
    # najstarszy pierwszy; przy rownym czasie (b, d) decyduje nazwa
    assert [p.name for p in D.kandydaci(tmp_path, set())] == ["c.jpg", "b.jpg", "d.jpg", "a.jpg"]


def test_model_jako_trzeci_argument(tmp_path, capsys):
    kat = tmp_path / "z"; kat.mkdir()
    _zdjecie(kat / "a.jpg")
    zly = tmp_path / "model.txt"; zly.write_text("x")
    assert D.main([str(kat), str(tmp_path / "w"), str(zly)]) == 2                    # zly format
    assert "nieobslugiwany format" in capsys.readouterr().err
    assert D.main([str(kat), str(tmp_path / "w"), str(tmp_path / "brak.onnx")]) == 2  # brak pliku
    assert "brak modelu" in capsys.readouterr().err
    dobry = tmp_path / "m.pt"; dobry.write_text("x")
    assert D.main([str(kat), str(tmp_path / "w"), str(dobry), "--model", str(dobry)]) == 2   # podany dwa razy
