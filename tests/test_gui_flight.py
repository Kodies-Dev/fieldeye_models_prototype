"""Zakladka "Lot z detekcja": pelny przebieg na sztucznym polu z atrapa detektora."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("streamlit")


def test_lot_z_detekcja_konczy_sie_mozaika():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=500)
    at.run()
    at.radio(key="lot_zrodlo").set_value("Sztuczne pole")
    at.text_input(key="lot_model").set_value("nie_ma.pt")      # brak modelu -> atrapa
    at.slider(key="lot_odstep").set_value(0.0)
    at.slider(key="lot_kadr").set_value(500)
    at.run()
    at.button(key="lot_start").click().run()
    assert not at.exception, [e.value for e in at.exception]
    w = at.session_state["lot_wynik"]
    assert w["w_mozaice"] == w["kadrow"] > 10
    assert w["mozaika_czysta"].ndim == 3
    # ten sam obiekt z kilku kadrow liczony raz
    assert 0 < w["unikalnych"] < w["wykryc"]
    assert any("atrapy" in x.value for x in at.warning)       # uczciwe ostrzezenie


def _kafelki_dwoch_dni(katalog, n=9):
    """Jak WeedsGalore: dwa dni lotu nad tym samym polem - te same miejsca w obu."""
    import importlib.util
    import cv2
    spec = importlib.util.spec_from_file_location("gp", ROOT / "scripts" / "generate_field.py")
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    katalog.mkdir(parents=True)
    for i in range(n):
        img = gp.generuj(600, 600, "kukurydza", seed=200 + i)[0]
        for dzien in ("2023-05-25", "2023-06-06"):
            cv2.imwrite(str(katalog / f"weedsgalore__{dzien}__{dzien}_{i:04d}.jpg"), img)


def test_kafelki_grupowane_po_dniach(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT))
    from app import kafelki_wedlug_dni, pole_z_kafelkow
    _kafelki_dwoch_dni(tmp_path / "k", n=9)
    dni = kafelki_wedlug_dni(tmp_path / "k")
    assert set(dni) == {"2023-05-25", "2023-06-06"}
    assert all(len(v) == 9 for v in dni.values())
    assert pole_z_kafelkow(dni["2023-05-25"]).shape[:2] == (1800, 1800)    # siatka 3 x 3


def test_lot_nad_jednym_dniem_sklada_cala_mozaike(tmp_path):
    """Gdy w zrodle sa te same miejsca z dwoch dni, mozaika sie nie skladala."""
    from streamlit.testing.v1 import AppTest
    _kafelki_dwoch_dni(tmp_path / "k", n=9)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=500)
    at.run()
    at.text_input(key="lot_katalog").set_value(str(tmp_path / "k"))
    at.text_input(key="lot_model").set_value("nie_ma.pt")
    at.slider(key="lot_odstep").set_value(0.0)
    at.slider(key="lot_kadr").set_value(500)
    at.run()
    assert at.selectbox(key="lot_dzien").value in ("2023-05-25", "2023-06-06")
    at.button(key="lot_start").click().run()
    assert not at.exception, [e.value for e in at.exception]
    w = at.session_state["lot_wynik"]
    assert w["w_mozaice"] == w["kadrow"]


def test_prawda_w_kadrze_trafia_w_obiekty(tmp_path):
    """Biale kwadraty w znanych miejscach zrodla: po przeniesieniu do kadrow
    (obroconych!) srodek kazdej ramki musi wypasc na bialym."""
    import sys
    import cv2
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from app import prawda_w_kadrze
    from stitchlive.flight_sim import SimSettings, rozloz
    rng = np.random.default_rng(1)
    zr = np.full((1400, 2000, 3), 60, np.uint8)
    prawda = []
    for _ in range(60):
        x, y = rng.integers(100, 1880), rng.integers(100, 1280)
        cv2.rectangle(zr, (int(x), int(y)), (int(x) + 20, int(y) + 20), (255, 255, 255), -1)
        prawda.append((x, y, x + 20, y + 20))
    # tekstura do skladania; int16, bo w uint8 255 + szum "przekreca sie" do malych wartosci
    zr = np.clip(zr.astype(np.int16) + rng.integers(0, 30, zr.shape), 0, 255).astype(np.uint8)
    pr = rozloz(zr, tmp_path, SimSettings(kadr=500, overlap=0.7, obrot=8.0, ekspozycja=0, seed=1))
    trafione, wszystkie = 0, 0
    for info in pr["kadry"]:
        img = cv2.imread(str(tmp_path / "images" / info["plik"]), cv2.IMREAD_GRAYSCALE)
        for x0, y0, x1, y1 in prawda_w_kadrze(prawda, info, 500, min_widocznosc=0.95):
            wszystkie += 1
            trafione += img[int((y0 + y1) / 2), int((x0 + x1) / 2)] > 200
    assert wszystkie > 50 and trafione / wszystkie > 0.95


def _zbior_z_etykietami(katalog, n=9):
    """Struktura jak w zbior_lotnicze: images/test + labels/test, dwa dni lotu."""
    import importlib.util
    import cv2
    spec = importlib.util.spec_from_file_location("gp", ROOT / "scripts" / "generate_field.py")
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    (katalog / "images" / "test").mkdir(parents=True)
    (katalog / "labels" / "test").mkdir(parents=True)
    for i in range(n):
        img, et = gp.generuj(600, 600, "kukurydza", seed=300 + i)
        linie = [f"0 {(e['xyxy'][0] + e['xyxy'][2]) / 1200:.6f} {(e['xyxy'][1] + e['xyxy'][3]) / 1200:.6f} "
                 f"{(e['xyxy'][2] - e['xyxy'][0]) / 600:.6f} {(e['xyxy'][3] - e['xyxy'][1]) / 600:.6f}" for e in et]
        for dzien in ("2023-05-25", "2023-06-06"):
            nazwa = f"weedsgalore__{dzien}__{dzien}_{i:04d}"
            cv2.imwrite(str(katalog / "images" / "test" / f"{nazwa}.jpg"), img)
            (katalog / "labels" / "test" / f"{nazwa}.txt").write_text("\n".join(linie))


def test_lot_z_prawdziwymi_ramkami_przelacznikiem_i_gifem(tmp_path):
    from streamlit.testing.v1 import AppTest
    _zbior_z_etykietami(tmp_path / "ds")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=500)
    at.run()
    at.text_input(key="lot_katalog").set_value(str(tmp_path / "ds" / "images" / "test"))
    at.text_input(key="lot_model").set_value("nie_ma.pt")
    at.slider(key="lot_odstep").set_value(0.0)
    at.slider(key="lot_kadr").set_value(500)
    at.run()
    at.button(key="lot_start").click().run()
    assert not at.exception, [e.value for e in at.exception]
    w = at.session_state["lot_wynik"]
    t_, p_, f_ = w["porownanie"]
    assert t_ + p_ > 5                                    # prawdziwe chwasty dotarly na mozaike
    assert w["gif"][:4] == b"GIF8"
    assert "Prawdziwe" in at.radio(key="lot_widok").options
    at.radio(key="lot_widok").set_value("Obie").run()
    assert not at.exception
