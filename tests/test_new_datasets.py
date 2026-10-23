"""Nowe zbiory: EWIS1 (Pascal VOC, dron 5 m) i PhenoBench (maski instancji 16-bit)."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from normalization.adapters import ewis1, phenobench  # noqa: E402
from normalization.build import zbuduj  # noqa: E402
from normalization.taxonomy import Taxonomy  # noqa: E402

XML = """<annotation><filename>{plik}</filename><size><width>{w}</width><height>{h}</height></size>
{ob}</annotation>"""
OB = "<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"


@pytest.fixture
def tax():
    return Taxonomy()


def _ewis1(kat):
    """Uklad jak w EWIS1: bounding_boxes/, images/, masks/, additional_images/."""
    for k in ("bounding_boxes", "images", "masks", "additional_images"):
        (kat / k).mkdir(parents=True)
    rng = np.random.default_rng(0)
    for nazwa in ("Loew_20220505_Maize_006", "Loew_20220512_Maize_010", "img_20210629_A1_1_002",
                  "Aholfing_20220525_Sorghum_008"):
        cv2.imwrite(str(kat / "images" / f"{nazwa}.png"), rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
        ob = OB.format(n="weed", a=10, b=10, c=40, d=40) + OB.format(n="maize", a=100, b=100, c=200, d=220)
        (kat / "bounding_boxes" / f"{nazwa}.xml").write_text(XML.format(plik=f"{nazwa}.png", w=400, h=300, ob=ob))


def test_ewis1_bez_sorgo_i_klasy(tmp_path, tax):
    _ewis1(tmp_path / "e")
    r = ewis1(tmp_path / "e", tax)
    assert len(r) == 2 and all("Maize" in x.obraz for x in r)             # tylko kukurydza
    klasy = {o.klasa for x in r for o in x.obiekty}
    assert klasy == {"chwast_inny", "zea_mays"}
    assert all(x.domena == "uav" and x.agl_m == 5.0 and x.gsd_mm == 1.18 for x in r)
    assert len({x.grupa for x in r}) == 2                                  # kazde zdjecie osobna grupa
    assert len(ewis1(tmp_path / "e", tax, z_sorgo=True)) == 4            # z sorgo i img_2021...
    assert not tax.raport_nieznanych()


def test_sorgo_to_uprawa_a_sorgo_aleppo_to_chwast(tax):
    assert tax.klucz("sorghum") == "sorghum_bicolor"
    assert tax.klucz("johnson grass") == "sorghum_halepense"
    assert tax.klucz("crop") == "uprawa_inna"


def _phenobench(kat):
    """train/ i val/ z images, semantics, plant_instances (16-bit); test/ bez etykiet."""
    for czesc in ("train", "val", "test"):
        for k in (("images",) if czesc == "test" else ("images", "semantics", "plant_instances")):
            (kat / czesc / k).mkdir(parents=True)
    rng = np.random.default_rng(1)
    for czesc, n in (("train", 2), ("val", 1)):
        for i in range(n):
            nazwa = f"05-15_{czesc}{i:04d}.png"
            # rozne zdjecia - identyczne odrzucilaby deduplikacja
            cv2.imwrite(str(kat / czesc / "images" / nazwa), rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
            sem = np.zeros((128, 128), np.uint16)
            ins = np.zeros((128, 128), np.uint16)
            sem[10:30, 10:30], ins[10:30, 10:30] = 1, 300           # burak
            sem[50:60, 50:60], ins[50:60, 50:60] = 2, 301           # chwast
            sem[0:8, 100:120], ins[0:8, 100:120] = 4, 302           # chwast przy krawedzi (partial)
            cv2.imwrite(str(kat / czesc / "semantics" / nazwa), sem)
            cv2.imwrite(str(kat / czesc / "plant_instances" / nazwa), ins)
    cv2.imwrite(str(kat / "test" / "images" / "phenoBench_00000.png"), np.zeros((128, 128, 3), np.uint8))


def test_phenobench_ramki_z_instancji(tmp_path, tax):
    _phenobench(tmp_path / "p")
    r = phenobench(tmp_path / "p", tax)
    assert len(r) == 3                                                     # train 2 + val 1, test pominiety
    ob = sorted(r[0].obiekty, key=lambda o: o.xyxy[0])
    assert [o.klasa for o in ob] == ["beta_vulgaris", "chwast_inny", "chwast_inny"]
    assert ob[0].xyxy == [10, 10, 30, 30]
    assert all(x.domena == "uav" and x.wyczerpujace for x in r)
    assert r[0].grupa.startswith("phenobench_train_05-15")


def test_budowa_obu_zbiorow_w_trybie_chwastow(tmp_path):
    _ewis1(tmp_path / "e")
    _phenobench(tmp_path / "p")
    info = zbuduj({"ewis1": tmp_path / "e", "phenobench": tmp_path / "p"}, tmp_path / "zb",
                  tryb_klas="chwasty", proporcje=(1.0, 0.0, 0.0), skale={"ewis1": 0.5},
                  udzial_pustych=1.0)
    assert not info["nierozpoznane_etykiety"]
    ds = tmp_path / "zb" / "dataset"
    obrazy = list((ds / "images" / "train").glob("*.jpg"))
    assert len(obrazy) == 5                                                # 2 kukurydza + 3 PhenoBench
    wm = [p for p in obrazy if "ewis1" in p.name]
    assert cv2.imread(str(wm[0])).shape == (150, 200, 3)                   # 400x300 * 0,5
    etykiety = (ds / "labels" / "train" / (wm[0].stem + ".txt")).read_text().split()
    assert len(etykiety) == 5                                              # 1 chwast; kukurydza tlem
    assert json.loads((tmp_path / "zb" / "build_info.json").read_text())["parametry"]["skale"] == {"ewis1": 0.5}


def test_skrypt_bez_bledow_skladni():
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / "new_data_experiment.sh")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _ewis1_z_maska(kat, ramki_xml, prawdziwe, w_xml, h_xml):
    """Zdjecie i maska 400x300 (poziomo); XML z podanym rozmiarem i ramkami."""
    for k in ("bounding_boxes", "images", "masks"):
        (kat / k).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    cv2.imwrite(str(kat / "images" / "Loew_20220505_Maize_006.png"),
                rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    maska = np.zeros((300, 400, 3), np.uint8)
    for x0, y0, x1, y1 in prawdziwe:
        maska[y0:y1, x0:x1] = (14, 127, 255)
    cv2.imwrite(str(kat / "masks" / "Loew_20220505_Maize_006.png"), maska)
    ob = "".join(OB.format(n="weed", a=a, b=b, c=c, d=d) for a, b, c, d in ramki_xml)
    (kat / "bounding_boxes" / "Loew_20220505_Maize_006.xml").write_text(
        XML.format(plik="Loew_20220505_Maize_006.png", w=w_xml, h=h_xml, ob=ob))


PRAWDZIWE = [(20, 20, 44, 44), (300, 40, 324, 64), (150, 200, 174, 224), (360, 250, 384, 274),
             (60, 150, 84, 174), (220, 100, 244, 124), (100, 260, 124, 284), (330, 160, 354, 184)]


def test_ewis1_zamienione_wymiary_w_xml(tmp_path, tax):
    """Jak w prawdziwym EWIS1: XML mowi 300x400, zdjecie ma 400x300, ramki w ukladzie zdjecia."""
    _ewis1_z_maska(tmp_path / "e", PRAWDZIWE, PRAWDZIWE, w_xml=300, h_xml=400)
    r = ewis1(tmp_path / "e", tax)[0]
    assert (r.szer, r.wys) == (400, 300)                                   # rozmiar ze zdjecia
    assert sorted(tuple(o.xyxy) for o in r.obiekty) == sorted(tuple(map(float, p)) for p in PRAWDZIWE)


def test_ewis1_ramki_w_ukladzie_obroconym(tmp_path, tax):
    """Ramki zapisane dla zdjecia obroconego o 90 st. - adapter wybiera obrot, ktory trafia w maske."""
    w_xml, h_xml = 300, 400
    xml = [(y0, h_xml - x1, y1, h_xml - x0) for x0, y0, x1, y1 in PRAWDZIWE]   # odwrotnosc obrotu w prawo
    _ewis1_z_maska(tmp_path / "e", xml, PRAWDZIWE, w_xml, h_xml)
    r = ewis1(tmp_path / "e", tax)[0]
    assert sorted(tuple(o.xyxy) for o in r.obiekty) == sorted(tuple(map(float, p)) for p in PRAWDZIWE)


def test_xml_z_inna_nazwa_pliku_znajduje_zdjecie(tmp_path, tax):
    """W czesci XML EWIS1 pole filename nie zgadza sie z plikiem - szukamy po nazwie bez rozszerzenia."""
    kat = tmp_path / "e"
    for k in ("bounding_boxes", "images"):
        (kat / k).mkdir(parents=True)
    cv2.imwrite(str(kat / "images" / "img_20210824_D1_3_021.png"), np.zeros((300, 400, 3), np.uint8))
    (kat / "bounding_boxes" / "img_20210824_D1_3_021.xml").write_text(XML.format(
        plik="img_20210824_D1_3_021.jpg", w=400, h=300, ob=OB.format(n="weed", a=1, b=1, c=30, d=30)))
    assert len(ewis1(kat, tax, z_sorgo=True)) == 1


def test_ewis1_odlozony_dzien_do_testu(tmp_path, tax):
    _ewis1(tmp_path / "e")                       # Loew_20220505 i Loew_20220512
    r = {Path(x.obraz).name: x for x in ewis1(tmp_path / "e", tax, test_dni=["20220512"])}
    assert r["Loew_20220512_Maize_010.png"].split == "test"
    assert r["Loew_20220505_Maize_006.png"].split == "train"
    assert r["Loew_20220505_Maize_006.png"].grupa == "ewis1_Loew_20220505"   # grupa = pole + dzien


def test_budowa_z_odlozonym_dniem_i_ocena_wedlug_dni(tmp_path):
    from models.evaluation import ocen_wedlug_grup
    _ewis1(tmp_path / "e")
    zbuduj({"ewis1": tmp_path / "e"}, tmp_path / "zb", tryb_klas="chwasty", proporcje=(1.0, 0.0, 0.0),
           udzial_pustych=1.0, opcje_adapterow={"ewis1": {"test_dni": ["20220512"]}})
    ds = tmp_path / "zb" / "dataset"
    assert [p.name for p in (ds / "images" / "test").glob("*.jpg")] == ["ewis1__Loew_20220512_Maize_010.jpg"]
    assert len(list((ds / "images" / "train").glob("*.jpg"))) == 1
    # daty RRRRMMDD (EWIS1) i RRRR-MM-DD (WeedsGalore) w jednym formacie
    p = {"ewis1__Loew_20220512_Maize_010": [(0, 0, 0, 10, 10)], "weedsgalore__2023-05-25__x": [(0, 0, 0, 10, 10)]}
    w = ocen_wedlug_grup({k: [(0, 0.9, 0, 0, 10, 10)] for k in p}, p)
    assert {"2022-05-12", "2023-05-25", "razem"} == set(w)


def test_cli_ewis1_test_dni():
    from normalization import build
    zlap = {}

    def falszywe(*a, **k):
        zlap.update(k)
        raise SystemExit(0)
    stare, build.zbuduj = build.zbuduj, falszywe
    try:
        with pytest.raises(SystemExit):
            build.main(["--ewis1", "x", "--ewis1-test-dni", "20220509", "20220512"])
    finally:
        build.zbuduj = stare
    assert zlap["opcje_adapterow"]["ewis1"]["test_dni"] == ["20220509", "20220512"]


def test_skrypt_drugiego_testu_bez_bledow_skladni():
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / "ewis1_test.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_skrypt_kandydatow_bez_bledow_skladni():
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / "final_candidates.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr