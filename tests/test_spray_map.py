"""Telemetria GeoJSON -> wspolrzedne chwastow -> mapa oprysku; symulator drona."""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mission import spray_map as S  # noqa: E402


def test_uklad_lokalny_skala_i_powrot():
    u = S.UkladLokalny(50.0, 18.0)
    lat, lon = u.na_stopnie(100.0, 100.0)                       # 100 m na wschod i polnoc
    assert abs(lat - 50.0 - 100 / 111229) < 2e-6                # ok. 111,2 km na stopien szerokosci
    e, n = u.na_metry(lat, lon)
    assert abs(e - 100) < 1e-6 and abs(n - 100) < 1e-6


def test_piksel_na_metry_kierunki():
    e, n = S.piksel_na_metry(300, 0, 600, 600, 0.01, 0)         # gora zdjecia przy kursie 0 = polnoc
    assert e == pytest.approx(0.0, abs=1e-9) and n == pytest.approx(3.0)
    e, n = S.piksel_na_metry(300, 0, 600, 600, 0.01, 90)        # gora zdjecia przy kursie 90 = wschod
    assert e == pytest.approx(3.0) and n == pytest.approx(0.0, abs=1e-9)


def _tele(**wl):
    """Minimalna telemetria w formacie drona."""
    wl.setdefault("survey", {"agl_m": 5.0})
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [18.5, 50.1, 240.0]}, "properties": wl}]}


def test_czytaj_telemetrie_format_drona(tmp_path):
    f = tmp_path / "20261109T142859.998Z.geojson"
    f.write_text(json.dumps(_tele(image="20261109T142859.998Z.jpg", heading_deg=90, time_utc="t")))
    t = S.czytaj_telemetrie(f)
    assert (t["lat"], t["lon"], t["agl"], t["kurs"], t["czas"]) == (50.1, 18.5, 5.0, 90.0, "t")
    d = _tele(image="a.jpg", heading_deg=0)
    d["features"].append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [19, 51]},
                          "properties": {"image": "b.jpg", "heading_deg": 0, "survey": {"agl_m": 5}}})
    (tmp_path / "lot.geojson").write_text(json.dumps(d))
    assert S.czytaj_telemetrie(tmp_path / "lot.geojson", "b.jpg")["lat"] == 51     # wybor po image
    (tmp_path / "zla.geojson").write_text("{")
    assert S.czytaj_telemetrie(tmp_path / "zla.geojson") is None
    (tmp_path / "feature.geojson").write_text(json.dumps(_tele()["features"][0]))   # sam Feature - nie nasz format
    assert S.czytaj_telemetrie(tmp_path / "feature.geojson") is None
    (tmp_path / "20261109T142859.998Z.jpg").write_bytes(b"x")
    assert S.plik_telemetrii(tmp_path / "20261109T142859.998Z.jpg") == f           # nazwa z kropka w srodku


def test_mapa_laczy_ten_sam_chwast_z_dwoch_zdjec(tmp_path):
    m = S.MapaOprysku(promien_scalania_m=0.10)
    tele = {"lat": 50.0, "lon": 18.0, "kurs": 0.0, "czas": None}
    m.dodaj_zdjecie(tele, "a.jpg", 600, 600, [(290, 290, 310, 310, 0.9)], 0.0025)
    # to samo miejsce widziane z drona przesunietego o 0,5 m na polnoc: chwast 200 px nizej na zdjeciu
    tele2 = dict(tele, lat=50.0 + 0.5 / 111229)
    assert m.dodaj_zdjecie(tele2, "b.jpg", 600, 600, [(290, 490, 310, 510, 0.7)], 0.0025) == 0
    assert len(m.punkty) == 1 and m.punkty_oprysku()[0][6] == 2       # jeden punkt, widziany na 2 zdjeciach
    m.zapisz(tmp_path)
    assert json.loads((tmp_path / "oprysk.geojson").read_text())["features"][0]["properties"]["liczba_zdjec"] == 2
    assert len(json.loads((tmp_path / "chwasty.geojson").read_text())["features"]) == 2
    assert (tmp_path / "oprysk.csv").read_text().splitlines()[0].startswith("id,szerokosc,dlugosc")


def test_lancuch_dron_detekcja_oprysk(tmp_path):
    """Symulowany dron -> fieldeye-detect (atrapa) -> oprysk.geojson trafia w prawdziwe plamy."""
    lot, wyn = tmp_path / "lot", tmp_path / "wyn"
    py = [sys.executable, "-m"]
    subprocess.run(py + ["mission.drone_sim", str(lot), "--sztuczne", "--interwal", "0", "--zaklocenia"],
                   cwd=ROOT, check=True, capture_output=True)
    pierwszy = sorted(lot.glob("*.geojson"))[0]
    assert pierwszy.with_suffix(".jpg").exists()                     # para: zdjecie + telemetria, ta sama nazwa
    assert json.loads(pierwszy.read_text())["type"] == "FeatureCollection"
    r = subprocess.run(py + ["mission.detect_cli", str(lot), str(wyn), "--raz", "--atrapa"],
                       cwd=ROOT, check=True, capture_output=True, text=True)
    assert "GPS" in r.stdout and "Punkty oprysku" in r.stdout
    for n in ("oprysk.geojson", "oprysk.csv", "chwasty.geojson", "trasa.geojson"):
        assert (wyn / n).exists()
    w = S.porownaj(wyn / "oprysk.geojson", Path(str(lot) + "_prawda.geojson"), 0.10)
    assert w["czulosc"] >= 0.95 and w["mediana_cm"] < 1.0       # namiary dokladne co do centymetra


def test_zdjecie_bez_telemetrii_idzie_bez_gps(tmp_path, capsys):
    from mission import detect_cli as D
    import cv2
    import numpy as np
    kat = tmp_path / "z"; kat.mkdir()
    cv2.imwrite(str(kat / "a.jpg"), np.full((300, 400, 3), 90, np.uint8))
    assert D.main([str(kat), "--atrapa", "--stabilnosc", "0", "--czekaj-telemetrie", "0.2",
                   "--interwal", "0.1", "--timeout", "0.6"]) == 0
    assert "bez GPS" in capsys.readouterr().out


def test_kontrola_sensownosci_telemetrii():
    ok = {"lat": 50.3, "lon": 18.7, "agl": 5.0, "kurs": 90.0, "gsd_mm": 1.57}
    assert S.blad_telemetrii(ok) is None
    assert "zamieniona" in S.blad_telemetrii(dict(ok, lat=200.0))
    assert "brak pozycji" in S.blad_telemetrii(dict(ok, lat=0.0, lon=0.0))
    assert "npm" in S.blad_telemetrii(dict(ok, agl=262.0))                  # wysokosc nad morzem zamiast nad polem
    assert "jednostki" in S.blad_telemetrii(dict(ok, gsd_mm=0.00157))       # metry zamiast milimetrow


def test_przechyl_z_telemetrii(tmp_path):
    f = tmp_path / "a.geojson"
    f.write_text(json.dumps(_tele(heading_deg=0, roll_deg=-2.5, pitch_deg=1.0)))
    t = S.czytaj_telemetrie(f)
    assert (t["przechylenie"], t["pochylenie"], S.przechyl_st(t)) == (-2.5, 1.0, 2.5)
    assert S.przechyl_st({"lat": 1, "lon": 1}) is None


def _lot(tmp_path):
    lot = tmp_path / "lot"
    subprocess.run([sys.executable, "-m", "mission.drone_sim", str(lot), "--sztuczne", "--interwal", "0"],
                   cwd=ROOT, check=True, capture_output=True)
    return lot, sorted(lot.glob("*.geojson"))


def test_przechylone_zdjecia_poza_mapa_a_male_z_niepewnoscia(tmp_path):
    """Lot bez bujania, potem recznie: drugie zdjecie przechylone 8 st., trzecie 2 st."""
    lot, pliki = _lot(tmp_path)
    t1 = json.loads(pliki[0].read_text())["features"][0]["properties"]
    assert t1["roll_deg"] == 0.0 and t1["pitch_deg"] == 0.0
    for plik, kat in ((pliki[1], 8.0), (pliki[2], 2.0)):
        t = json.loads(plik.read_text())
        t["features"][0]["properties"]["roll_deg"] = kat
        plik.write_text(json.dumps(t))
    wyn = tmp_path / "wyn"
    r = subprocess.run([sys.executable, "-m", "mission.detect_cli", str(lot), str(wyn), "--raz", "--atrapa"],
                       cwd=ROOT, check=True, capture_output=True, text=True)
    linie = {l.split()[1]: l for l in r.stdout.splitlines() if l.startswith("[")}
    z2, z3 = pliki[1].stem + ".jpg", pliki[2].stem + ".jpg"
    assert "PRZECHYL 8.0" in linie[z2] and "GPS" in linie[z3]
    assert "przechylonych ponad 5 st.: 1" in r.stdout
    cechy = json.loads((wyn / "chwasty.geojson").read_text())["features"]
    assert z2 not in {c["properties"]["zdjecie"] for c in cechy}            # przechylone - poza mapa
    niep = {c["properties"]["niepewnosc_przechylu_m"] for c in cechy if c["properties"]["zdjecie"] == z3}
    assert niep == {round(5.0 * math.tan(math.radians(2.0)), 3)}            # ok. 17 cm przy 2 st. z 5 m


def test_zla_telemetria_idzie_bez_gps(tmp_path):
    lot, pliki = _lot(tmp_path)
    t = json.loads(pliki[3].read_text())
    t["features"][0]["geometry"]["coordinates"][1] += 100                   # szerokosc ok. 150 st. - niemozliwa
    pliki[3].write_text(json.dumps(t))
    r = subprocess.run([sys.executable, "-m", "mission.detect_cli", str(lot), str(tmp_path / "wyn"), "--raz",
                        "--atrapa"], cwd=ROOT, check=True, capture_output=True, text=True)
    assert f"{pliki[3].name}: wspolrzedne poza zakresem" in r.stdout and "zdjec bez GPS: 1" in r.stdout


DRON = ROOT / "tests" / "dane" / "telemetria_dron.geojson"


def test_format_telemetrii_z_drona():
    """Prawdziwy format z drona: FeatureCollection, pola w survey/camera, image, time_utc."""
    t = S.czytaj_telemetrie(DRON, "20261109T142859.998Z.jpg")
    assert (t["lat"], t["lon"], t["agl"], t["gsd_mm"]) == (50.29004902, 18.73101322, 2.0, 0.623)
    assert t["fix"] is True and t["fix_quality"] == 2 and t["czas"] == "2026-11-09T14:28:59.998Z"
    assert t["kurs"] is None                                 # zawis 0,02 m/s: course_deg to szum
    assert "brak kursu" in S.blad_telemetrii(t)


def test_kurs_heading_przed_course_a_course_tylko_w_locie(tmp_path):
    d = json.loads(DRON.read_text())
    wl = d["features"][0]["properties"]
    wl["speed_mps"] = 3.0                                    # w locie: course_deg wystarczy
    (tmp_path / "a.geojson").write_text(json.dumps(d))
    t = S.czytaj_telemetrie(tmp_path / "a.geojson")
    assert (t["kurs"], t["zrodlo_kursu"]) == (7.9, "course")
    wl["heading_deg"] = 92.5                                 # heading ma pierwszenstwo
    (tmp_path / "b.geojson").write_text(json.dumps(d))
    t = S.czytaj_telemetrie(tmp_path / "b.geojson")
    assert (t["kurs"], t["zrodlo_kursu"]) == (92.5, "heading") and S.blad_telemetrii(t) is None


def test_brak_fixa_gps_odrzucony(tmp_path):
    d = json.loads(DRON.read_text())
    d["features"][0]["properties"].update(fix=False, heading_deg=0)
    (tmp_path / "a.geojson").write_text(json.dumps(d))
    assert "brak fixa" in S.blad_telemetrii(S.czytaj_telemetrie(tmp_path / "a.geojson"))


def test_zdjecie_z_drona_na_mapie_z_kursem_domyslnym(tmp_path):
    import cv2
    import numpy as np
    kat = tmp_path / "lot"; kat.mkdir()
    img = np.full((3472, 4624, 3), (60, 100, 140), np.uint8)
    cv2.circle(img, (2312, 1736), 30, (40, 170, 40), -1)     # plama na srodku zdjecia
    cv2.imwrite(str(kat / "20261109T142859.998Z.jpg"), img)
    (kat / "20261109T142859.998Z.geojson").write_text(DRON.read_text())
    r = subprocess.run([sys.executable, "-m", "mission.detect_cli", str(kat), str(tmp_path / "w"), "--raz",
                        "--atrapa", "--kurs-domyslny", "0"], cwd=ROOT, capture_output=True, text=True)
    assert "GPS 50.290049, 18.731013" in r.stdout, r.stdout + r.stderr
    p = json.loads((tmp_path / "w" / "oprysk.geojson").read_text())["features"][0]["geometry"]["coordinates"]
    assert abs(p[0] - 18.73101322) < 1e-6 and abs(p[1] - 50.29004902) < 1e-6   # srodek zdjecia = pod dronem
