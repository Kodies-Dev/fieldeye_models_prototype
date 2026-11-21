"""Symulator drona bez Raspberry Pi: zdjecia + telemetria GeoJSON wrzucane do folderu jak w locie.

Bierze zdjecie pola widzianego z gory, "kladzie" je na mapie (lewy gorny rog w podanym
punkcie GPS, polnoc u gory), przelatuje nad nim symulatorem z stitchlive (trasa
wezykiem, zachodzace kadry, opcjonalnie zaklocenia) i co --interwal sekund zapisuje
do folderu kolejne zdjecie oraz obok jego telemetrie - w TYM SAMYM formacie i z takimi
samymi nazwami jak prawdziwy dron (20261109T142859.998Z.jpg + 20261109T142859.998Z.geojson,
FeatureCollection z image, time_utc, heading_deg, survey.agl_m, survey.gsd_mm_per_px, ...;
opis formatu: mission/spray_map.py). Symulacja jest bez bujania (roll_deg = pitch_deg = 0).

Gdy znane sa prawdziwe chwasty (kafelki EWIS1 z etykietami), zapisuje tez
prawda.geojson - ich prawdziwe wspolrzedne - zeby sprawdzic namiary oprysku:
    python -m mission.spray_map WYNIKI/oprysk.geojson prawda.geojson

  python -m mission.drone_sim lot/ --ewis1 zbior_ewis1_podzial/dataset/images/test
  python -m mission.drone_sim lot/ --pole moje_pole.jpg --gsd-mm 2.5
  python -m mission.drone_sim lot/ --sztuczne                   # bez danych: sztuczne pole
  ... --szum-gps-m 1.5        # blad GPS jak bez RTK (rozne dla kazdego zdjecia)

Pelny test bez drona (dwa terminale):
    python -m mission.drone_sim lot/ --ewis1 ...           # "dron" zapisuje zdjecia i telemetrie
    python -m mission.detect_cli lot/ wyniki/              # detekcja + mapa oprysku na biezaco
    python -m mission.spray_map wyniki/oprysk.geojson lot_prawda.geojson   # ocena namiarow
"""
import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from .spray_map import UkladLokalny


def sztuczne_pole(szer=3000, wys=2000, chwasty=120, seed=0):
    """Gleba z rzedami 'uprawy' i zielonymi plamami 'chwastow' o znanym polozeniu.

    Do sprawdzenia calego lancucha bez danych. Prawdziwy model nie musi uznac tych
    plam za chwasty - do oceny namiarow sluzy tu atrapa (--atrapa w fieldeye-detect).

    Argumenty:
        szer, wys - rozmiar pola w px
        chwasty   - ile plam
        seed      - ziarno losowania

    Zwraca:
        (obraz BGR, ramki plam)

    Uzywana w:
        main() (--sztuczne)
    """
    rng = np.random.default_rng(seed)
    pole = np.empty((wys, szer, 3), np.uint8)
    pole[:] = (55, 85, 120)
    pole = np.clip(pole.astype(int) + rng.normal(0, 8, pole.shape), 0, 255).astype(np.uint8)
    for y in range(150, wys, 300):                                    # rzedy uprawy (brazowo-zolte)
        cv2.line(pole, (0, y), (szer, y), (60, 120, 140), 22)
    ramki = []
    while len(ramki) < chwasty:
        x, y, r = int(rng.integers(40, szer - 40)), int(rng.integers(40, wys - 40)), int(rng.integers(9, 18))
        if all(abs(x - (a + c) / 2) > 40 or abs(y - (b + d) / 2) > 40 for a, b, c, d in ramki):
            cv2.circle(pole, (x, y), r, (40, 175, 45), -1)
            ramki.append((x - r, y - r, x + r, y + r))
    return pole, ramki


def _w_kadrze(px, py, naroza):
    """Czy punkt (px, py) lezy w obrysie kadru (naroza z symulatora)."""
    return cv2.pointPolygonTest(np.array(naroza, np.float32), (float(px), float(py)), False) >= 0


def _zapisz_atomowo(cel, dane_bajty):
    """Zapis przez plik tymczasowy i zmiane nazwy - obserwator nigdy nie zobaczy polowy pliku.

    Argumenty:
        cel        - Path docelowy
        dane_bajty - zawartosc pliku
    """
    tmp = cel.with_name(f".{cel.name}.tmp")
    tmp.write_bytes(dane_bajty)
    tmp.replace(cel)


def main(argv=None):
    """Caly symulowany lot: pole -> kadry (rozloz) -> co --interwal s zdjecie + telemetria do folderu.

    Argumenty:
        argv - lista opcji albo None (sys.argv); opisy w --help

    Zwraca:
        0 = OK, 2 = zle wejscie (nieczytelne zdjecie, pole za male na kadr)

    Uzywana w:
        python -m mission.drone_sim, scripts/fieldeye_detect_main.py (fieldeye-detect dron ...)
    """
    ap = argparse.ArgumentParser(prog="python -m mission.drone_sim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder, do ktorego 'kamera' zapisuje zdjecia i telemetrie (np. lot/)")
    zr = ap.add_mutually_exclusive_group(required=True)
    zr.add_argument("--ewis1", metavar="KATALOG_KAFELKOW", help="kafelki EWIS1 z etykietami -> odtworzone zdjecie")
    zr.add_argument("--pole", metavar="ZDJECIE", help="dowolne zdjecie pola z gory (bez prawdziwych chwastow)")
    zr.add_argument("--sztuczne", action="store_true", help="sztuczne pole z zielonymi plamami (test bez danych)")
    ap.add_argument("--numer", type=int, default=0, help="ktore zdjecie EWIS1 (domyslnie pierwsze)")
    ap.add_argument("--gsd-mm", type=float, default=2.5, help="mm pola na piksel zrodla (EWIS1 po normalizacji: 2.5)")
    ap.add_argument("--lat", type=float, default=50.2945, help="szerokosc geograficzna lewego gornego rogu pola")
    ap.add_argument("--lon", type=float, default=18.6714, help="dlugosc geograficzna lewego gornego rogu pola")
    ap.add_argument("--wysokosc", type=float, default=5.0, help="wysokosc lotu nad polem [m] (informacyjnie)")
    ap.add_argument("--kadr", type=int, default=600, help="bok zdjecia w px zrodla (600 = 1,5 m przy 2,5 mm/px)")
    ap.add_argument("--overlap", type=float, default=0.7)
    ap.add_argument("--sidelap", type=float, default=0.5)
    ap.add_argument("--zaklocenia", action="store_true", help="zaklocenia jak 'realistyczny dron' w symulatorze")
    ap.add_argument("--szum-gps-m", type=float, default=0.0, help="blad GPS (odch. standardowe w metrach)")
    ap.add_argument("--szum-kursu-st", type=float, default=0.0, help="blad kursu w telemetrii [st.]")
    ap.add_argument("--interwal", type=float, default=0.7, help="co ile sekund 'kamera' robi zdjecie")
    ap.add_argument("--prawda", help="gdzie zapisac prawda.geojson (domyslnie obok folderu: FOLDER_prawda.geojson)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    from stitchlive.flight_sim import SimSettings, rozloz

    if a.ewis1:
        # zdjecie EWIS1 odtworzone z kafelkow (_k{x}_{y}) - ten sam kod co zakladka 4 w app.py
        from .sim_flight import pole_ze_zdjecia, zdjecia_z_kafelkow
        zdjecia = zdjecia_z_kafelkow(Path(a.ewis1).iterdir())
        if not zdjecia:
            print(f"brak kafelkow z polozeniem w nazwie (_k{{x}}_{{y}}) w {a.ewis1}", file=sys.stderr)
            return 2
        nazwa = sorted(zdjecia)[a.numer % len(zdjecia)]
        pole, ramki = pole_ze_zdjecia(zdjecia[nazwa], z_etykietami=True)
    elif a.pole:
        pole, ramki, nazwa = cv2.imread(a.pole), [], Path(a.pole).name
        if pole is None:
            print(f"nie da sie wczytac {a.pole}", file=sys.stderr)
            return 2
    else:
        pole, ramki = sztuczne_pole(seed=a.seed)
        nazwa = "sztuczne pole"
    H, W = pole.shape[:2]
    g = a.gsd_mm / 1000.0
    if min(W, H) < a.kadr * 1.2:
        print(f"pole {W}x{H} px jest za male na kadr {a.kadr} px - zmniejsz --kadr", file=sys.stderr)
        return 2

    u = SimSettings(tryb="trasa", kadr=a.kadr, overlap=a.overlap, sidelap=a.sidelap, obrot_linii=True,
                    jitter=0.08 if a.zaklocenia else 0.0, obrot=3.0 if a.zaklocenia else 0.0,
                    skala=0.03 if a.zaklocenia else 0.0, ekspozycja=0.05 if a.zaklocenia else 0.0,
                    szum=2.0 if a.zaklocenia else 0.0, winieta=0.3 if a.zaklocenia else 0.0, seed=a.seed)
    with tempfile.TemporaryDirectory() as tmp:
        prawda_lotu = rozloz(pole, tmp, u)
        kadry = prawda_lotu["kadry"]
        uklad = UkladLokalny(a.lat, a.lon)
        na_geo = lambda x, y: uklad.na_stopnie(x * g, -y * g)        # noqa: E731  (x na wschod, y na poludnie)

        # prawdziwe chwasty widoczne na co najmniej jednym zdjeciu
        widoczne = [r for r in ramki
                    if any(_w_kadrze((r[0] + r[2]) / 2, (r[1] + r[3]) / 2, k["naroza"]) for k in kadry)]
        sciezka_prawdy = Path(a.prawda) if a.prawda else Path(str(Path(a.folder).resolve()) + "_prawda.geojson")
        if ramki:
            cechy = []
            for x0, y0, x1, y1 in widoczne:
                lat, lon = na_geo((x0 + x1) / 2, (y0 + y1) / 2)
                cechy.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lon, 8), round(lat, 8)]},
                              "properties": {"rozmiar_m": round(max(x1 - x0, y1 - y0) * g, 3)}})
            sciezka_prawdy.write_text(json.dumps({"type": "FeatureCollection", "features": cechy}), encoding="utf-8")

        folder = Path(a.folder)
        folder.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(a.seed + 1)
        start = datetime.now(timezone.utc)
        print(f"Symulowany dron: {nazwa} ({W}x{H} px, {W * g:.1f} x {H * g:.1f} m), {len(kadry)} zdjec "
              f"co {a.interwal} s -> {folder}")
        print(f"Pole: lewy gorny rog {a.lat:.6f}, {a.lon:.6f}; blad GPS {a.szum_gps_m} m"
              + (f"; prawdziwe chwasty ({len(widoczne)}): {sciezka_prawdy}" if ramki else ""))
        krok_s = a.interwal if a.interwal > 0 else 0.7        # czas symulowany miedzy zdjeciami
        lot_id = start.strftime("%Y%m%dT%H%M%SZ")
        poprzedni = None
        for i, k in enumerate(kadry):
            cx, cy = k["srodek"]
            e_prawda, n_prawda = cx * g, -cy * g
            e, n = e_prawda + rng.normal(0, a.szum_gps_m), n_prawda + rng.normal(0, a.szum_gps_m)
            lat, lon = uklad.na_stopnie(e, n)
            kurs = (k["kat_st"] + rng.normal(0, a.szum_kursu_st)) % 360
            gsd_mm = a.gsd_mm * k["zasieg_px"] / k["kadr_px"][0]
            chwila = start + timedelta(seconds=i * krok_s)
            nazwa = chwila.strftime("%Y%m%dT%H%M%S.") + f"{chwila.microsecond // 1000:03d}Z"
            predkosc = 0.0 if poprzedni is None else round(
                ((e_prawda - poprzedni[0]) ** 2 + (n_prawda - poprzedni[1]) ** 2) ** 0.5 / krok_s, 2)
            poprzedni = (e_prawda, n_prawda)
            w_px, h_px = k["kadr_px"]
            tele = {"type": "FeatureCollection", "features": [{
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(lon, 8), round(lat, 8), round(240.0 + a.wysokosc, 2)]},
                "properties": {
                    "image": nazwa + ".jpg", "seq": i + 1, "flight_id": lot_id,
                    "time_utc": chwila.strftime("%Y-%m-%dT%H:%M:%S.") + f"{chwila.microsecond // 1000:03d}Z",
                    "time_source": "symulacja", "fix": True, "position_method": "symulacja",
                    "fix_quality": 4 if a.szum_gps_m <= 0.05 else 2, "satellites": 30, "hdop": 0.7,
                    "alt_msl_m": round(240.0 + a.wysokosc, 2), "speed_mps": predkosc,
                    "course_deg": round(kurs, 2), "heading_deg": round(kurs, 2),
                    "roll_deg": 0.0, "pitch_deg": 0.0,                  # symulacja bez bujania
                    "survey": {"agl_m": a.wysokosc, "gsd_mm_per_px": round(gsd_mm, 4),
                               "footprint_across_m": round(w_px * gsd_mm / 1000, 3),
                               "footprint_along_m": round(h_px * gsd_mm / 1000, 3), "overlap": a.overlap},
                    "camera": {"width": w_px, "height": h_px}}}]}
            # jak kamera: najpierw zdjecie, potem telemetria; oba zapisy atomowe
            _zapisz_atomowo(folder / (nazwa + ".jpg"), (Path(tmp) / "images" / k["plik"]).read_bytes())
            _zapisz_atomowo(folder / (nazwa + ".geojson"), json.dumps(tele, indent=1).encode())
            print(f"  [{i + 1}/{len(kadry)}] {nazwa}.jpg  {lat:.7f}, {lon:.7f}  kurs {kurs:5.1f}", flush=True)
            if i < len(kadry) - 1:
                time.sleep(a.interwal)
    print("Koniec lotu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
