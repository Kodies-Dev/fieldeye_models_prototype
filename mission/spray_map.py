"""Telemetria drona w GeoJSON -> wspolrzedne wykrytych chwastow -> mapa oprysku dla rolnika.

Format telemetrii (jeden, ten sam w locie i w symulacji): obok kazdego zdjecia plik o tej
samej nazwie z rozszerzeniem .geojson (20261109T142859.998Z.jpg -> 20261109T142859.998Z.geojson),
zapisywany atomowo, z jednym punktem - pozycja drona w chwili zdjecia:

    {"type": "FeatureCollection",
     "features": [{"type": "Feature",
       "geometry": {"type": "Point", "coordinates": [18.73101322, 50.29004902, 239.73]},   # lon, lat, alt
       "properties": {
         "image": "20261109T142859.998Z.jpg", "seq": 1, "flight_id": "20261109T142856Z",
         "time_utc": "2026-11-09T14:28:59.998Z", "fix": true, "fix_quality": 2,
         "speed_mps": 0.02, "course_deg": 7.9,
         "heading_deg": 7.5, "roll_deg": 0.4, "pitch_deg": -1.2,           # z autopilota (IMU/kompas)
         "survey": {"agl_m": 2.0, "gsd_mm_per_px": 0.623, ...},
         "camera": {"width": 4624, "height": 3472, ...}}}]}

Uzywane pola:
    coordinates [lon, lat]   pozycja drona = srodek zdjecia na polu            (wymagane)
    survey.agl_m             wysokosc nad polem [m] -> skala z modelu kamery   (wymagane albo gsd)
    survey.gsd_mm_per_px     skala wprost [mm/px]; ma pierwszenstwo            (opcjonalne)
    heading_deg              kurs drona od polnocy, zgodnie z zegarem          (wymagane*)
    roll_deg, pitch_deg      przechyl: ponad limit zdjecie poza mapa           (zalecane)
    fix, fix_quality         false / 0 -> zdjecie bez GPS                      (zalecane)
    time_utc, image          czas zapisywany na mapie; dopasowanie punktu do zdjecia
  * gdy brak heading_deg, kurs = course_deg, ale tylko w locie (speed_mps >= 1 m/s);
    w zawisie kierunek ruchu z GPS to szum. Inne pola sa pomijane.

Obliczenia zakladaja kamere skierowana pionowo w dol (BEZ poprawki na bujanie); roll/pitch
sluzy do kontroli: zdjecia przechylone ponad limit nie trafiaja na mape, a przy mniejszym
przechyle punkt dostaje szacowana niepewnosc polozenia (wysokosc * tg(przechyl)).

Wynik dla rolnika (w folderze wynikow, odswiezany po kazdym zdjeciu):
    oprysk.geojson  punkty do oprysku - ten sam chwast z kilku zdjec to jeden punkt
    oprysk.csv      to samo jako tabela: id, szerokosc, dlugosc, promien, pewnosc
    chwasty.geojson wszystkie wykrycia ze wszystkich zdjec
    trasa.geojson   trasa drona i miejsca zdjec

Wspolrzedne liczone w lokalnym ukladzie metrycznym (stycznym do elipsoidy WGS84 w
punkcie pierwszego zdjecia). Na polu rzedu setek metrow blad tego przyblizenia to
ulamki milimetra - pomijalny wobec bledu GPS, a modul nie potrzebuje pyproj.
"""
import json
import math
from pathlib import Path

# elipsoida WGS84
_A = 6378137.0
_E2 = 6.69437999014e-3


class UkladLokalny:
    """(lat, lon) <-> (east, north) w metrach wzgledem punktu odniesienia.

    Po co: lekka wersja bez pyproj (binarka) - styczna do elipsoidy WGS84 w punkcie odniesienia.

    Uzywana w:
        MapaOprysku (ten plik), porownaj() (ten plik), mission/drone_sim.py
    """

    def __init__(self, lat0, lon0):
        """Liczy, ile metrow ma stopien szerokosci i dlugosci w punkcie odniesienia.

        Argumenty:
            lat0, lon0 - punkt odniesienia (np. pierwsze zdjecie)
        """
        self.lat0, self.lon0 = lat0, lon0
        f = math.radians(lat0)
        w = 1 - _E2 * math.sin(f) ** 2
        self._m = _A * (1 - _E2) / w ** 1.5 * math.pi / 180          # metry na stopien szerokosci
        self._n = _A / math.sqrt(w) * math.cos(f) * math.pi / 180     # metry na stopien dlugosci

    def na_metry(self, lat, lon):
        """(lat, lon) -> (east, north) w metrach od punktu odniesienia."""
        return (lon - self.lon0) * self._n, (lat - self.lat0) * self._m

    def na_stopnie(self, east, north):
        """(east, north) w metrach -> (lat, lon)."""
        return self.lat0 + north / self._m, self.lon0 + east / self._n


def piksel_na_metry(px, py, szer_px, wys_px, gsd_m, kurs_st):
    """Piksel zdjecia -> (east, north) w metrach od srodka zdjecia.

    Obraz: x w prawo, y w dol; przod drona to gora obrazu. Kurs od polnocy zgodnie
    z zegarem.

    Argumenty:
        px, py          - piksel (srodek ramki chwastu)
        szer_px, wys_px - rozmiar zdjecia
        gsd_m           - metry na piksel
        kurs_st         - kurs drona [st.]

    Zwraca:
        (east, north) [m]

    Uzywana w:
        MapaOprysku.dodaj_zdjecie() (ten plik)
    """
    dx = (px - szer_px / 2.0) * gsd_m
    dy = (py - wys_px / 2.0) * gsd_m
    t = math.radians(kurs_st or 0.0)
    return dx * math.cos(t) - dy * math.sin(t), -dx * math.sin(t) - dy * math.cos(t)


def _liczba(v):
    """Wartosc -> float albo None (puste, brak, nie-liczba)."""
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def czytaj_telemetrie(sciezka, zdjecie=None):
    """Plik telemetrii drona (format z opisu modulu) -> slownik albo None, gdy plik jest nieczytelny.

    Bierze punkt z properties.image == nazwa zdjecia, a gdy takiego nie ma - pierwszy punkt.

    Argumenty:
        sciezka - plik .geojson obok zdjecia
        zdjecie - nazwa zdjecia albo None

    Zwraca:
        dict: lon, lat, agl, gsd_mm, kurs, zrodlo_kursu ("heading" / "course" / None),
        przechylenie, pochylenie, fix, fix_quality, czas, zdjecie, szer_px, wys_px

    Uzywana w:
        mission/detect_cli.py main()
    """
    try:
        d = json.loads(Path(sciezka).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("type") != "FeatureCollection" or not d.get("features"):
        return None
    cechy = d["features"]
    wybrana = next((c for c in cechy if zdjecie and (c.get("properties") or {}).get("image") == zdjecie), cechy[0])
    geom, wl = wybrana.get("geometry") or {}, wybrana.get("properties") or {}
    survey, camera = wl.get("survey") or {}, wl.get("camera") or {}
    wsp = geom.get("coordinates") if geom.get("type") == "Point" else None
    if not wsp or len(wsp) < 2 or _liczba(wsp[0]) is None or _liczba(wsp[1]) is None:
        return None
    # kurs = kierunek, w ktory obrocony jest dron (heading, z kompasu / autopilota);
    # course_deg (kierunek RUCHU z GPS) tylko w locie >= 1 m/s - w zawisie to szum.
    kurs, zrodlo_kursu = _liczba(wl.get("heading_deg")), "heading"
    if kurs is None:
        zrodlo_kursu = None
        kurs_ruchu, predkosc = _liczba(wl.get("course_deg")), _liczba(wl.get("speed_mps"))
        if kurs_ruchu is not None and predkosc is not None and predkosc >= 1.0:
            kurs, zrodlo_kursu = kurs_ruchu, "course"
    fix = wl.get("fix")
    return {"lon": _liczba(wsp[0]), "lat": _liczba(wsp[1]),
            "agl": _liczba(survey.get("agl_m")), "gsd_mm": _liczba(survey.get("gsd_mm_per_px")),
            "kurs": kurs, "zrodlo_kursu": zrodlo_kursu,
            "przechylenie": _liczba(wl.get("roll_deg")), "pochylenie": _liczba(wl.get("pitch_deg")),
            "fix": fix if isinstance(fix, bool) else None, "fix_quality": _liczba(wl.get("fix_quality")),
            "czas": wl.get("time_utc"), "zdjecie": wl.get("image"),
            "szer_px": _liczba(camera.get("width")), "wys_px": _liczba(camera.get("height"))}


def blad_telemetrii(tele):
    """Powod, dla ktorego telemetria nie nadaje sie na mape, albo None, gdy jest w porzadku.

    Lapie typowe pomylki: wspolrzedne poza zakresem albo zerowe (brak fixa GPS),
    wysokosc nad poziomem morza zamiast nad polem, skala w zlych jednostkach.

    Argumenty:
        tele - wynik czytaj_telemetrie()

    Zwraca:
        str z opisem problemu albo None

    Uzywana w:
        mission/detect_cli.py main() - zdjecie idzie wtedy bez GPS, z ostrzezeniem
    """
    if not (-90 <= tele["lat"] <= 90 and -180 <= tele["lon"] <= 180):
        return f"wspolrzedne poza zakresem ({tele['lat']}, {tele['lon']}) - zamieniona kolejnosc?"
    if tele["lat"] == 0 and tele["lon"] == 0:
        return "wspolrzedne 0, 0 - brak pozycji GPS"
    if tele.get("fix") is False or tele.get("fix_quality") == 0:
        return "brak fixa GPS (fix = false / fix_quality = 0)"
    if tele.get("kurs") is None:
        return ("brak kursu drona - dodaj heading_deg (z kompasu/autopilota); course_deg uzywany "
                "tylko w locie >= 1 m/s (test bez kursu: --kurs-domyslny 0)")
    if tele.get("agl") is not None and not 0.3 <= tele["agl"] <= 150:
        return f"wysokosc nad polem {tele['agl']} m - poza zakresem 0,3-150 m (wysokosc npm zamiast nad polem?)"
    if tele.get("gsd_mm") is not None and not 0.1 <= tele["gsd_mm"] <= 100:
        return f"gsd {tele['gsd_mm']} mm/px - poza zakresem 0,1-100 (zle jednostki?)"
    if not math.isfinite(tele["kurs"]):
        return "kurs nie jest liczba"
    return None


def przechyl_st(tele):
    """Najwiekszy z katow przechylenia i pochylenia [st.] albo None, gdy telemetria ich nie ma.

    Uzywana w:
        MapaOprysku.dodaj_zdjecie() (niepewnosc polozenia), mission/detect_cli.py (limit --maks-przechyl)
    """
    k = [abs(v) for v in (tele.get("przechylenie"), tele.get("pochylenie")) if v is not None]
    return max(k) if k else None


def plik_telemetrii(zdjecie):
    """Telemetria lezaca obok zdjecia: ta sama nazwa z rozszerzeniem .geojson (None, gdy brak).

    Argumenty:
        zdjecie - Path zdjecia, np. "lot/20261109T142859.998Z.jpg"

    Zwraca:
        Path(".../20261109T142859.998Z.geojson") albo None (brak albo pusty plik)

    Uzywana w:
        mission/detect_cli.py main()
    """
    p = Path(zdjecie).with_suffix(".geojson")
    return p if p.is_file() and p.stat().st_size > 0 else None


def gsd_m(tele, szer_obrazu_px, kamera=None):
    """Metry na piksel dla zdjecia: survey.gsd_mm_per_px albo survey.agl_m + model kamery.

    Argumenty:
        tele           - wynik czytaj_telemetrie()
        szer_obrazu_px - szerokosc zdjecia
        kamera         - model kamery (stitchlive/camera.py, np. ARDUCAM_64MP_16MP) albo None

    Zwraca:
        float [m/px] albo None

    Uzywana w:
        mission/detect_cli.py main() (skala + dobor kafelka do 1,5 m pola)
    """
    if tele.get("gsd_mm"):
        return tele["gsd_mm"] / 1000.0
    if tele.get("agl") and kamera is not None:
        return kamera.zasieg_m(tele["agl"])[0] / szer_obrazu_px
    return None


def _zapisz_atomowo(sciezka, tekst):
    """Zapis tekstu przez plik tymczasowy i zmiane nazwy - nigdy pol pliku."""
    tmp = sciezka.with_name(f".{sciezka.name}.tmp")
    tmp.write_text(tekst, encoding="utf-8")
    tmp.replace(sciezka)


class MapaOprysku:
    """Zbiera wykrycia z kolejnych zdjec i laczy je w punkty oprysku.

    Ten sam chwast widziany na kilku zachodzacych zdjeciach trafia do jednego
    punktu, gdy wykrycie lezy blizej niz promien_scalania_m od punktu. Polozenie
    punktu to srednia wazona pewnoscia wszystkich jego wykryc.

    Uzywana w:
        mission/detect_cli.py main() (po kazdym zdjeciu dodaj_zdjecie + zapisz)
    """

    def __init__(self, promien_scalania_m=0.10, margines_m=0.05):
        """Argumenty:
            promien_scalania_m - wykrycia blizej niz tyle metrow to jeden punkt, np. 0.10
            margines_m         - dodatek do promienia oprysku wokol chwastu, np. 0.05
        """
        self.promien = promien_scalania_m
        self.margines = margines_m
        self.uklad = None
        self.wykrycia = []           # (lat, lon, pewnosc, rozmiar_m, zdjecie, czas)
        self.punkty = []             # [suma_w, sum_e*w, sum_n*w, pewn_max, rozmiar_max, liczba, zdjecia]
        self.trasa = []              # (lat, lon, zdjecie, czas)

    def dodaj_zdjecie(self, tele, zdjecie, szer, wys, wykrycia, gsd):
        """wykrycia: [(x0, y0, x1, y1, pewnosc)] w pikselach zdjecia. Zwraca liczbe nowych punktow.

        Kazde wykrycie: srodek ramki -> metry (piksel_na_metry) -> stopnie -> scalenie z punktami.

        Argumenty:
            tele      - telemetria zdjecia (lat, lon, kurs, agl, ...)
            zdjecie   - nazwa zdjecia
            szer, wys - rozmiar zdjecia
            wykrycia  - lista ramek z pewnoscia
            gsd       - metry na piksel

        Zwraca:
            int - ile nowych punktow oprysku doszlo
        """
        if self.uklad is None:
            self.uklad = UkladLokalny(tele["lat"], tele["lon"])
        self.trasa.append((tele["lat"], tele["lon"], zdjecie, tele.get("czas")))
        e0, n0 = self.uklad.na_metry(tele["lat"], tele["lon"])
        k = przechyl_st(tele)
        niepewnosc = round(tele["agl"] * math.tan(math.radians(k)), 3) if k is not None and tele.get("agl") else None
        nowe = 0
        for x0, y0, x1, y1, pw in wykrycia:
            de, dn = piksel_na_metry((x0 + x1) / 2, (y0 + y1) / 2, szer, wys, gsd, tele["kurs"])
            e, n = e0 + de, n0 + dn
            rozmiar = max(x1 - x0, y1 - y0) * gsd
            lat, lon = self.uklad.na_stopnie(e, n)
            self.wykrycia.append((lat, lon, pw, rozmiar, zdjecie, tele.get("czas"), niepewnosc))
            nowe += self._scal(e, n, pw, rozmiar, zdjecie)
        return nowe

    def _scal(self, e, n, pw, rozmiar, zdjecie):
        """Dolacza wykrycie do najblizszego punktu w promieniu albo zaklada nowy punkt.

        Zwraca:
            1 = nowy punkt, 0 = dolaczone do istniejacego
        """
        najblizszy, odl = None, self.promien
        for p in self.punkty:
            d = math.hypot(p[1] / p[0] - e, p[2] / p[0] - n)
            if d <= odl:
                najblizszy, odl = p, d
        if najblizszy is None:
            self.punkty.append([pw, e * pw, n * pw, pw, rozmiar, 1, {zdjecie}])
            return 1
        p = najblizszy
        p[0] += pw; p[1] += e * pw; p[2] += n * pw
        p[3] = max(p[3], pw); p[4] = max(p[4], rozmiar); p[5] += 1; p[6].add(zdjecie)
        return 0

    def punkty_oprysku(self):
        """[(id, lat, lon, promien_m, pewnosc_max, liczba_wykryc, liczba_zdjec)]

        Promien = polowa najwiekszego chwastu + margines.
        """
        out = []
        for i, (w, se, sn, pmax, rozm, n, zdj) in enumerate(self.punkty, 1):
            lat, lon = self.uklad.na_stopnie(se / w, sn / w)
            out.append((i, lat, lon, round(rozm / 2 + self.margines, 3), round(pmax, 3), n, len(zdj)))
        return out

    def zapisz(self, katalog):
        """Zapisuje oprysk.geojson, oprysk.csv, chwasty.geojson i trasa.geojson (atomowo).

        Argumenty:
            katalog - folder WYNIKI
        """
        katalog = Path(katalog)
        if self.uklad is None:
            return
        oprysk = self.punkty_oprysku()
        fc = lambda cechy: json.dumps({"type": "FeatureCollection", "features": cechy}, ensure_ascii=False)  # noqa: E731
        _zapisz_atomowo(katalog / "oprysk.geojson", fc([
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lon, 8), round(lat, 8)]},
             "properties": {"id": i, "promien_m": r, "pewnosc": pw, "liczba_wykryc": n, "liczba_zdjec": z}}
            for i, lat, lon, r, pw, n, z in oprysk]))
        linie = ["id,szerokosc,dlugosc,promien_m,pewnosc,liczba_zdjec"]
        linie += [f"{i},{lat:.8f},{lon:.8f},{r},{pw},{z}" for i, lat, lon, r, pw, n, z in oprysk]
        _zapisz_atomowo(katalog / "oprysk.csv", "\n".join(linie) + "\n")
        _zapisz_atomowo(katalog / "chwasty.geojson", fc([
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lon, 8), round(lat, 8)]},
             "properties": {"pewnosc": round(pw, 3), "rozmiar_m": round(rozm, 3), "zdjecie": zdj, "czas": czas,
                            "niepewnosc_przechylu_m": niep}}
            for lat, lon, pw, rozm, zdj, czas, niep in self.wykrycia]))
        _zapisz_atomowo(katalog / "trasa.geojson", fc(
            [{"type": "Feature", "geometry": {"type": "LineString",
                                              "coordinates": [[round(lo, 8), round(la, 8)] for la, lo, _, _ in self.trasa]},
              "properties": {"nazwa": "trasa drona"}}] +
            [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lo, 8), round(la, 8)]},
              "properties": {"zdjecie": z, "czas": c}} for la, lo, z, c in self.trasa]))


def porownaj(oprysk_geojson, prawda_geojson, promien_m=0.10):
    """Ocena namiarow: ile prawdziwych chwastow ma punkt oprysku w promieniu promien_m.

    Zwraca slownik: trafione, pominiete, falszywe, czulosc, precyzja, mediana i 90. centyl
    odleglosci trafionych punktow od prawdziwych chwastow [cm]. Dopasowanie zachlanne
    od najblizszych par, kazdy chwast i kazdy punkt najwyzej raz.

    Argumenty:
        oprysk_geojson - oprysk.geojson z fieldeye-detect
        prawda_geojson - prawda.geojson z symulatora drona
        promien_m      - trafienie, gdy blizej niz tyle metrow, np. 0.10

    Uzywana w:
        main() (ten plik), scripts/fieldeye_detect_main.py (fieldeye-detect porownaj), tests
    """
    def punkty(p):
        """Punkty z pliku GeoJSON -> lista (lat, lon)."""
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        return [(c["geometry"]["coordinates"][1], c["geometry"]["coordinates"][0]) for c in d["features"]
                if (c.get("geometry") or {}).get("type") == "Point"]
    op, pr = punkty(oprysk_geojson), punkty(prawda_geojson)
    if not pr:
        raise ValueError("pusta prawda")
    u = UkladLokalny(*pr[0])
    E = [u.na_metry(*a) for a in op]
    P = [u.na_metry(*b) for b in pr]
    pary = sorted((math.hypot(a[0] - b[0], a[1] - b[1]), i, j)
                  for i, a in enumerate(E) for j, b in enumerate(P)
                  if abs(a[0] - b[0]) <= promien_m and abs(a[1] - b[1]) <= promien_m)
    uz_o, uz_p, odl = set(), set(), []
    for d, i, j in pary:
        if d <= promien_m and i not in uz_o and j not in uz_p:
            uz_o.add(i); uz_p.add(j); odl.append(d)
    t = len(odl)
    odl.sort()
    return {"trafione": t, "pominiete": len(P) - t, "falszywe": len(E) - t,
            "czulosc": round(t / len(P), 3), "precyzja": round(t / max(1, len(E)), 3),
            "mediana_cm": round(100 * odl[len(odl) // 2], 1) if odl else None,
            "p90_cm": round(100 * odl[int(0.9 * (len(odl) - 1))], 1) if odl else None}


def main(argv=None):
    """Wiersz polecen: python -m mission.spray_map OPRYSK PRAWDA [--promien 0.10] -> wynik porownaj().

    Zwraca:
        0
    """
    import argparse
    ap = argparse.ArgumentParser(description="Porownanie punktow oprysku z prawdziwymi chwastami (symulacja).")
    ap.add_argument("oprysk", help="oprysk.geojson z fieldeye-detect")
    ap.add_argument("prawda", help="prawda.geojson z symulatora drona")
    ap.add_argument("--promien", type=float, default=0.10, help="trafienie, gdy punkt jest blizej niz tyle metrow")
    a = ap.parse_args(argv)
    w = porownaj(a.oprysk, a.prawda, a.promien)
    print(f"  prawdziwych chwastow: {w['trafione'] + w['pominiete']},  punktow oprysku: {w['trafione'] + w['falszywe']}")
    print(f"  trafione {w['trafione']}, pominiete {w['pominiete']}, falszywe {w['falszywe']}  "
          f"(czulosc {w['czulosc']:.2f}, precyzja {w['precyzja']:.2f}, promien {a.promien * 100:.0f} cm)")
    if w["mediana_cm"] is not None:
        print(f"  odleglosc punktu oprysku od chwastu: mediana {w['mediana_cm']} cm, 90% ponizej {w['p90_cm']} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
