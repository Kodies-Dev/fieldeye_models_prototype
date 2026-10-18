"""ETAP 2 - po wyladowaniu. Scalenie detekcji, mapa, mozaika.

Trzy kroki, kazdy niezalezny:

1. SCALENIE. Przy pokryciu 60-80% ten sam chwast jest widoczny na dwoch-trzech
   zdjeciach, wiec w wynikach z lotu wystepuje kilka razy. Do mapy zabiegowej
   potrzebna jest jedna obserwacja na rosline.

2. MAPA. Zapis GeoJSON (otwiera sie w QGIS i prawie kazdym narzedziu GIS)
   plus prosty podglad HTML, ktory dziala bez instalowania czegokolwiek.

3. MOZAIKA. Zlozenie zdjec w jeden obraz - juz istniejacym kodem z pakietu
   stitchlive. Sluzy czlowiekowi do ogladania; do samej mapy nie jest potrzebna.
"""

import json
import logging
import math
from pathlib import Path

from . import geo
from .in_flight import wczytaj_wyniki

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- scalanie

def scal_detekcje(wyniki, promien_m=0.35, min_pewnosc=0.0, klasy=None):
    """Laczy detekcje tej samej rosliny z nakladajacych sie zdjec.

    Zasada: detekcje sortowane od najpewniejszej; kazda kolejna, ktora lezy
    blizej niz promien_m od juz przyjetej (tej samej klasy), jest uznawana
    za to samo i tylko dopisuje sie jako kolejna obserwacja.

    Liczba obserwacji jest uzyteczna sama w sobie: roslina widziana na trzech
    zdjeciach jest pewniejsza niz ta widziana raz.

    Siatka zamiast porownywania kazdy z kazdym - przy tysiacach detekcji
    roznica jest miedzy sekunda a kilkoma minutami.
    """
    punkty = []
    for rek in wyniki:
        for d in rek["detekcje"]:
            if "lat" not in d or d["pewnosc"] < min_pewnosc:
                continue
            if klasy and d["klasa"] not in klasy:
                continue
            punkty.append({**d, "kadr": rek["kadr"]["plik"]})

    if not punkty:
        # najczestsza przyczyna pustej mapy - prog odciecia wyzszy niz pewnosci
        wszystkie = [d for rek in wyniki for d in rek["detekcje"] if "lat" in d]
        if wszystkie and min_pewnosc > 0:
            najwyzsza = max(d["pewnosc"] for d in wszystkie)
            if najwyzsza < min_pewnosc:
                log.warning("Wszystkie %d detekcji odrzucone przez prog pewnosci "
                            "(%.2f); najwyzsza pewnosc to %.3f. Obniz "
                            "mapa.min_pewnosc_na_mape w konfiguracji.",
                            len(wszystkie), min_pewnosc, najwyzsza)

    punkty.sort(key=lambda d: -d["pewnosc"])
    bok = promien_m / 111320.0                      # promien w przyblizeniu na stopnie
    siatka, wynik = {}, []

    for p in punkty:
        gx, gy = int(p["lon"] / bok), int(p["lat"] / bok)
        trafiony = None
        # wystarczy przejrzec 9 sasiednich komorek siatki
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for kand in siatka.get((gx + dx, gy + dy), ()):
                    if kand["klasa"] != p["klasa"]:
                        continue
                    if geo.odleglosc_m(p["lat"], p["lon"],
                                       kand["lat"], kand["lon"]) < promien_m:
                        trafiony = kand
                        break
                if trafiony:
                    break
            if trafiony:
                break

        if trafiony:
            trafiony["obserwacje"] += 1
            trafiony["kadry"].append(p["kadr"])
        else:
            nowy = {"lat": p["lat"], "lon": p["lon"], "klasa": p["klasa"],
                    "pewnosc": p["pewnosc"], "rozmiar_m": p.get("rozmiar_m"),
                    "obserwacje": 1, "kadry": [p["kadr"]]}
            siatka.setdefault((gx, gy), []).append(nowy)
            wynik.append(nowy)

    log.info("Scalanie: %d detekcji -> %d roslin", len(punkty), len(wynik))
    return wynik


# ---------------------------------------------------------------- mapa

def zapisz_geojson(rosliny, sciezka):
    """GeoJSON w WGS84 - wczytuje go QGIS, Folium, geopandas i reszta swiata."""
    cechy = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
        "properties": {k: v for k, v in r.items() if k not in ("lat", "lon")},
    } for r in rosliny]
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
          "features": cechy}
    sciezka = Path(sciezka)
    sciezka.parent.mkdir(parents=True, exist_ok=True)
    sciezka.write_text(json.dumps(fc, indent=1, ensure_ascii=False), encoding="utf-8")
    return sciezka


def zapisz_slad_lotu(wyniki, sciezka):
    """Trasa drona jako linia - do sprawdzenia, czy misja objela cale pole."""
    pkt = [[r["kadr"]["lon"], r["kadr"]["lat"]] for r in wyniki
           if r["kadr"].get("lat") is not None]
    fc = {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": pkt},
        "properties": {"kadrow": len(pkt)}}]}
    Path(sciezka).write_text(json.dumps(fc, indent=1), encoding="utf-8")
    return sciezka


def podglad_html(rosliny, wyniki, sciezka, tytul="Mapa zachwaszczenia"):
    """Mapa w przegladarce bez instalowania czegokolwiek.

    Uzywa Leaflet z internetu; gdy go nie ma, punkty i tak sa w GeoJSON.
    """
    pkt = [[r["kadr"]["lat"], r["kadr"]["lon"]] for r in wyniki
           if r["kadr"].get("lat") is not None]
    srodek = (sum(p[0] for p in pkt) / len(pkt), sum(p[1] for p in pkt) / len(pkt)) \
        if pkt else (rosliny[0]["lat"], rosliny[0]["lon"]) if rosliny else (52.0, 19.0)

    dane = [{"lat": r["lat"], "lon": r["lon"], "k": r["klasa"],
             "p": round(r["pewnosc"], 2), "o": r["obserwacje"]} for r in rosliny]

    html = f"""<!DOCTYPE html><html lang="pl"><head><meta charset="utf-8">
<title>{tytul}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#m{{height:100%;margin:0}}
.info{{position:absolute;z-index:1000;right:10px;top:10px;background:#fff;
padding:10px 14px;font:14px sans-serif;border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,.3)}}</style>
</head><body><div id="m"></div>
<div class="info"><b>{tytul}</b><br>roslin: {len(rosliny)}<br>kadrow: {len(pkt)}</div>
<script>
var m=L.map('m').setView([{srodek[0]},{srodek[1]}],19);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
 {{maxZoom:22,maxNativeZoom:19,attribution:'© OpenStreetMap'}}).addTo(m);
var slad={json.dumps(pkt)};
if(slad.length) L.polyline(slad,{{color:'#3388ff',weight:2,opacity:.6}}).addTo(m);
var d={json.dumps(dane, ensure_ascii=False)};
d.forEach(function(r){{
  L.circleMarker([r.lat,r.lon],{{radius:4+r.o,color:'#c0392b',fillColor:'#e74c3c',
   fillOpacity:.75,weight:1}}).addTo(m)
   .bindPopup(r.k+'<br>pewnosc: '+r.p+'<br>obserwacji: '+r.o);
}});
if(d.length) m.fitBounds(L.latLngBounds(d.map(function(r){{return [r.lat,r.lon]}})).pad(.15));
</script></body></html>"""
    Path(sciezka).write_text(html, encoding="utf-8")
    return sciezka


# ---------------------------------------------------------------- mozaika

def zloz_mozaike(katalog_zdjec, wyjscie, megapix=0.6, zakres_par=0, szwy=False):
    """Skladanie mozaiki istniejacym kodem z pakietu stitchlive.

    Wydzielone do osobnej funkcji, bo to jedyne miejsce, gdzie etap po locie
    zalezy od stitchlive - mape mozna zrobic bez tego.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from stitchlive.global_stitcher import GlobalStitcher

    pliki = sorted(p for p in Path(katalog_zdjec).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    s = GlobalStitcher(work_megapix=megapix, zakres_par=zakres_par,
                         szwy=szwy, renderuj_co=10 ** 6)
    for p in pliki:
        s.dodaj(p)
    s.zakoncz()
    if s.mozaika is None:
        log.warning("Nie udalo sie zlozyc mozaiki.")
        return None
    import cv2
    Path(wyjscie).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(wyjscie), s.mozaika, [cv2.IMWRITE_JPEG_QUALITY, 90])
    log.info("Mozaika %dx%d -> %s", s.mozaika.shape[1], s.mozaika.shape[0], wyjscie)
    return Path(wyjscie)


# ---------------------------------------------------------------- raport

def raport(wyniki, rosliny, cfg=None):
    """Podsumowanie misji - liczby do wydruku i do pliku JSON."""
    z_poz = [r for r in wyniki if r["kadr"].get("lat") is not None]
    wszystkie = sum(len(r["detekcje"]) for r in wyniki)
    czasy = [r["czas_s"] for r in wyniki if "czas_s" in r]
    agl = [r["kadr"]["agl"] for r in wyniki if r["kadr"].get("agl")]
    gsd = [r["gsd_m_px"] for r in wyniki if r.get("gsd_m_px")]

    rap = {
        "kadrow": len(wyniki),
        "kadrow_z_pozycja": len(z_poz),
        "detekcji_lacznie": wszystkie,
        "roslin_po_scaleniu": len(rosliny),
        "duplikatow_usunietych": wszystkie - len(rosliny) if rosliny else 0,
        "sredni_czas_kadru_s": round(sum(czasy) / len(czasy), 3) if czasy else None,
        "srednia_wysokosc_m": round(sum(agl) / len(agl), 1) if agl else None,
        "sredni_gsd_m_px": round(sum(gsd) / len(gsd), 5) if gsd else None,
    }
    if rosliny:
        rap["obserwacji_na_rosline"] = round(
            sum(r["obserwacje"] for r in rosliny) / len(rosliny), 2)
        klasy = {}
        for r in rosliny:
            klasy[r["klasa"]] = klasy.get(r["klasa"], 0) + 1
        rap["klasy"] = klasy
    if rap["sredni_gsd_m_px"] and rap["srednia_wysokosc_m"]:
        rap["uwaga_gsd"] = (f"1 piksel = {rap['sredni_gsd_m_px'] * 100:.2f} cm terenu "
                            f"przy {rap['srednia_wysokosc_m']} m wysokosci")
    return rap
