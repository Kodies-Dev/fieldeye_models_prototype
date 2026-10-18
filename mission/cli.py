"""Wiersz polecen misji.

    python -m mission.cli wlocie  zdjecia/ -o mission/      # etap 1 (wsadowo)
    python -m mission.cli lot     naplyw/  -o mission/      # etap 1 (na zywo)
    python -m mission.cli polocie mission/wyniki.jsonl -o mission/
    python -m mission.cli pelna   zdjecia/ -o mission/      # oba etapy naraz
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .detector import zbuduj_detektor
from .config import MissionConfig
from . import post_flight as P
from . import in_flight as W

log = logging.getLogger("mission")


def _postep(i, n):
    if n:
        sys.stderr.write(f"\r  {i}/{n} ({100 * i / n:.0f}%)")
        if i == n:
            sys.stderr.write("\n")
    else:
        sys.stderr.write(f"\r  kadrow: {i}")
    sys.stderr.flush()


def _cfg(a):
    cfg = MissionConfig.wczytaj(a.config)
    if getattr(a, "model", None):
        cfg.detekcja.model = a.model
    if getattr(a, "agl", None):
        cfg.lot.agl_domyslne = a.agl
    if getattr(a, "bez_kafelkow", False):
        cfg.detekcja.kafelkowanie = False
    return cfg


def cmd_wlocie(a):
    cfg = _cfg(a)
    det = zbuduj_detektor(cfg.detekcja)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    plik = out / "wyniki.jsonl"
    if plik.exists():
        plik.unlink()                       # nowa misja = nowy plik
    W.przetworz_katalog(a.zdjecia, plik, det, cfg, a.telemetria, a.limit, _postep)
    print(f"\nWyniki: {plik}")
    print("Nastepny krok: python -m mission.cli polecie", plik, "-o", out)
    return 0


def cmd_lot(a):
    cfg = _cfg(a)
    det = zbuduj_detektor(cfg.detekcja)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    plik = out / "wyniki.jsonl"
    print(f"Obserwuje {a.zdjecia}. Ctrl+C konczy.")
    try:
        W.obserwuj_katalog(a.zdjecia, plik, det, cfg, a.telemetria,
                           a.interwal, a.timeout, _postep)
    except KeyboardInterrupt:
        print("\nPrzerwano.")
    print(f"Wyniki: {plik}")
    return 0


def cmd_polocie(a):
    cfg = _cfg(a)
    wyniki = W.wczytaj_wyniki(a.wyniki)
    if not wyniki:
        raise SystemExit("pusty plik wynikow")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    rosliny = P.scal_detekcje(wyniki, cfg.mapa.promien_scalania_m,
                              cfg.mapa.min_pewnosc_na_mape,
                              klasy=cfg.detekcja.klasy_chwastow if a.tylko_chwasty else None)
    P.zapisz_geojson(rosliny, out / "chwasty.geojson")
    P.zapisz_slad_lotu(wyniki, out / "slad_lotu.geojson")
    P.podglad_html(rosliny, wyniki, out / "mapa.html")

    if a.zdjecia:
        P.zloz_mozaike(a.zdjecia, out / "mozaika.jpg", a.megapix, a.zakres_par, a.szwy)

    rap = P.raport(wyniki, rosliny, cfg)
    (out / "raport.json").write_text(json.dumps(rap, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print()
    for k, v in rap.items():
        print(f"  {k:<26} {v}")
    print(f"\nMapa:     {out / 'mapa.html'}  (otworz w przegladarce)")
    print(f"GeoJSON:  {out / 'chwasty.geojson'}  (QGIS)")
    return 0


def cmd_pelna(a):
    cmd_wlocie(a)
    a.wyniki = Path(a.output) / "wyniki.jsonl"
    return cmd_polocie(a)


def cmd_symulacja(a):
    """Symulowany lot nad kafelkami kazdego dnia: zwykle scalanie kontra laczenie widokow."""
    import json
    import tempfile
    from stitchlive.flight_sim import SimSettings
    from . import detector as D
    from .sim_flight import kafelki_wedlug_dni, lot_z_detekcja, pole_z_kafelkow

    # niski prog detektora: slabe wykrycia to dowody dla laczenia widokow
    det = D.zbuduj_z_plikow(a.model, 0.05, a.wejscie, tta=a.tta)
    dni = kafelki_wedlug_dni(a.kafelki)
    wybrane = a.dni or [d for d, p in dni.items() if len(p) >= 4]
    wyniki = {}
    for dzien in wybrane:
        pole, prawda = pole_z_kafelkow(dni[dzien], z_etykietami=True)
        u = SimSettings(kadr=a.kadr, overlap=a.pokrycie, sidelap=0.5,
                        obrot=2.0 if a.zaklocenia else 0.0, ekspozycja=0.03 if a.zaklocenia else 0.0,
                        jitter=0.08 if a.zaklocenia else 0.0, seed=7)
        w = lot_z_detekcja(pole, prawda, det, u, tempfile.mkdtemp(prefix="lot_"), prog=a.prog)
        wyniki[dzien] = {"kadrow": w["kadrow"], "w_mozaice": w["w_mozaice"], **w["metryki"]}
        print(f"\n  {dzien}: {w['kadrow']} kadrow, w mozaice {w['w_mozaice']}")
        print(f"    {'sposob':<26}{'trafione':>9}{'pominiete':>10}{'falszywe':>9}{'czulosc':>9}{'precyzja':>9}")
        for k, m in w["metryki"].items():
            print(f"    {k:<26}{m['trafione']:>9}{m['pominiete']:>10}{m['falszywe']:>9}"
                  f"{m['czulosc']:>9.3f}{m['precyzja']:>9.3f}")
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps({"model": [str(m) for m in a.model], "tta": a.tta,
                                              "prog": a.prog,
                                              "zaklocenia": a.zaklocenia, "dni": wyniki},
                                             indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def parser():
    ap = argparse.ArgumentParser(prog="mission", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def wspolne(p):
        p.add_argument("-o", "--output", default="misja_out")
        p.add_argument("--config", help="plik YAML/JSON z konfiguracja")
        p.add_argument("--model", help="sciezka do modelu YOLO (.pt); brak = atrapa testowa")
        p.add_argument("--agl", type=float, help="wysokosc lotu [m], gdy brak w metadanych")
        p.add_argument("--telemetria", help="plik CSV z telemetria kapsuly")
        p.add_argument("--bez-kafelkow", action="store_true",
                       help="nie tnij kadru na kafelki (szybciej, gorzej na malych obiektach)")

    def po_locie(p):
        p.add_argument("--zdjecia", help="katalog zdjec - gdy podany, sklada tez mozaike")
        p.add_argument("--megapix", type=float, default=0.6)
        p.add_argument("--zakres-par", type=int, default=0)
        p.add_argument("--szwy", action="store_true")
        p.add_argument("--tylko-chwasty", action="store_true",
                       help="na mape trafiaja tylko klasy uznane za chwasty")

    p = sub.add_parser("wlocie", help="etap 1 wsadowo: gotowy katalog zdjec")
    p.add_argument("zdjecia")
    p.add_argument("--limit", type=int)
    wspolne(p)
    p.set_defaults(func=cmd_wlocie)

    p = sub.add_parser("lot", help="etap 1 na zywo: obserwuj katalog w trakcie lotu")
    p.add_argument("zdjecia")
    p.add_argument("--interwal", type=float, default=1.0)
    p.add_argument("--timeout", type=float, help="zakoncz po N s bez nowych zdjec")
    wspolne(p)
    p.set_defaults(func=cmd_lot)

    p = sub.add_parser("polocie", help="etap 2: scalenie, mapa, mozaika")
    p.add_argument("wyniki", help="plik wyniki.jsonl z etapu 1")
    wspolne(p)
    po_locie(p)
    p.set_defaults(func=cmd_polocie)

    p = sub.add_parser("symulacja", help="symulowany lot nad kafelkami: scalanie kontra laczenie widokow")
    p.add_argument("--model", required=True, nargs="+", help=".pt / .hef; kilka = ensemble (WBF)")
    p.add_argument("--tta", action="store_true", help="dodatkowo detekcja na odbiciu poziomym")
    p.add_argument("--kafelki", default="zbior_lotnicze/dataset/images/test")
    p.add_argument("--dni", nargs="*", help="domyslnie wszystkie dni z co najmniej 4 kafelkami")
    p.add_argument("--prog", type=float, default=0.15)
    p.add_argument("--wejscie", type=int, default=1280)
    p.add_argument("--kadr", type=int, default=600)
    p.add_argument("--pokrycie", type=float, default=0.7)
    p.add_argument("--zaklocenia", action="store_true")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_symulacja)

    p = sub.add_parser("pelna", help="oba etapy naraz")
    p.add_argument("zdjecia")
    p.add_argument("--limit", type=int)
    wspolne(p)
    po_locie(p)
    p.set_defaults(func=cmd_pelna)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    try:
        return a.func(a)
    except Exception as e:
        log.error("%s", e)
        if a.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())