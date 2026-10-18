"""Orkiestracja: zbiory zrodlowe -> gotowy zbior treningowy YOLO.

    python -m normalization.build --weedsgalore dane/weedsgalore-dataset \\
        --sudars dane/sudars/dataset -o zbior --tryb-klas binarnie

Kolejnosc krokow ma znaczenie:

  adaptery -> deduplikacja -> mapowanie klas -> podzial -> kafelkowanie -> eksport

Deduplikacja PRZED podzialem, bo inaczej duplikat moze trafic do treningu
i do testu. Kafelkowanie PO podziale, bo kafelki dziedzicza split ze zdjecia -
inaczej kafelki tego samego zdjecia rozjechalyby sie po czesciach.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from . import export as E
from . import steps as K
from . import manifest as M
from .adapters import ADAPTERY
from .taxonomy import Taxonomy

log = logging.getLogger("normalization")


def zbuduj(zrodla, wyjscie, tryb_klas="binarnie", kafelek=896, zakladka=0.1,
           udzial_pustych=0.15, prog_dedup=5, proporcje=(0.70, 0.15, 0.15),
           splity_weedsgalore=None, taksonomia=None, kopiuj_obrazy=True,
           tylko_wlasne_do_testu=False, oficjalne_splity=True, seed=0, skale=None, limity=None,
           opcje_adapterow=None):
    """Caly potok. zrodla: {"weedsgalore": "sciezka", ...}"""
    tax = Taxonomy(taksonomia)
    wyjscie = Path(wyjscie)
    wyjscie.mkdir(parents=True, exist_ok=True)

    # 1. adaptery
    rekordy = []
    for nazwa, sciezka in zrodla.items():
        if not sciezka or not Path(sciezka).exists():
            log.warning("pomijam %s - brak katalogu %s", nazwa, sciezka)
            continue
        kw = {"splity": splity_weedsgalore} if nazwa == "weedsgalore" else {}
        kw.update((opcje_adapterow or {}).get(nazwa, {}))
        czesc = ADAPTERY[nazwa](sciezka, tax, **kw)
        log.info("%-12s %4d zdjec, %5d obiektow", nazwa, len(czesc),
                 sum(r.liczba_obiektow for r in czesc))
        rekordy += czesc

    if not rekordy:
        raise SystemExit("nie wczytano zadnego zdjecia - sprawdz sciezki")

    # 1b. limit zdjec z duzych zrodel (np. WeedMaize: 7784 zdjec)
    rekordy = K.ogranicz(rekordy, limity or {}, seed)

    # 2. deduplikacja
    rekordy, odrzucone = K.deduplikuj(rekordy, prog_dedup)

    # 3. mapowanie klas na etykiety treningowe
    K.zastosuj_tryb_klas(rekordy, tax, tryb_klas)

    # 4. podzial - PRZED kafelkowaniem
    K.podziel(rekordy, proporcje, zachowaj_istniejace=oficjalne_splity,
              tylko_wlasne_do_testu=tylko_wlasne_do_testu)
    kontrola = K.sprawdz_podzial(rekordy)
    if kontrola["bledy"]:
        log.error("BLAD: grupy w kilku czesciach podzialu: %s", kontrola["bledy"])
    for g, cz in kontrola["oficjalne"].items():
        log.warning("grupa %s jest w %s - tak dzieli ja oficjalny podzial autorow; "
                    "sprawdz, czy to samo miejsce pola nie trafia do dwoch czesci "
                    "(albo uzyj --wlasny-podzial)", g, "/".join(cz))

    # 4b. skala zdjec z ziemi do skali drona - przed kafelkowaniem
    if skale:
        K.przeskaluj(rekordy, skale)

    M.zapisz(rekordy, wyjscie / "manifest_przed_kafelkami.jsonl")

    # 5. kafelkowanie
    if kafelek:
        rekordy = K.kafelkuj_wszystkie(rekordy, kafelek=kafelek, zakladka=zakladka,
                                       udzial_pustych=udzial_pustych, seed=seed)

    M.zapisz(rekordy, wyjscie / "manifest.jsonl")

    # 6. eksport
    nazwy = sorted({o.klasa for r in rekordy for o in r.obiekty})
    wynik = E.eksportuj(rekordy, wyjscie / "dataset", nazwy, kopiuj_obrazy)
    E.podglad_ramek(rekordy, wyjscie / "podglad")

    stat = M.statystyki(rekordy)
    info = {
        "zrodla": {k: str(v) for k, v in zrodla.items()},
        "parametry": {"tryb_klas": tryb_klas, "kafelek": kafelek, "zakladka": zakladka,
                      "udzial_pustych": udzial_pustych, "prog_dedup": prog_dedup,
                      "proporcje": list(proporcje), "seed": seed,
                      "oficjalne_splity": oficjalne_splity,
                      "skale": skale or {}, "limity": limity or {}},
        "statystyki": stat,
        "klasy": wynik["klasy"],
        "obiektow_na_klase": wynik["obiektow"],
        "odrzucone_duplikaty": len(odrzucone),
        "nierozpoznane_etykiety": tax.raport_nieznanych(),
        "kontrola_podzialu": kontrola,
    }
    E.zapisz_info(wyjscie, info)
    (wyjscie / "duplikaty.json").write_text(
        json.dumps(odrzucone, indent=1, ensure_ascii=False), encoding="utf-8")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for z in ADAPTERY:
        ap.add_argument(f"--{z}", help=f"katalog zbioru {z}")
    ap.add_argument("-o", "--output", default="zbior")
    ap.add_argument("--tryb-klas", default="binarnie",
                    choices=["binarnie", "chwasty", "gatunki"])
    ap.add_argument("--kafelek", type=int, default=896, help="0 = bez kafelkowania")
    ap.add_argument("--zakladka", type=float, default=0.1)
    ap.add_argument("--udzial-pustych", type=float, default=0.15,
                    help="ile kafelkow bez obiektow zostawic (przyklady negatywne)")
    ap.add_argument("--prog-dedup", type=int, default=5)
    ap.add_argument("--splity-weedsgalore", help="katalog splits z repozytorium autorow")
    ap.add_argument("--taksonomia", help="wlasny plik taxonomy.yaml")
    ap.add_argument("--bez-obrazow", action="store_true",
                    help="zapisz tylko etykiety i manifest (szybki test)")
    ap.add_argument("--tylko-wlasne-do-testu", action="store_true")
    ap.add_argument("--wlasny-podzial", action="store_true",
                    help="ignoruj oficjalne podzialy autorow zbiorow i podziel "
                         "wszystko po grupach (rozlacznie, kosztem porownywalnosci "
                         "z wynikami z publikacji)")
    ap.add_argument("--skala", action="append", default=[],
                    help="ZRODLO=WSPOLCZYNNIK - zmniejsz zdjecia zrodla do skali drona, "
                         "np. weedmaize=0.144 (GSD 0,36 -> 2,5 mm/px)")
    ap.add_argument("--maks-zdjec", action="append", default=[],
                    help="ZRODLO=N - najwyzej N zdjec z tego zrodla, np. weedmaize=2000")
    ap.add_argument("--proporcje", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                    metavar=("TRAIN", "VAL", "TEST"),
                    help="proporcje podzialu; 1 0 0 = wszystko do treningu (zbior pomocniczy)")
    ap.add_argument("--ewis1-z-sorgo", action="store_true",
                    help="EWIS1: wlacz tez sorgo i zdjecia bez nazwy uprawy (domyslnie tylko 'Maize')")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    zrodla = {z: getattr(a, z) for z in ADAPTERY if getattr(a, z)}
    if not zrodla:
        ap.error("podaj co najmniej jeden zbior, np. --weedsgalore sciezka/")

    def _pary(lista, typ):
        out = {}
        for x in lista:
            k, _, v = x.partition("=")
            out[k] = typ(v)
        return out

    info = zbuduj(zrodla, a.output, a.tryb_klas, a.kafelek, a.zakladka,
                  a.udzial_pustych, a.prog_dedup, proporcje=tuple(a.proporcje),
                  skale=_pary(a.skala, float), limity=_pary(a.maks_zdjec, int),
                  opcje_adapterow={"ewis1": {"z_sorgo": a.ewis1_z_sorgo}},
                  splity_weedsgalore=a.splity_weedsgalore, taksonomia=a.taksonomia,
                  kopiuj_obrazy=not a.bez_obrazow,
                  tylko_wlasne_do_testu=a.tylko_wlasne_do_testu,
                  oficjalne_splity=not a.wlasny_podzial, seed=a.seed)

    s = info["statystyki"]
    print()
    print(f"  zdjec/kafelkow       {s['zdjec']}")
    print(f"  obiektow             {s['obiektow']}")
    print(f"  bez obiektow         {s['zdjec_bez_obiektow']}  (przyklady negatywne)")
    print(f"  klasy                {info['klasy']}")
    print(f"  podzial              {s['splity']}")
    print(f"  odrzucone duplikaty  {info['odrzucone_duplikaty']}")

    nz = info["nierozpoznane_etykiety"]
    if nz:
        print(f"\n  UWAGA: {len(nz)} nierozpoznanych etykiet - dopisz do taxonomy.yaml:")
        for nazwa, ile in nz[:10]:
            print(f"    {nazwa}  ({ile}x)")
    else:
        print("  nierozpoznane        brak - wszystko zmapowane")

    kon = info["kontrola_podzialu"]
    if kon["bledy"]:
        print("\n  BLAD: te same grupy w kilku czesciach podzialu!")
        for g, cz in kon["bledy"].items():
            print(f"    {g}: {'/'.join(cz)}")
    if kon["oficjalne"]:
        print("\n  UWAGA: oficjalny podzial autorow rozdziela te grupy:")
        for g, cz in kon["oficjalne"].items():
            print(f"    {g}: {'/'.join(cz)}")
        print("    To jedno pole z kilku dni - sprawdz, czy te same MIEJSCA")
        print("    nie trafiaja do dwoch czesci, albo uzyj --wlasny-podzial.")
    if not kon["bledy"] and not kon["oficjalne"]:
        print("  podzial grup       rozlaczny - bez przeciekow")

    print(f"\n  zbior:    {Path(a.output) / 'dataset'}")
    print(f"  podglad:  {Path(a.output) / 'podglad'}  <- OBEJRZYJ przed treningiem")
    return 0


if __name__ == "__main__":
    sys.exit(main())