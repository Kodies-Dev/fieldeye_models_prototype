"""fieldeye-detect: zdjecia z katalogu -> model chwastow -> wynik w terminalu.

Obserwuje katalog i kazde nowe zdjecie przepuszcza przez model: zdjecie jest
ciete na kafelki ok. 1,5 m pola (jak na dronie), model wykrywa chwasty na kazdym
kafelku, a w terminalu pojawia sie linijka z wynikiem. Ten sam program dziala
z modelem .pt (komputer z ultralytics), .onnx i .hef (Raspberry Pi z Hailo-8).

  fieldeye-detect ZDJECIA WYNIKI MODEL              # MODEL: sciezka z nazwa pliku (.pt / .onnx / .hef)
  fieldeye-detect ZDJECIA WYNIKI                    # model obok binarki albo model koncowy z projektu
  fieldeye-detect ZDJECIA WYNIKI MODEL --raz        # przetwarza to, co jest, i konczy
  fieldeye-detect ZDJECIA WYNIKI --jsonl            # dodatkowo WYNIKI/wyniki.jsonl z ramkami w liczbach
  fieldeye-detect ZDJECIA                           # bez zapisu - tylko wynik w terminalu
  fieldeye-detect ZDJECIA --json                    # jedna linia JSON na zdjecie (dla innych programow)

Folder WYNIKI powstaje sam; zdjecia z ramkami maja te same nazwy co oryginaly.

Telemetria GPS: gdy obok zdjecia lezy IMG_0001.geojson z pozycja drona (opis formatu:
mission/spray_map.py), kazdy chwast dostaje wspolrzedne, a w WYNIKI powstaje mapa
oprysku dla rolnika: oprysk.geojson, oprysk.csv, chwasty.geojson, trasa.geojson.
Kafelek jest wtedy dobierany z telemetrii tak, zeby obejmowal ok. 1,5 m pola.

Bez instalacji pakietu: python -m mission.detect_cli zdjecia/
W binarce dodatkowo: fieldeye-detect dron FOLDER ... (symulowany dron z GPS)
                     fieldeye-detect porownaj OPRYSK PRAWDA (ocena namiarow)

Petla main(), dla kazdego nowego zdjecia (FIFO):
    kandydaci() -> gotowy() (zapisane do konca) -> telemetria .geojson (spray_map)
    -> dobor kafelka do 1,5 m -> przetworz() (detekcja) -> mapa oprysku -> linia() w terminalu
    -> zdjecie z ramkami w WYNIKI (zapis atomowy)
Binarka: scripts/fieldeye_detect_main.py, build_binary.sh; opis: docs/fieldeye-detect.md.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROZSZERZENIA = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MODELE_OBOK_BINARKI = ("fieldeye_final_1280_h8.hef", "fieldeye_final.onnx", "fieldeye_final_1280.onnx",
                       "fieldeye_final.pt")


def domyslny_model():
    """Model, gdy nie podano --model.

    W binarce (PyInstaller): pierwszy z MODELE_OBOK_BINARKI lezacy obok pliku
    programu, a potem model wbudowany w binarke. Ze zrodel: model koncowy z repozytorium.

    Zwraca:
        Path, np. Path(".../models/weights/fieldeye_final.pt")

    Uzywana w:
        main() (ten plik)
    """
    if getattr(sys, "frozen", False):
        for katalog in (Path(sys.executable).resolve().parent, Path(getattr(sys, "_MEIPASS", "."))):
            for n in MODELE_OBOK_BINARKI:
                if (katalog / n).exists():
                    return katalog / n
        return Path(sys.executable).resolve().parent / MODELE_OBOK_BINARKI[0]
    return Path(__file__).resolve().parents[1] / "models" / "weights" / "fieldeye_final.pt"
ROZOWY = (255, 0, 255)
KAFELEK_DOMYSLNY = 950           # ok. 1,5 m pola dla Arducam 16 MP z 5 m
KAFELEK_M = 1.5                  # tyle pola obejmowaly kafelki przy treningu


def kandydaci(katalog, znane):
    """Pliki zdjec w katalogu, jeszcze nieprzetworzone, w kolejnosci FIFO.

    Kolejnosc zapisu (czas modyfikacji pliku), przy rownym czasie - nazwa. Dzieki
    temu FIFO nie zalezy od tego, jak kamera nazywa pliki. Pomijane sa pliki
    ukryte i tymczasowe (np. '.img.jpg', '~img.jpg', 'img.jpg.part'), ktore
    zapisuja niektore programy przed zmiana nazwy na docelowa.

    Argumenty:
        katalog - folder zdjec
        znane   - zbior nazw juz przetworzonych

    Zwraca:
        lista Path, od najstarszego

    Uzywana w:
        main() (ten plik)
    """
    out = []
    for p in Path(katalog).iterdir():
        if (p.suffix.lower() in ROZSZERZENIA and p.name not in znane
                and not p.name.startswith((".", "~"))):
            try:
                st = p.stat()
            except OSError:                               # plik zniknal w miedzyczasie
                continue
            if p.is_file():
                out.append((st.st_mtime_ns, p.name, p))
    return [p for _, _, p in sorted(out)]


def kompletny(p):
    """Czy plik zdjecia jest zapisany do konca (JPEG: znacznik FF D9, PNG: blok IEND).

    Argumenty:
        p - Path zdjecia

    Zwraca:
        bool (inne formaty zawsze True)

    Uzywana w:
        gotowy() (ten plik)
    """
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 64))
            koniec = f.read()
    except OSError:
        return False
    s = p.suffix.lower()
    if s in (".jpg", ".jpeg"):
        return koniec.rstrip(b"\x00").endswith(b"\xff\xd9")
    if s == ".png":
        return b"IEND" in koniec[-12:]
    return True


def gotowy(p, stan, min_stabilnosc=1.0, teraz=None):
    """Plik jest gotowy, gdy jego rozmiar nie zmienia sie od min_stabilnosc sekund
    i jest kompletny (kompletny()).

    Chroni przed czytaniem zdjecia, ktore kamera albo kopiowanie dopiero zapisuje -
    takze wtedy, gdy zapis robi przerwy. stan: {nazwa: (rozmiar, od_kiedy)}.

    Argumenty:
        p              - Path zdjecia
        stan           - slownik zapamietanych rozmiarow (zmieniany w miejscu)
        min_stabilnosc - ile sekund rozmiar musi stac, np. 1.0
        teraz          - czas (do testow) albo None = time.time()

    Zwraca:
        bool

    Uzywana w:
        main() (ten plik)
    """
    teraz = time.time() if teraz is None else teraz
    try:
        r = p.stat().st_size
    except OSError:
        return False
    poprzedni = stan.get(p.name)
    if poprzedni is None or poprzedni[0] != r:
        stan[p.name] = (r, teraz)
        return False
    return r > 0 and teraz - poprzedni[1] >= min_stabilnosc and kompletny(p)


def zbuduj(a):
    """Detektor z argumentow: model (+ TTA) owiniety w kafelkowanie.

    Argumenty:
        a - opcje: a.model, a.prog, a.wejscie, a.urzadzenie, a.tta, a.kafelek, a.zakladka, a.atrapa

    Zwraca:
        TiledDetector(...) albo sam detektor przy --kafelek 0

    Uzywana w:
        main() (ten plik)
    """
    from .detector import DummyDetector, TiledDetector, zbuduj_z_plikow
    if a.atrapa:
        det = DummyDetector()
    else:
        # z TTA nizszy prog na pojedynczym przebiegu: laczenie i tak przefiltruje
        det = zbuduj_z_plikow(a.model, prog=min(a.prog, 0.05) if a.tta else a.prog,
                              wejscie=a.wejscie, urzadzenie=a.urzadzenie, tta=a.tta)
    if a.kafelek == 0:
        return det
    return TiledDetector(det, kafelek=a.kafelek or KAFELEK_DOMYSLNY, zakladka=a.zakladka)


def zamknij(det):
    """Zamyka urzadzenie Hailo, jesli jest gdzies w srodku detektora.

    Argumenty:
        det - detektor (moze byc owiniety: Tiled -> TTA -> Hailo)

    Uzywana w:
        main() - na koncu (finally)
    """
    while det is not None:
        if hasattr(det, "zamknij"):
            det.zamknij()
            return
        det = getattr(det, "detektor", None)


def przetworz(p, det, prog):
    """Jedno zdjecie: wczytanie i detekcja.

    Argumenty:
        p    - Path zdjecia
        det  - detektor z zbuduj()
        prog - prog pewnosci, np. 0.15

    Zwraca:
        (img, r) - obraz BGR i slownik: plik, czas, szer, wys, chwasty (liczba), sekundy,
        wykrycia [[x0, y0, x1, y1, pewnosc], ...] od najpewniejszego

    Uzywana w:
        main() (ten plik)
    """
    import cv2
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError("nie da sie wczytac zdjecia")
    t = time.time()
    ramki = sorted((d for d in det.wykryj(img) if d.pewnosc >= prog), key=lambda d: -d.pewnosc)
    h, w = img.shape[:2]
    return img, {
        "plik": p.name, "czas": datetime.now().isoformat(timespec="seconds"),
        "szer": w, "wys": h, "chwasty": len(ramki), "sekundy": round(time.time() - t, 3),
        "wykrycia": [[round(d.x0, 1), round(d.y0, 1), round(d.x1, 1), round(d.y1, 1), round(d.pewnosc, 3)]
                     for d in ramki]}


def linia(r, szczegoly=0):
    """Jedna linia wyniku do terminala, np. "[14:29:01] IMG_0001.jpg  4624x3472  chwasty:   12   max 0.81  0.84 s".

    Argumenty:
        r         - slownik z przetworz()
        szczegoly - ile najpewniejszych wykryc dopisac pod spodem

    Zwraca:
        str

    Uzywana w:
        main() (ten plik)
    """
    maks = f"max {r['wykrycia'][0][4]:.2f}" if r["wykrycia"] else ""
    s = (f"[{r['czas'][11:]}] {r['plik']:<28} {r['szer']}x{r['wys']:<6} "
         f"chwasty: {r['chwasty']:>4}   {maks:<9} {r['sekundy']:.2f} s")
    for i, (x0, y0, x1, y1, pw) in enumerate(r["wykrycia"][:szczegoly], 1):
        s += f"\n      #{i:<3} x {x0:.0f}-{x1:.0f}  y {y0:.0f}-{y1:.0f}  pewnosc {pw:.2f}"
    return s


def main(argv=None):
    """Caly program fieldeye-detect: opcje -> detektor -> petla po nowych zdjeciach -> podsumowanie.

    Argumenty:
        argv - lista opcji albo None (sys.argv); opisy opcji w --help

    Zwraca:
        0 = OK, 2 = zle opcje (brak modelu, zly format, folder wynikow = folder zdjec, ...)

    Uzywana w:
        python -m mission.detect_cli, scripts/fieldeye_detect_main.py (binarka), fieldeye-detect.sh
    """
    ap = argparse.ArgumentParser(prog="fieldeye-detect", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("katalog", help="katalog ze zdjeciami (np. ten, do ktorego zapisuje kamera)")
    ap.add_argument("wyniki", nargs="?",
                    help="folder na zdjecia z ramkami (powstaje sam); bez niego tylko wynik w terminalu")
    ap.add_argument("model_poz", nargs="?", metavar="model",
                    help="sciezka z nazwa pliku modelu (.pt / .onnx / .hef); domyslnie model obok binarki "
                         "albo model koncowy z projektu")
    ap.add_argument("--model", help="to samo co trzeci argument (np. gdy nie zapisujesz wynikow)")
    ap.add_argument("--prog", type=float, default=0.15, help="prog pewnosci (domyslnie 0.15)")
    ap.add_argument("--wejscie", type=int, default=1280, help="wejscie sieci dla .pt/.onnx (dla .hef wbudowane)")
    ap.add_argument("--kafelek", type=int, default=None,
                    help="bok kafelka w px; domyslnie z telemetrii (1,5 m pola), a bez niej 950 "
                         "(1,5 m dla Arducam 16 MP z 5 m); 0 = bez ciecia")
    ap.add_argument("--zakladka", type=float, default=0.15)
    ap.add_argument("--tta", action="store_true", help="dodatkowo odbicie lustrzane (ok. 2x wolniej, mniej falszywych)")
    ap.add_argument("--urzadzenie", help="cpu, mps, 0 (CUDA); domyslnie automatycznie")
    ap.add_argument("--raz", action="store_true", help="przetworz zdjecia, ktore sa w katalogu, i zakoncz")
    ap.add_argument("--tylko-nowe", action="store_true", help="pomin zdjecia, ktore byly w katalogu przy starcie")
    ap.add_argument("--interwal", type=float, default=0.5, help="co ile sekund sprawdzac katalog")
    ap.add_argument("--stabilnosc", type=float, default=1.0,
                    help="ile sekund rozmiar pliku musi sie nie zmieniac, zanim zostanie wczytany")
    ap.add_argument("--timeout", type=float, help="zakoncz po tylu sekundach bez nowych zdjec")
    ap.add_argument("--jsonl", action="store_true", help="zapisz tez WYNIKI/wyniki.jsonl (ramki w liczbach)")
    ap.add_argument("--szczegoly", type=int, default=0, metavar="N", help="wypisz N najpewniejszych wykryc")
    ap.add_argument("--json", action="store_true", help="na wyjsciu tylko JSON, jedna linia na zdjecie")
    ap.add_argument("--bez-telemetrii", action="store_true", help="nie szukaj plikow .geojson obok zdjec")
    ap.add_argument("--czekaj-telemetrie", type=float, default=3.0,
                    help="ile sekund czekac na telemetrie gotowego zdjecia, zanim pojdzie bez GPS")
    ap.add_argument("--kurs-domyslny", type=float, default=None,
                    help="kurs [st.] przyjmowany, gdy telemetria go nie podaje (np. 0 przy testach w zawisie); "
                         "domyslnie takie zdjecia ida bez GPS")
    ap.add_argument("--maks-przechyl", type=float, default=5.0,
                    help="zdjecia przechylone bardziej niz tyle stopni (roll/pitch z telemetrii) nie trafiaja "
                         "na mape oprysku - obliczenia zakladaja kamere pionowo w dol")
    ap.add_argument("--promien-scalania", type=float, default=0.10,
                    help="wykrycia blizej niz tyle metrow to ten sam chwast (jeden punkt oprysku)")
    ap.add_argument("--atrapa", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.model_poz and a.model:
        print("podaj model raz: jako trzeci argument albo --model", file=sys.stderr)
        return 2
    a.model = str(Path(a.model_poz or a.model or domyslny_model()).expanduser())
    katalog = Path(a.katalog)
    if not katalog.is_dir():
        if a.raz:
            print(f"brak katalogu: {katalog}", file=sys.stderr)
            return 2
        katalog.mkdir(parents=True, exist_ok=True)   # obserwacja: kamera zacznie tu zapisywac
    if not a.atrapa:
        if not Path(a.model).is_file():
            print(f"brak modelu: {a.model}", file=sys.stderr)
            return 2
        if Path(a.model).suffix.lower() not in (".pt", ".onnx", ".hef"):
            print(f"nieobslugiwany format modelu: {Path(a.model).name} (obslugiwane: .pt, .onnx, .hef)",
                  file=sys.stderr)
            return 2
    wyniki = Path(a.wyniki) if a.wyniki else None
    if wyniki and wyniki.resolve() == katalog.resolve():
        print("folder wynikow musi byc inny niz folder zdjec", file=sys.stderr)
        return 2
    if a.jsonl and not wyniki:
        print("--jsonl wymaga folderu WYNIKI", file=sys.stderr)
        return 2
    # przy --json na stdout idzie tylko JSON, a komunikaty na stderr
    info = lambda t: print(t, file=sys.stderr if a.json else sys.stdout, flush=True)  # noqa: E731

    det = zbuduj(a)
    plik_wynikow = None
    if wyniki:
        wyniki.mkdir(parents=True, exist_ok=True)
        if a.jsonl:
            plik_wynikow = open(wyniki / "wyniki.jsonl", "a", encoding="utf-8")
    info(f"FIELDEYE detekcja | model: {Path(a.model).resolve() if not a.atrapa else 'atrapa'} | prog {a.prog} | "
         f"kafelek {'brak' if a.kafelek == 0 else (a.kafelek or 'auto')} px | TTA: {'tak' if a.tta else 'nie'}")
    info(f"{'Przetwarzam' if a.raz else 'Obserwuje'}: {katalog}" + ("" if a.raz else "  (Ctrl+C konczy)"))
    if wyniki:
        info(f"Wyniki: {wyniki}")

    znane = {p.name for p in kandydaci(katalog, set())} if a.tylko_nowe and not a.raz else set()
    rozmiary, n, suma_chwastow, suma_czasu = {}, 0, 0, 0.0
    mapa, czeka_na_tele, bez_gps, odrzucone_przechyl = None, {}, 0, 0
    if not a.bez_telemetrii:
        from .spray_map import MapaOprysku
        mapa = MapaOprysku(a.promien_scalania)
    ostatnio = time.time()
    try:
        while True:
            oczekujace = kandydaci(katalog, znane)
            for p in oczekujace:
                if not a.raz and not gotowy(p, rozmiary, a.stabilnosc):
                    continue                              # jeszcze sie zapisuje - nastepnym razem
                tele = None
                if mapa is not None:
                    from .spray_map import czytaj_telemetrie, plik_telemetrii
                    pt = plik_telemetrii(p)
                    if pt is None and not a.raz:
                        od = czeka_na_tele.setdefault(p.name, time.time())
                        if time.time() - od < a.czekaj_telemetrie:
                            continue                      # zdjecie jest, telemetria zaraz dojdzie
                    if pt is not None:
                        from .spray_map import blad_telemetrii
                        tele = czytaj_telemetrie(pt, p.name)
                        if tele is not None and tele["kurs"] is None and a.kurs_domyslny is not None:
                            tele["kurs"], tele["zrodlo_kursu"] = a.kurs_domyslny, "domyslny"
                        powod = "nie da sie odczytac pliku" if tele is None else blad_telemetrii(tele)
                        if powod:
                            info(f"[{datetime.now():%H:%M:%S}] {p.name:<28} UWAGA: telemetria {pt.name}: {powod}")
                            tele = None
                znane.add(p.name)
                gsd = None
                if tele is not None:
                    from .spray_map import gsd_m
                    from stitchlive.camera import ARDUCAM_64MP_16MP
                    import cv2
                    szer = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_8)
                    szer = szer.shape[1] * 8 if szer is not None else None
                    gsd = gsd_m(tele, szer, ARDUCAM_64MP_16MP) if szer else None
                    if gsd and a.kafelek is None and hasattr(det, "kafelek"):
                        det.kafelek = max(64, round(KAFELEK_M / gsd))      # 1,5 m pola przy tej wysokosci
                try:
                    img, r = przetworz(p, det, a.prog)
                except Exception as e:                    # uszkodzony plik nie zatrzymuje obserwacji
                    info(f"[{datetime.now():%H:%M:%S}] {p.name:<28} POMINIETE: {e}")
                    continue
                dopisek = ""
                from .spray_map import przechyl_st
                k = przechyl_st(tele) if tele is not None else None
                if tele is not None and gsd and k is not None and k > a.maks_przechyl:
                    odrzucone_przechyl += 1
                    dopisek = f"   PRZECHYL {k:.1f} st. > {a.maks_przechyl:g} - poza mapa oprysku"
                elif tele is not None and gsd:
                    nowe = mapa.dodaj_zdjecie(tele, p.name, r["szer"], r["wys"], r["wykrycia"], gsd)
                    r["gps"] = {"lat": tele["lat"], "lon": tele["lon"], "kurs_st": tele["kurs"],
                                "gsd_mm": round(gsd * 1000, 3)}
                    r["wykrycia_geo"] = [[round(w[0], 8), round(w[1], 8)] for w in mapa.wykrycia[-r["chwasty"]:]] \
                        if r["chwasty"] else []
                    dopisek = f"   GPS {tele['lat']:.6f}, {tele['lon']:.6f}   oprysk: {len(mapa.punkty)} pkt (+{nowe})"
                    if wyniki:
                        mapa.zapisz(wyniki)
                elif mapa is not None:
                    bez_gps += 1
                    dopisek = "   bez GPS"
                n += 1
                suma_chwastow += r["chwasty"]
                suma_czasu += r["sekundy"]
                ostatnio = time.time()
                if a.json:
                    print(json.dumps(r, ensure_ascii=False), flush=True)
                else:
                    czeka = len(kandydaci(katalog, znane))
                    info(linia(r, a.szczegoly) + dopisek + (f"   (w kolejce: {czeka})" if czeka else ""))
                if plik_wynikow:
                    plik_wynikow.write(json.dumps(r, ensure_ascii=False) + "\n")
                    plik_wynikow.flush()
                if wyniki:
                    import cv2
                    for x0, y0, x1, y1, _ in r["wykrycia"]:
                        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), ROZOWY, 3)
                    cel = wyniki / p.name
                    tmp = cel.with_name(f".{p.name}")                # zapis atomowy: najpierw ukryty plik
                    if cv2.imwrite(str(tmp), img):
                        tmp.replace(cel)
            if a.raz:
                break
            if a.timeout and time.time() - ostatnio > a.timeout:
                info(f"brak nowych zdjec od {a.timeout:.0f} s - koniec")
                break
            time.sleep(a.interwal)
    except KeyboardInterrupt:
        info("")
    finally:
        zamknij(det)
        if plik_wynikow:
            plik_wynikow.close()
    info(f"Podsumowanie: {n} zdjec, {suma_chwastow} chwastow"
         + (f", srednio {suma_czasu / n:.2f} s na zdjecie" if n else ""))
    if mapa is not None and mapa.punkty:
        info(f"Punkty oprysku: {len(mapa.punkty)}"
             + (f" -> {wyniki / 'oprysk.geojson'}, {wyniki / 'oprysk.csv'}" if wyniki else " (podaj WYNIKI, zeby zapisac)")
             + (f"; zdjec bez GPS: {bez_gps}" if bez_gps else "")
             + (f"; przechylonych ponad {a.maks_przechyl:g} st.: {odrzucone_przechyl}" if odrzucone_przechyl else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
