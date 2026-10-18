"""Wiersz polecen modulu modele.

    python -m models.cli sprawdz   zbior/dataset/data.yaml
    python -m models.cli rozmiary  zbior/dataset
    python -m models.cli trenuj    --naziemne a/data.yaml --lotnicze b/data.yaml
    python -m models.cli ocen      model.pt zbior/dataset/data.yaml
    python -m models.cli prog      model.pt zbior/dataset/data.yaml
    python -m models.cli onnx      model.pt -o model.onnx
    python -m models.cli kalibracja zbior/dataset/images/train -o kal/
"""

from . import device as _urz  # ustawia MPS fallback przed importem torch

import argparse
import json
import logging
import sys
from pathlib import Path

from .plan import TrainingPlan, sprawdz_dane

log = logging.getLogger("models")


def _wypisz(d):
    print(json.dumps(d, indent=2, ensure_ascii=False))


def cmd_analiza(a):
    """Statystyki zbioru: czesci, dni, rozmiary obiektow, wykresy."""
    from .analysis import analizuj
    r = analizuj(a.dane, a.output)
    print(f"\n  {'':<12} {'obrazow':>8} {'obiektow':>9} {'puste':>6} {'na obraz':>9} {'bok [px]':>9} {'male':>6}")
    for nazwa, b in [("RAZEM", r["razem"])] + list(r["czesci"].items()) + list(r["dni"].items()):
        print(f"  {nazwa:<12} {b['obrazow']:>8} {b['obiektow']:>9} {b['puste']:>6} "
              f"{b['obiekty_na_obraz']['srednia']:>9} {b['bok_px']['mediana']:>9} {b['udzial']['male']:>6.0%}")
    if a.output:
        print(f"\n  raport i wykresy: {a.output}/")
    return 0


def cmd_podzial_dni(a):
    """Zbior z podzialem calymi dniami: jeden dzien w calosci jako test."""
    from .splits import podziel_dniami
    info = podziel_dniami(a.zbior, a.output, a.test, a.udzial_val, a.dodaj_trening or ())
    print(json.dumps(info, indent=2, ensure_ascii=False))
    ds = Path(a.output) / "dataset"
    print(f"\n  trening tylko na glownym:   {ds / 'data.yaml'}")
    if a.dodaj_trening:
        print(f"  trening z dodatkowymi:      {ds / 'data_z_dodatkowymi.yaml'}")
    print(f"  test (dzien {a.test}) i walidacja sa w obu z samego zbioru glownego.")
    return 0


def cmd_srodowisko(a):
    """Co jest zainstalowane i na czym pojdzie trening."""
    info = _urz.opis()
    for k, v in info.items():
        print(f"  {k:<20} {v}")
    if info["wybrane_urzadzenie"] == "cpu":
        print("\n  UWAGA: trening pojdzie na procesorze - bedzie bardzo wolny.")
        print("  Windows z karta NVIDIA: zainstaluj PyTorch z CUDA (patrz README).")
    return 0


def cmd_sprawdz(a):
    """Kontrola danych przed treningiem - najtansza rzecz, jaka mozna zrobic."""
    ok, problemy, info = sprawdz_dane(a.dane)
    print(f"\n  plik: {a.dane}")
    for k, v in info.items():
        print(f"  {k:<8} {v}")
    if problemy:
        print("\n  PROBLEMY:")
        for p in problemy:
            print(f"    - {p}")
        print("\n  Popraw je przed treningiem.")
        return 1
    print("\n  Dane wygladaja poprawnie.")
    return 0


def cmd_rozmiary(a):
    """Rozklad rozmiarow obiektow - czy kafelkowanie jest potrzebne."""
    from .evaluation import rozklad_rozmiarow, wczytaj_etykiety

    ds = Path(a.dataset)
    etykiety = ds / "labels" / a.split
    obrazy = ds / "images" / a.split
    if not etykiety.exists():
        etykiety, obrazy = ds, ds
    nazwy = None
    yml = ds / "data.yaml"
    if yml.exists():
        ok, _, info = sprawdz_dane(yml)
        nazwy = info.get("klasy")

    dane = wczytaj_etykiety(etykiety, obrazy)
    if not dane:
        print(f"  brak etykiet w {etykiety}")
        return 1
    wynik = rozklad_rozmiarow(dane, nazwy)
    _wypisz(wynik)
    u = wynik["udzial_malych"]
    if u and u > 0.5:
        print(f"\n  {u:.0%} obiektow to obiekty male (< 32x32 px).")
        print("  Kafelkowanie zamiast przeskalowywania jest tu konieczne.")
    return 0


def cmd_trenuj(a):
    from .training import ablacja, trenuj

    plan = (TrainingPlan.wczytaj(a.plan) if a.plan else
            TrainingPlan.domyslny(a.naziemne, a.lotnicze, a.wlasne,
                                  a.model_bazowy, a.wejscie))
    plan.wyjscie = a.output
    if a.batch:
        plan.batch = a.batch
    if a.urzadzenie:
        plan.urzadzenie = a.urzadzenie

    print("\n  Plan treningu:")
    for e in plan.etapy:
        print(f"    {e.nazwa:<12} {e.epoki:>4} epok  lr0={e.lr0:<8g} {e.dane}")
    print()

    if a.ablacja:
        _wypisz(ablacja(plan, not a.bez_kontroli))
    else:
        model, historia = trenuj(plan, not a.bez_kontroli)
        print(f"\n  Model: {model}")
        for h in historia:
            print(f"    {h['etap']:<12} {h['czas_min']:>6.1f} min")
    return 0


def cmd_ocen(a):
    from .evaluation import ocen, ocen_male_obiekty, zapisz

    wynik = ocen(a.model, a.dane, a.split, a.wejscie, a.urzadzenie)
    if a.male_obiekty:
        ds = Path(a.dane).parent
        ok, _, info = sprawdz_dane(a.dane)
        wynik["wedlug_rozmiaru"] = ocen_male_obiekty(
            a.model, ds / "images" / a.split, ds / "labels" / a.split,
            info.get("klasy"), a.wejscie, prog=a.prog, urzadzenie=a.urzadzenie)
        wynik["wedlug_rozmiaru"]["prog"] = a.prog
    if a.wedlug_dni:
        from .evaluation import ocen_wedlug_grup, predykcje_splitu
        pred, prawda = predykcje_splitu(a.model, a.dane, a.split, a.wejscie,
                                        min(0.05, a.prog), a.urzadzenie)
        wynik["wedlug_dni"] = ocen_wedlug_grup(pred, prawda, a.prog)
        print(f"\n  dzien          obrazow  trafione  pominiete  falszywe  czulosc  precyzja   (prog {a.prog})")
        for d, w in wynik["wedlug_dni"].items():
            print(f"  {d:<14} {w['obrazow']:>7}  {w['trafione']:>8}  {w['pominiete']:>9}  "
                  f"{w['falszywe']:>8}  {w['czulosc']:>7.3f}  {w['precyzja']:>8.3f}")
    _wypisz(wynik)
    if a.output:
        zapisz(wynik, a.output)
    return 0


def cmd_prog(a):
    from .evaluation import dobierz_prog, zapisz

    wynik = dobierz_prog(a.model, a.dane, a.split, waga_czulosci=a.waga_czulosci,
                         wejscie=a.wejscie, urzadzenie=a.urzadzenie)
    _wypisz(wynik)
    if a.output:
        zapisz(wynik, a.output)
    return 0


def cmd_onnx(a):
    from .export import do_onnx, rozmiary

    p = do_onnx(a.model, a.output, a.wejscie, opset=a.opset)
    _wypisz(rozmiary(p))
    return 0


def cmd_ocen_detektor(a):
    """Ta sama ocena dla .pt (na Macu) i .hef (na urzadzeniu z Hailo)."""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from mission.detector import zbuduj_z_plikow
    from .evaluation import ocen_detektor, zapisz
    # niski prog, zeby policzyc AP50; prog uzytkowy stosuje sie przy liczeniu wyniku.
    # Kilka modeli = ensemble (WBF); --tta = dodatkowo odbicie poziome.
    det = zbuduj_z_plikow(a.model, 0.001, a.wejscie, a.urzadzenie, tta=a.tta)
    w = ocen_detektor(det, a.dane, a.split, a.prog)
    for d in getattr(det, "detektory", [det]) + [getattr(det, "detektor", None)]:
        if hasattr(d, "zamknij"):
            d.zamknij()
    r = w["przy_progu"]
    print(f"\n  {w['detektor']}  ({w['obrazow']} obrazow, czesc {w['split']})")
    print(f"  AP50: {w['AP50']:.3f}   przy progu {a.prog}: czulosc {r['czulosc']:.3f}, "
          f"precyzja {r['precyzja']:.3f}  (trafione {r['trafione']}, pominiete {r['pominiete']}, "
          f"falszywe {r['falszywe']})")
    print(f"\n  {'dzien':<14} {'obrazow':>7} {'czulosc':>8} {'precyzja':>9}")
    for d, g in w["wedlug_dni"].items():
        print(f"  {d:<14} {g['obrazow']:>7} {g['czulosc']:>8.3f} {g['precyzja']:>9.3f}")
    if a.output:
        zapisz(w, a.output)
    return 0


def cmd_nadprobkuj(a):
    """Kafelki wczesnych dni czesciej w treningu."""
    from .season import nadprobkuj
    wagi = {}
    for w in a.waga:
        dzien, _, n = w.partition("=")
        wagi[dzien] = int(n or 1)
    licz = nadprobkuj(a.zbior, wagi, a.output, a.dodaj_trening)
    print(f"\n  {'dzien':<14} {'kafelkow':>9} {'w treningu':>11}")
    for d, c in sorted(licz.items()):
        print(f"  {d:<14} {c['przed']:>9} {c['po']:>11}")
    if a.output:
        print(f"\n  data.yaml: {a.output}")
    return 0


def cmd_wybierz_dni(a):
    from .season import wybierz_dni
    ile = wybierz_dni(a.zbior, a.dni, a.output)
    print("  " + ", ".join(f"{c}: {n}" for c, n in ile.items()) + f"  ->  {a.output}/dataset/data.yaml")
    return 0


def _wypisz_dni(tytul, wynik):
    print(f"\n  {tytul}")
    print(f"  {'dzien':<14} {'prog':>6} {'obrazow':>8} {'trafione':>9} {'pominiete':>10} "
          f"{'falszywe':>9} {'czulosc':>8} {'precyzja':>9}")
    for d, w in wynik.items():
        prog = w["prog"] if isinstance(w["prog"], str) else f"{w['prog']:.2f}"
        print(f"  {d:<14} {prog:>6} {w['obrazow']:>8} {w['trafione']:>9} {w['pominiete']:>10} "
              f"{w['falszywe']:>9} {w['czulosc']:>8.3f} {w['precyzja']:>9.3f}")


def cmd_prog_dni(a):
    """Osobny prog dla kazdego dnia, dobrany na walidacji."""
    from .evaluation import predykcje_splitu, zapisz
    from .season import progi_wedlug_dni
    pred, prawda = predykcje_splitu(a.model, a.dane, a.split, a.wejscie, 0.05, a.urzadzenie)
    w = progi_wedlug_dni(pred, prawda, waga_czulosci=a.waga_czulosci)
    print(f"\n  {'dzien':<14} {'prog':>6} {'F-beta':>7} {'czulosc':>8} {'precyzja':>9}   (czesc {a.split})")
    for d, x in w.items():
        print(f"  {d:<14} {x['prog']:>6.2f} {x['f_beta']:>7.3f} {x['czulosc']:>8.3f} {x['precyzja']:>9.3f}")
    if a.output:
        zapisz(w, a.output)
    return 0


def cmd_ocen_sezon(a):
    """Test wedlug dni: przy jednym progu i (z --progi) przy progu kazdego dnia."""
    import json
    from .evaluation import ocen_wedlug_grup, predykcje_splitu, zapisz
    from .season import ocen_z_progami_dni
    pred, prawda = predykcje_splitu(a.model, a.dane, "test", a.wejscie, 0.05, a.urzadzenie)
    wynik = {"model": str(a.model), "jeden_prog": ocen_wedlug_grup(pred, prawda, a.prog)}
    _wypisz_dni(f"jeden prog {a.prog}", wynik["jeden_prog"])
    if a.progi:
        progi = json.loads(Path(a.progi).read_text(encoding="utf-8"))
        wynik["progi_dni"] = ocen_z_progami_dni(pred, prawda, progi, a.prog)
        _wypisz_dni("prog kazdego dnia (z walidacji)", wynik["progi_dni"])
    if a.output:
        zapisz(wynik, a.output)
    return 0


def cmd_kalibracja(a):
    from .export import przygotuj_kalibracje

    kat, lista = przygotuj_kalibracje(a.obrazy, a.output, a.liczba, rozmiar=a.rozmiar)
    print(f"  obrazy: {kat}\n  lista:  {lista}")
    print("\n  UWAGA: obrazy kalibracyjne musza pochodzic z docelowego pola")
    print("  i docelowej kamery - inaczej zakresy wartosci beda zle dobrane.")
    return 0


def cmd_porownaj(a):
    from .export import porownaj_po_kwantyzacji, zapisz

    wynik = porownaj_po_kwantyzacji(a.fp32, a.int8, a.dane, a.split, a.wejscie,
                                    a.urzadzenie)
    _wypisz(wynik)
    if a.output:
        zapisz(wynik, a.output)
    return 0


def parser():
    ap = argparse.ArgumentParser(prog="models", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def wspolne_ocena(p):
        p.add_argument("--split", default="test", choices=["train", "val", "test"])
        p.add_argument("--wejscie", type=int, default=640)
        p.add_argument("--urzadzenie", help="cpu albo numer GPU")
        p.add_argument("-o", "--output", help="zapisz wynik do pliku JSON")

    p = sub.add_parser("analiza", help="statystyki zbioru: czesci, dni, rozmiary obiektow, wykresy")
    p.add_argument("dane", help="data.yaml")
    p.add_argument("-o", "--output", help="katalog na raport.json, raport.md i wykresy")
    p.set_defaults(func=cmd_analiza)

    p = sub.add_parser("podzial-dni", help="podzial calymi dniami: jeden dzien jako test")
    p.add_argument("zbior", help="katalog dataset zbioru glownego (np. zbior_lotnicze/dataset)")
    p.add_argument("--test", required=True, help="dzien testowy, np. 2023-05-25")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--udzial-val", type=float, default=0.15,
                   help="udzial walidacji w calym zbiorze (z dni treningowych)")
    p.add_argument("--dodaj-trening", nargs="*",
                   help="katalogi dataset innych zbiorow - dokladana ich czesc treningowa")
    p.set_defaults(func=cmd_podzial_dni)

    p = sub.add_parser("srodowisko", help="wersje bibliotek i wybrane urzadzenie (CUDA/MPS/CPU)")
    p.set_defaults(func=cmd_srodowisko)

    p = sub.add_parser("sprawdz", help="kontrola data.yaml przed treningiem")
    p.add_argument("dane")
    p.set_defaults(func=cmd_sprawdz)

    p = sub.add_parser("rozmiary", help="rozklad rozmiarow obiektow w zbiorze")
    p.add_argument("dataset")
    p.add_argument("--split", default="train")
    p.set_defaults(func=cmd_rozmiary)

    p = sub.add_parser("trenuj", help="trening trzyetapowy")
    p.add_argument("--naziemne", help="data.yaml etapu 1")
    p.add_argument("--lotnicze", help="data.yaml etapu 2")
    p.add_argument("--wlasne", help="data.yaml etapu 3")
    p.add_argument("--plan", help="gotowy plan JSON/YAML zamiast powyzszych")
    p.add_argument("--model-bazowy", default="yolov8n.pt")
    p.add_argument("--wejscie", type=int, default=640)
    p.add_argument("--batch", type=int)
    p.add_argument("--urzadzenie")
    p.add_argument("-o", "--output", default="modele_out")
    p.add_argument("--ablacja", action="store_true",
                   help="dodatkowo wytrenuj sam ostatni etap - sprawdzenie, "
                        "czy etapowanie cokolwiek daje")
    p.add_argument("--bez-kontroli", action="store_true")
    p.set_defaults(func=cmd_trenuj)

    p = sub.add_parser("ocen", help="ocena modelu")
    p.add_argument("model")
    p.add_argument("dane")
    p.add_argument("--male-obiekty", action="store_true",
                   help="dodatkowo czulosc osobno dla malych/srednich/duzych")
    p.add_argument("--wedlug-dni", action="store_true",
                   help="dodatkowo czulosc i precyzja osobno dla kazdego dnia lotu")
    p.add_argument("--prog", type=float, default=0.25,
                   help="prog ufnosci dla podzialu wedlug rozmiaru - podaj ten "
                        "wybrany na walidacji, zeby wszystkie liczby byly z jednego punktu")
    wspolne_ocena(p)
    p.set_defaults(func=cmd_ocen)

    p = sub.add_parser("prog", help="dobor progu ufnosci (czulosc > precyzja)")
    p.add_argument("model")
    p.add_argument("dane")
    p.add_argument("--waga-czulosci", type=float, default=3.0)
    wspolne_ocena(p)
    # prog dobiera sie na walidacji - test musi zostac niezalezny
    p.set_defaults(func=cmd_prog, split="val")

    p = sub.add_parser("onnx", help="eksport do ONNX")
    p.add_argument("model")
    p.add_argument("-o", "--output")
    p.add_argument("--wejscie", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    p.set_defaults(func=cmd_onnx)

    p = sub.add_parser("kalibracja", help="zbior kalibracyjny do kwantyzacji INT8")
    p.add_argument("obrazy")
    p.add_argument("-o", "--output", default="kalibracja")
    p.add_argument("--liczba", type=int, default=200)
    p.add_argument("--rozmiar", type=int,
                   help="wpisz obrazy w kwadrat wejscia sieci (np. 640) - dla kompilatora Hailo")
    p.set_defaults(func=cmd_kalibracja)

    p = sub.add_parser("nadprobkuj", help="kafelki wybranych dni czesciej w treningu (sezonowosc)")
    p.add_argument("zbior", help="katalog zbioru (z images/ i labels/)")
    p.add_argument("--waga", action="append", default=[], help="DZIEN=N, np. 2023-05-25=3")
    p.add_argument("--dodaj-trening", action="append", default=[],
                   help="dodatkowy zbior, ktorego images/train wejdzie do treningu (np. Sudars)")
    p.add_argument("-o", "--output", help="data.yaml do treningu")
    p.set_defaults(func=cmd_nadprobkuj)

    p = sub.add_parser("wybierz-dni", help="zbior tylko z wybranych dni, ten sam podzial")
    p.add_argument("zbior")
    p.add_argument("--dni", nargs="+", required=True)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_wybierz_dni)

    p = sub.add_parser("prog-dni", help="prog ufnosci dla kazdego dnia osobno (na walidacji)")
    p.add_argument("model")
    p.add_argument("dane")
    p.add_argument("--split", default="val")
    p.add_argument("--wejscie", type=int, default=1280)
    p.add_argument("--waga-czulosci", type=float, default=3.0)
    p.add_argument("--urzadzenie")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_prog_dni)

    p = sub.add_parser("ocen-sezon", help="test wedlug dni przy jednym progu i przy progu kazdego dnia")
    p.add_argument("model")
    p.add_argument("dane")
    p.add_argument("--wejscie", type=int, default=1280)
    p.add_argument("--prog", type=float, default=0.15)
    p.add_argument("--progi", help="plik z prog-dni")
    p.add_argument("--urzadzenie")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_ocen_sezon)

    p = sub.add_parser("ocen-detektor",
                       help="ocena .pt albo .hef tym samym kodem: AP50, prog, wynik wedlug dni")
    p.add_argument("model", nargs="+", help="model .pt / .onnx / .hef; kilka = ensemble (WBF)")
    p.add_argument("dane", help="data.yaml")
    p.add_argument("--tta", action="store_true", help="dodatkowo detekcja na odbiciu poziomym")
    p.add_argument("--split", default="test")
    p.add_argument("--prog", type=float, default=0.15)
    p.add_argument("--wejscie", type=int, default=1280, help="tylko dla .pt; .hef ma wejscie wbudowane")
    p.add_argument("--urzadzenie")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_ocen_detektor)

    p = sub.add_parser("porownaj", help="FP32 kontra INT8 na tym samym zbiorze")
    p.add_argument("fp32")
    p.add_argument("int8")
    p.add_argument("dane")
    wspolne_ocena(p)
    p.set_defaults(func=cmd_porownaj)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    try:
        return a.func(a)
    except ImportError as e:
        log.error("%s", e)
        return 2
    except Exception as e:
        log.error("%s", e)
        if a.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())