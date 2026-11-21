"""Wiersz polecen modulu models.

Komendy uzywane przez scripts/final_model.sh (model koncowy):
    python -m models.cli nadprobkuj    zbior_lotnicze/dataset --waga 2023-05-25=3 ... -o data.yaml
    python -m models.cli trenuj        --plan plan_dane_ewis1_bt.json -o modele_dane_ewis1_bt
    python -m models.cli ocen-detektor model.pt zbior/dataset/data.yaml --prog 0.15 --tta -o wynik.json
    python -m models.cli onnx          model.pt -o model.onnx --wejscie 1280 --opset 11
    python -m models.cli kalibracja    obrazy/ -o kal/ --liczba 128 --rozmiar 1280

Pomocnicze:
    python -m models.cli srodowisko                        wersje bibliotek, CUDA / MPS / CPU
    python -m models.cli sprawdz   zbior/dataset/data.yaml kontrola danych przed treningiem

Kazda komenda to funkcja cmd_<nazwa>(a): a = opcje z terminala (argparse),
zwraca kod wyjscia (0 = OK). parser() opisuje opcje, main() wybiera komende.
"""

from . import device as _urz  # ustawia MPS fallback przed importem torch

import argparse
import json
import logging
import sys

from .plan import TrainingPlan, sprawdz_dane

log = logging.getLogger("models")


def _wypisz(d):
    """Wypisuje slownik jako czytelny JSON (wciecia; ensure_ascii=False - polskie znaki zostaja).

    Argumenty:
        d - slownik z wynikiem

    Zwraca:
        nic

    Uzywana w:
        cmd_onnx() (ten plik)
    """
    print(json.dumps(d, indent=2, ensure_ascii=False))


def cmd_srodowisko(a):
    """Komenda "srodowisko": co jest zainstalowane i na czym pojdzie trening.

    Argumenty:
        a - opcje (brak wlasnych)

    Zwraca:
        0; wypisuje wynik device.opis() i ostrzezenie, gdy trening poszedlby na CPU

    Uzywana w:
        parser() -> main() - python -m models.cli srodowisko
    """
    info = _urz.opis()
    for k, v in info.items():
        print(f"  {k:<20} {v}")
    if info["wybrane_urzadzenie"] == "cpu":
        print("\n  UWAGA: trening pojdzie na procesorze - bedzie bardzo wolny.")
        print("  Windows z karta NVIDIA: zainstaluj PyTorch z CUDA (patrz README).")
    return 0


def cmd_sprawdz(a):
    """Komenda "sprawdz": kontrola danych przed treningiem - najtansza rzecz, jaka mozna zrobic.

    Argumenty:
        a - opcje: a.dane (data.yaml)

    Zwraca:
        0 = dane poprawne, 1 = sa problemy (wypisuje je); uzywa plan.sprawdz_dane()

    Uzywana w:
        parser() -> main() - python -m models.cli sprawdz
    """
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


def cmd_trenuj(a):
    """Komenda "trenuj": trening wedlug planu z pliku (--plan).

    Po co: krok 2 w scripts/final_model.sh - trening modelu koncowego z plan_dane_ewis1_bt.json.

    Argumenty:
        a - opcje: a.plan (JSON/YAML), a.output (folder wynikow),
            a.batch, a.urzadzenie (nadpisuja plan)

    Zwraca:
        0; wypisuje plan, a po treningu sciezke model.pt i czas etapow (training.trenuj)

    Uzywana w:
        parser() -> main() - python -m models.cli trenuj --plan plan_dane_ewis1_bt.json -o modele_dane_ewis1_bt
    """
    from .training import trenuj

    plan = TrainingPlan.wczytaj(a.plan)
    plan.wyjscie = a.output
    if a.batch:
        plan.batch = a.batch
    if a.urzadzenie:
        plan.urzadzenie = a.urzadzenie

    print("\n  Plan treningu:")
    for e in plan.etapy:
        print(f"    {e.nazwa:<12} {e.epoki:>4} epok  lr0={e.lr0:<8g} {e.dane}")
    print()

    model, historia = trenuj(plan)
    print(f"\n  Model: {model}")
    for h in historia:
        print(f"    {h['etap']:<12} {h['czas_min']:>6.1f} min")
    return 0


def cmd_onnx(a):
    """Komenda "onnx": eksport .pt -> .onnx i wypis rozmiaru pliku.

    Argumenty:
        a - opcje: a.model (.pt), a.output (.onnx), a.wejscie (np. 1280), a.opset (np. 11)

    Zwraca:
        0; plik .onnx (export.do_onnx) i wypis export.rozmiary()

    Uzywana w:
        parser() -> main() - scripts/final_model.sh krok 4, scripts/przygotuj_folder.sh
    """
    from .export import do_onnx, rozmiary

    p = do_onnx(a.model, a.output, a.wejscie, opset=a.opset)
    _wypisz(rozmiary(p))
    return 0


def cmd_ocen_detektor(a):
    """Ta sama ocena dla .pt (na Macu) i .hef (na urzadzeniu z Hailo).

    Komenda "ocen-detektor": buduje detektor z pliku modelu z niskim progiem (0.001 - do AP50),
    opcjonalnie z TTA, liczy evaluation.ocen_detektor() i wypisuje AP50, czulosc, precyzje i dni.

    Argumenty:
        a - opcje: a.model (.pt / .onnx / .hef), a.dane, a.split, a.prog (np. 0.15),
            a.wejscie (np. 1280, tylko .pt), a.urzadzenie, a.tta, a.output

    Zwraca:
        0; z -o zapisuje JSON, np. wyniki/model_koncowy/ewis1_tta.json

    Uzywana w:
        parser() -> main() - scripts/final_model.sh krok 3 (4 oceny), scripts/check_hef.sh (.hef na Pi)
    """
    from mission.detector import zbuduj_z_plikow
    from .evaluation import ocen_detektor, zapisz
    # niski prog, zeby policzyc AP50; prog uzytkowy stosuje sie przy liczeniu wyniku.
    # --tta = dodatkowo odbicie poziome.
    det = zbuduj_z_plikow(a.model, 0.001, a.wejscie, a.urzadzenie, tta=a.tta)
    w = ocen_detektor(det, a.dane, a.split, a.prog)
    for d in (det, getattr(det, "detektor", None)):          # Hailo: zwolnij akcelerator
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
    """Kafelki wczesnych dni czesciej w treningu.

    Komenda "nadprobkuj": zamienia opcje --waga DZIEN=N na slownik i wola season.nadprobkuj().

    Argumenty:
        a - opcje: a.zbior (dataset WeedsGalore), a.waga (np. ["2023-05-25=3", "2023-05-30=2"]),
            a.dodaj_trening (dodatkowe zbiory treningowe), a.output (nowy data.yaml)

    Zwraca:
        0; wypisuje tabele dzien / kafelkow / w treningu

    Uzywana w:
        parser() -> main() - scripts/final_model.sh krok 2 (przed treningiem)
    """
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


def cmd_kalibracja(a):
    """Komenda "kalibracja": zdjecia do kwantyzacji INT8 (dla kompilatora Hailo).

    Argumenty:
        a - opcje: a.obrazy (folder zdjec), a.output, a.liczba (np. 128), a.rozmiar (np. 1280)

    Zwraca:
        0; folder zdjec i lista kalibracja.txt (export.przygotuj_kalibracje)

    Uzywana w:
        parser() -> main() - scripts/final_model.sh krok 4
    """
    from .export import przygotuj_kalibracje

    kat, lista = przygotuj_kalibracje(a.obrazy, a.output, a.liczba, rozmiar=a.rozmiar)
    print(f"  obrazy: {kat}\n  lista:  {lista}")
    print("\n  UWAGA: obrazy kalibracyjne musza pochodzic z docelowego pola")
    print("  i docelowej kamery - inaczej zakresy wartosci beda zle dobrane.")
    return 0


def parser():
    """Opis wszystkich komend i ich opcji (argparse).

    Po co: jedno miejsce z nazwami opcji i wartosciami domyslnymi, np. ocen-detektor:
           --prog 0.15, --wejscie 1280, --split test. Kazda komenda dostaje func=cmd_<nazwa>.

    Zwraca:
        argparse.ArgumentParser

    Uzywana w:
        main() (ten plik), tests/test_hailo.py, test_inference.py, test_season.py (sprawdzanie opcji)
    """
    ap = argparse.ArgumentParser(prog="models", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("srodowisko", help="wersje bibliotek i wybrane urzadzenie (CUDA/MPS/CPU)")
    p.set_defaults(func=cmd_srodowisko)

    p = sub.add_parser("sprawdz", help="kontrola data.yaml przed treningiem")
    p.add_argument("dane")
    p.set_defaults(func=cmd_sprawdz)

    p = sub.add_parser("nadprobkuj", help="kafelki wybranych dni czesciej w treningu (sezonowosc)")
    p.add_argument("zbior", help="katalog zbioru (z images/ i labels/)")
    p.add_argument("--waga", action="append", default=[], help="DZIEN=N, np. 2023-05-25=3")
    p.add_argument("--dodaj-trening", action="append", default=[],
                   help="dodatkowy zbior, ktorego images/train wejdzie do treningu (np. Sudars)")
    p.add_argument("-o", "--output", help="data.yaml do treningu")
    p.set_defaults(func=cmd_nadprobkuj)

    p = sub.add_parser("trenuj", help="trening wedlug planu z pliku")
    p.add_argument("--plan", required=True, help="plan JSON/YAML, np. plan_dane_ewis1_bt.json")
    p.add_argument("--batch", type=int)
    p.add_argument("--urzadzenie")
    p.add_argument("-o", "--output", default="modele_out")
    p.set_defaults(func=cmd_trenuj)

    p = sub.add_parser("ocen-detektor",
                       help="ocena .pt albo .hef tym samym kodem: AP50, prog, wynik wedlug dni")
    p.add_argument("model", help="model .pt / .onnx / .hef")
    p.add_argument("dane", help="data.yaml")
    p.add_argument("--tta", action="store_true", help="dodatkowo detekcja na odbiciu poziomym")
    p.add_argument("--split", default="test")
    p.add_argument("--prog", type=float, default=0.15)
    p.add_argument("--wejscie", type=int, default=1280, help="tylko dla .pt; .hef ma wejscie wbudowane")
    p.add_argument("--urzadzenie")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_ocen_detektor)

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
    return ap


def main(argv=None):
    """Uruchamia komende z terminala: parsuje opcje, ustawia logi i wola a.func(a).

    Argumenty:
        argv - lista opcji, np. ["ocen-detektor", "m.pt", "d.yaml"]; None = sys.argv

    Zwraca:
        kod wyjscia: wynik komendy (0 = OK), 2 = brak pakietu (ImportError), 1 = inny blad
        (z -v pelny traceback)

    Uzywana w:
        python -m models.cli ... (if __name__ == "__main__")
    """
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