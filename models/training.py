"""Trening etap po etapie wedlug planu. Opakowanie ultralytics - cala logika etapow jest tutaj.

UWAGA: ten modul wymaga `pip install ultralytics`. Bez niego reszta pakietu
(plan, kontrola danych) dziala normalnie.

Model koncowy: jeden etap z plan_dane_ewis1_bt.json (scripts/final_model.sh, krok 2):
    python -m models.cli trenuj --plan plan_dane_ewis1_bt.json -o modele_dane_ewis1_bt

Wynik w folderze -o:
    plan.json             - plan, z ktorym uczono
    historia.json         - etapy: wagi, czas, parametry
    trening/<etap>/       - wszystko od ultralytics: wykresy, results.csv, weights/best.pt i last.pt
    model.pt              - kopia best.pt ostatniego etapu = model koncowy
                            (final_model.sh kopiuje go do models/weights/fieldeye_final.pt)
"""

import json
import logging
import shutil
import time
from pathlib import Path

from .plan import sprawdz_dane
from .device import wybierz

log = logging.getLogger(__name__)


def _yolo():
    """Importuje klase YOLO z ultralytics; bez pakietu - czytelny blad z komenda instalacji.

    Po co: ultralytics jest ciezki i potrzebny tylko do treningu - reszta pakietu dziala bez niego.

    Zwraca:
        klasa ultralytics.YOLO

    Uzywana w:
        trenuj_etap() (ten plik)
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "Do treningu potrzebny jest pakiet ultralytics:\n"
            "    pip install ultralytics\n"
            "Uwaga na licencje AGPL-3.0 - przy wdrozeniu komercyjnym wymaga "
            "udostepnienia kodu albo wykupienia licencji komercyjnej."
        ) from e
    return YOLO


def trenuj_etap(etap, wagi_startowe, plan, kontroluj=True):
    """Jeden etap treningu. Zwraca sciezke do najlepszych wag.

    Po co: kontrola danych, wywolanie model.train() ultralytics z parametrami z planu
           i znalezienie pliku best.pt (wagi z najlepszej epoki na walidacji).

    Argumenty:
        etap          - Stage (dane, epoki, lr0, wejscie, ...)
        wagi_startowe - od czego zaczac, np. "yolov8n.pt" albo best.pt poprzedniego etapu
        plan          - TrainingPlan (batch, cierpliwosc, seed, optymalizator, urzadzenie, wyjscie)
        kontroluj     - True = najpierw sprawdz_dane(); blad danych = koniec bez treningu

    Zwraca:
        (best, czas) - Path do weights/best.pt i czas treningu w sekundach,
        np. (Path(".../modele_dane_ewis1_bt/trening/dane-ewis1-bt/weights/best.pt"), 14400.0)

    Uzywana w:
        trenuj() (ten plik)
    """
    if kontroluj:
        ok, problemy, info = sprawdz_dane(etap.dane)
        for p in problemy:
            log.warning("dane %s: %s", etap.nazwa, p)
        if not ok:
            raise SystemExit(f"etap {etap.nazwa}: popraw dane przed treningiem")
        log.info("dane %s: %s", etap.nazwa, info)

    YOLO = _yolo()
    log.info("=== etap %s: %d epok, lr0=%g, wejscie=%d, start z %s",
             etap.nazwa, etap.epoki, etap.lr0, etap.wejscie, wagi_startowe)

    model = YOLO(str(wagi_startowe))
    t0 = time.time()
    model.train(
        data=str(etap.dane),
        epochs=etap.epoki,
        imgsz=etap.wejscie,
        lr0=etap.lr0,
        # bez jawnego optymalizatora ultralytics ("auto") IGNORUJE lr0 i dobiera
        # tempo uczenia sam - a plan etapowy opiera sie na malejacym lr0
        optimizer=plan.optymalizator,
        batch=plan.batch,
        patience=plan.cierpliwosc,
        device=wybierz(plan.urzadzenie),   # CUDA -> MPS (Mac) -> CPU
        seed=plan.seed,
        freeze=etap.zamrozone_warstwy or None,
        # sciezka bezwzgledna: nowsze ultralytics doklejaja runs/detect/ przed
        # sciezke wzgledna, przez co wagi trafialy w inne miejsce
        project=str((Path(plan.wyjscie) / "trening").resolve()),
        name=etap.nazwa,
        exist_ok=True,
        verbose=True,
        **(etap.augmentacje or {}),
    )
    czas = time.time() - t0
    # faktyczny katalog zapisu bierzemy od samego ultralytics, zamiast go zgadywac
    katalog = getattr(getattr(model, "trainer", None), "save_dir", None)
    if katalog is None:
        katalog = (Path(plan.wyjscie) / "trening" / etap.nazwa).resolve()
    best = Path(katalog) / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"etap {etap.nazwa}: brak pliku wag {best}")
    log.info("etap %s zakonczony w %.1f min -> %s", etap.nazwa, czas / 60, best)
    return best, czas


def trenuj(plan, kontroluj=True):
    """Caly plan po kolei. Kazdy etap startuje z wag poprzedniego.

    Po co: glowna funkcja treningu - na koncu kopiuje najlepsze wagi ostatniego etapu
           do <wyjscie>/model.pt (to jest plik modelu koncowego).

    Argumenty:
        plan      - TrainingPlan (plan.wyjscie = folder wynikow, np. "modele_dane_ewis1_bt")
        kontroluj - True = kontrola danych przed kazdym etapem

    Zwraca:
        (model_pt, historia) - Path(".../model.pt") i lista slownikow etapow
        (etap, wagi, czas_min + pola etapu); zapisuje tez plan.json i historia.json

    Uzywana w:
        models/cli.py cmd_trenuj() - komenda "trenuj" w scripts/final_model.sh, krok 2
    """
    wyjscie = Path(plan.wyjscie)
    wyjscie.mkdir(parents=True, exist_ok=True)
    plan.zapisz(wyjscie / "plan.json")

    wagi = plan.model_bazowy
    historia = []
    for etap in plan.etapy:
        wagi, czas = trenuj_etap(etap, wagi, plan, kontroluj)
        historia.append({"etap": etap.nazwa, "wagi": str(wagi),
                         "czas_min": round(czas / 60, 1), **etap.slownik()})
        (wyjscie / "historia.json").write_text(
            json.dumps(historia, indent=2, ensure_ascii=False), encoding="utf-8")

    koncowe = wyjscie / "model.pt"
    shutil.copy(wagi, koncowe)
    log.info("Gotowy model: %s", koncowe)
    return koncowe, historia


