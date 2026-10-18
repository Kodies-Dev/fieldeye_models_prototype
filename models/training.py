"""Trening trzyetapowy. Opakowanie ultralytics - cala logika etapow jest tutaj.

UWAGA: ten modul wymaga `pip install ultralytics`. Bez niego reszta pakietu
(plan, kontrola danych, porownanie modeli) dziala normalnie.
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
    """Jeden etap treningu. Zwraca sciezke do najlepszych wag."""
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
    """Caly plan po kolei. Kazdy etap startuje z wag poprzedniego."""
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


def ablacja(plan, kontroluj=True):
    """Czy etapowanie faktycznie pomaga - porownanie z treningiem jednoetapowym.

    Trenuje osobno sam ostatni etap (od wag COCO) i porownuje z pelnym planem.
    Bez tego nie wiadomo, czy trzy etapy cokolwiek daly - a to jest pytanie,
    ktore padnie przy omawianiu wynikow.
    """
    from .plan import TrainingPlan

    log.info("--- wariant A: pelny plan (%d etapow)", len(plan.etapy))
    model_a, hist_a = trenuj(plan, kontroluj)

    ostatni = plan.etapy[-1]
    plan_b = TrainingPlan(model_bazowy=plan.model_bazowy, batch=plan.batch,
                          cierpliwosc=plan.cierpliwosc, urzadzenie=plan.urzadzenie,
                          wyjscie=str(Path(plan.wyjscie) / "ablacja"), seed=plan.seed,
                          etapy=[ostatni])
    log.info("--- wariant B: tylko ostatni etap, od wag bazowych")
    model_b, hist_b = trenuj(plan_b, kontroluj)

    return {"pelny_plan": str(model_a), "tylko_ostatni_etap": str(model_b),
            "historia_a": hist_a, "historia_b": hist_b}
