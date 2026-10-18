"""Eksport modelu do ONNX i przygotowanie do kwantyzacji INT8 pod FPGA.

Scieżka wdrozeniowa:

    .pt (PyTorch, FP32)
      -> ONNX            format posredni, rozumiany przez narzedzia wdrozeniowe
      -> kwantyzacja     FP32 -> INT8, czterokrotnie mniejszy model
      -> kompilacja      pod konkretny uklad (Vitis AI dla AMD/Xilinx)

Ten modul obsluguje dwa pierwsze kroki. Kompilacja wymaga narzedzi producenta
ukladu i wykonuje sie poza tym projektem.

UWAGA o kwantyzacji: potreningowa (PTQ) jest szybka, ale kosztuje kilka punktow
mAP. Jesli strata okaze sie za duza, trzeba przejsc na kwantyzacje swiadoma
treningu (QAT), ktora wymaga powtorzenia treningu z symulacja INT8.
Pomiar straty jest tutaj wazniejszy niz sam eksport - patrz porownaj_po_kwantyzacji().
"""

import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def do_onnx(model, wyjscie=None, wejscie=640, uprosc=True, opset=12, dynamiczne=False):
    """Eksport .pt -> .onnx.

    opset 12 zamiast najnowszego: narzedzia wdrozeniowe pod FPGA czesto
    obsluguja starsze wersje. Jesli kompilator odrzuci model, sprobuj nizszego.

    dynamiczne=False (staly rozmiar wejscia) jest wymagane przez wiekszosc
    akceleratorow - potok sprzetowy musi znac rozmiary z gory.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("Do eksportu potrzebny pakiet ultralytics: pip install ultralytics") from e

    m = YOLO(str(model))
    sciezka = m.export(format="onnx", imgsz=wejscie, simplify=uprosc,
                       opset=opset, dynamic=dynamiczne)
    sciezka = Path(sciezka)
    if wyjscie:
        wyjscie = Path(wyjscie)
        wyjscie.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(sciezka), wyjscie)
        sciezka = wyjscie
    log.info("ONNX: %s (%.1f MB)", sciezka, sciezka.stat().st_size / 1e6)
    return sciezka


def przygotuj_kalibracje(katalog_obrazow, wyjscie, liczba=200, seed=0, rozmiar=None):
    """Zbior kalibracyjny do kwantyzacji potreningowej.

    Kwantyzacja musi poznac zakresy wartosci w sieci, zeby dobrac skale
    dla liczb calkowitych. Robi to, przepuszczajac przez model prawdziwe
    obrazy - bez etykiet, wiec wystarcza same zdjecia.

    Dwie zasady:
      - obrazy MUSZA pochodzic z rozkladu docelowego (nasze pole, nasza
        kamera), nie z publicznego zbioru - inaczej zakresy beda zle,
      - 100-500 obrazow w zupelnosci wystarcza; wiecej nie poprawia wyniku.
    """
    import random

    pliki = [p for p in Path(katalog_obrazow).rglob("*")
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not pliki:
        raise FileNotFoundError(f"brak obrazow w {katalog_obrazow}")
    random.Random(seed).shuffle(pliki)
    wybrane = pliki[:liczba]

    wyjscie = Path(wyjscie)
    wyjscie.mkdir(parents=True, exist_ok=True)
    for p in wybrane:
        if rozmiar:
            # obraz wpisany w kwadrat wejscia sieci, tak jak przy uruchomieniu
            # (letterbox) - wtedy kompilator Hailo dostaje dokladnie to, co model
            import cv2
            import numpy as np
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            s_ = min(rozmiar / w, rozmiar / h)
            nw, nh = max(1, round(w * s_)), max(1, round(h * s_))
            plotno = np.full((rozmiar, rozmiar, 3), 114, np.uint8)
            py, px = (rozmiar - nh) // 2, (rozmiar - nw) // 2
            plotno[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(wyjscie / (p.stem + ".jpg")), plotno, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            shutil.copy(p, wyjscie / p.name)

    lista = wyjscie.parent / "kalibracja.txt"
    lista.write_text("\n".join(str(wyjscie / (p.stem + ".jpg" if rozmiar else p.name)) for p in wybrane) + "\n",
                     encoding="utf-8")
    log.info("Zbior kalibracyjny: %d obrazow -> %s", len(wybrane), wyjscie)
    return wyjscie, lista


def porownaj_po_kwantyzacji(model_fp32, model_int8, dane, split="test",
                            wejscie=640, urzadzenie=None):
    """Ile mAP kosztuje kwantyzacja - mierzone na tym samym zbiorze testowym.

    To jest liczba, ktora trzeba podac w dokumentacji projektu. Bez niej
    nie wiadomo, czy model na FPGA dziala tak samo dobrze jak na komputerze.
    """
    from .evaluation import ocen

    a = ocen(model_fp32, dane, split, wejscie, urzadzenie)
    b = ocen(model_int8, dane, split, wejscie, urzadzenie)
    return {
        "fp32": a,
        "int8": b,
        "strata_mAP50": round(a["mAP50"] - b["mAP50"], 4),
        "strata_mAP50_95": round(a["mAP50_95"] - b["mAP50_95"], 4),
        "strata_czulosci": round(a["czulosc"] - b["czulosc"], 4),
        "wniosek": _wniosek(a["mAP50"] - b["mAP50"]),
    }


def _wniosek(strata):
    if strata < 0.01:
        return "strata pomijalna - kwantyzacja potreningowa wystarczy"
    if strata < 0.03:
        return "strata akceptowalna - warto sprawdzic inny zbior kalibracyjny"
    return ("strata znaczaca - rozwaz kwantyzacje swiadoma treningu (QAT) "
            "albo wiekszy/lepiej dobrany zbior kalibracyjny")


def rozmiary(model):
    """Rozmiar pliku i szacunek po kwantyzacji - do tabeli w dokumentacji."""
    p = Path(model)
    mb = p.stat().st_size / 1e6
    return {"plik": str(p), "rozmiar_MB": round(mb, 2),
            "szacowany_INT8_MB": round(mb / 4, 2),
            "uwaga": "szacunek: INT8 to 1 bajt na wage zamiast 4 (FP32)"}


def zapisz(wynik, sciezka):
    p = Path(sciezka)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(wynik, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
