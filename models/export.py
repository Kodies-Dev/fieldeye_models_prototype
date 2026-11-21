"""Eksport modelu do ONNX i przygotowanie do kwantyzacji INT8 pod akcelerator.

Sciezka wdrozeniowa:

    .pt (PyTorch, FP32)
      -> ONNX            format posredni, rozumiany przez narzedzia wdrozeniowe
      -> kwantyzacja     FP32 -> INT8, czterokrotnie mniejszy model
      -> kompilacja      pod konkretny uklad

Ten modul obsluguje dwa pierwsze kroki. Kompilacja wymaga narzedzi producenta
ukladu i wykonuje sie poza tym projektem.

W projekcie ukladem jest Hailo-8 (Raspberry Pi 5 + AI HAT+).
Droga modelu koncowego (scripts/final_model.sh, krok 4 -> scripts/HAILO.md):
    do_onnx()               -> hailo/final/fieldeye_final_1280.onnx (wejscie 1280, opset 11)
    przygotuj_kalibracje()  -> hailo/final/kalibracja_1280/ (128 kafelkow treningowych, letterbox)
    scripts/compile_hef.py  -> .hef (na WSL, kompilator Hailo - kwantyzacja na tych zdjeciach)
    scripts/check_hef.sh    -> sprawdzenie .hef na Pi (strata mierzona przez ocen-detektor)

UWAGA o kwantyzacji: potreningowa (PTQ) jest szybka, ale kosztuje kilka punktow
mAP. Jesli strata okaze sie za duza, trzeba przejsc na kwantyzacje swiadoma
treningu (QAT), ktora wymaga powtorzenia treningu z symulacja INT8.
Strate mierzy scripts/check_hef.sh (ten sam ocen-detektor dla .pt i .hef).
"""

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def do_onnx(model, wyjscie=None, wejscie=640, uprosc=True, opset=12, dynamiczne=False):
    """Eksport .pt -> .onnx.

    opset 12 zamiast najnowszego: narzedzia wdrozeniowe pod FPGA czesto
    obsluguja starsze wersje. Jesli kompilator odrzuci model, sprobuj nizszego.
    (Dla Hailo final_model.sh podaje --opset 11.)

    dynamiczne=False (staly rozmiar wejscia) jest wymagane przez wiekszosc
    akceleratorow - potok sprzetowy musi znac rozmiary z gory.

    Argumenty:
        model       - plik .pt, np. "models/weights/fieldeye_final.pt"
        wyjscie     - gdzie zapisac .onnx albo None (obok modelu)
        wejscie     - staly bok wejscia, np. 1280
        uprosc      - True = uproszczenie grafu (onnxsim)
        opset       - wersja zestawu operacji ONNX, np. 11
        dynamiczne  - False = staly rozmiar wejscia

    Zwraca:
        Path do pliku .onnx

    Uzywana w:
        models/cli.py cmd_onnx() - komenda "onnx" w scripts/final_model.sh (krok 4)
        i scripts/przygotuj_folder.sh
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
    (Bez wlasnych danych final_model.sh bierze kafelki TRENINGOWE WeedsGalore i EWIS1 - nigdy testowe.)

    Argumenty:
        katalog_obrazow - folder ze zdjeciami (szukane rekurencyjnie)
        wyjscie         - folder wyniku, np. "hailo/final/kalibracja_1280"
        liczba          - ile losowych zdjec, np. 128
        seed            - ziarno losowania (te same zdjecia przy kazdym uruchomieniu)
        rozmiar         - bok wejscia sieci, np. 1280 -> letterbox (szare pasy 114) jak przy detekcji;
                          None = kopie bez zmian

    Zwraca:
        (folder, lista) - folder zdjec i plik kalibracja.txt (obok folderu) z ich sciezkami

    Uzywana w:
        models/cli.py cmd_kalibracja() - komenda "kalibracja" w scripts/final_model.sh (krok 4)
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


def rozmiary(model):
    """Rozmiar pliku i szacunek po kwantyzacji - do tabeli w dokumentacji.

    Argumenty:
        model - plik modelu (.pt albo .onnx)

    Zwraca:
        dict, np. {"plik": "...onnx", "rozmiar_MB": 12.3, "szacowany_INT8_MB": 3.08, "uwaga": "..."}

    Uzywana w:
        models/cli.py cmd_onnx() - wypis po eksporcie
    """
    p = Path(model)
    mb = p.stat().st_size / 1e6
    return {"plik": str(p), "rozmiar_MB": round(mb, 2),
            "szacowany_INT8_MB": round(mb / 4, 2),
            "uwaga": "szacunek: INT8 to 1 bajt na wage zamiast 4 (FP32)"}

