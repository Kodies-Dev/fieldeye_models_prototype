"""Kompilacja YOLOv8n FIELDEYE do HEF dla Hailo-8, bezposrednio przez Dataflow Compiler.

Uruchamiac w WSL (Ubuntu 22.04) w srodowisku z Hailo DFC 3.33.0 i Model Zoo 2.17.0:

    source ~/hailo_venv/bin/activate
    python compile_hef.py model_1280.onnx kalibracja_1280 -o fieldeye_final_1280_h8.hef --rozmiar 1280

Dlaczego nie samo `hailomz compile`:
  - konfiguracja NMS z Model Zoo ma rozmiar obrazu 640 - przy 1280 ramki wyszlyby w zlej skali,
  - Model Zoo odrzuca na ukladzie wykrycia ponizej pewnosci 0,2, a nasz prog to 0,15
    (przy TTA 0,05) - czesc chwastow ginelaby przed naszym kodem,
  - limit 100 ramek na klase to za malo przy gesto rosnacych chwastach.

Skrypt bierze z zainstalowanego Model Zoo skrypt modelu (normalizacja, aktywacje wyjsc)
i konfiguracje NMS dla yolov8n, poprawia rozmiar, liczbe klas, prog i limit ramek,
a potem robi parse -> optimize -> compile. Plik .har z kazdego etapu zostaje obok.

Skad pliki wejsciowe: scripts/final_model.sh (krok 4) robi hailo/hailo_final.zip
z fieldeye_final_1280.onnx, kalibracja_1280/ (128 kafelkow treningowych 1280x1280)
i tym skryptem. Opis calej drogi: scripts/HAILO.md (krok 2).

Etapy (Dataflow Compiler, ClientRunner):
    1. parse     - ONNX -> wewnetrzny format Hailo (.parsed.har); siec ucieta na 6 ostatnich
                   warstwach Conv (KONCOWE); dekodowanie ramek i NMS dokleja kompilator
                   (nms_postprocess ze skryptu modelu)
    2. optimize  - kwantyzacja FP32 -> INT8 na zdjeciach kalibracyjnych (.quantized.har)
    3. compile   - przydzial do ukladu Hailo-8 -> plik .hef (+ .compiled.har)

Wynik (obok -o):
    <nazwa>.hef             - model dla Raspberry Pi (np. fieldeye_final_1280_h8.hef)
    <nazwa>_nms.json        - uzyta konfiguracja NMS
    <nazwa>.*.har           - zapis po kazdym etapie (do diagnozy)

Nastepny krok: scripts/check_hef.sh na Raspberry Pi.
Testy (bez kompilatora): tests/test_compile_hef.py.
"""

import argparse
import json
import re
from pathlib import Path

# 6 ostatnich warstw Conv glowicy YOLOv8 (3 skale x [ramki cv2, klasy cv3]) - tu konczy sie
# parsowanie ONNX; dalej (dekodowanie ramek + NMS) dokleja kompilator wg skryptu modelu
KONCOWE = ["/model.22/cv2.0/cv2.0.2/Conv", "/model.22/cv3.0/cv3.0.2/Conv",
           "/model.22/cv2.1/cv2.1.2/Conv", "/model.22/cv3.1/cv3.1.2/Conv",
           "/model.22/cv2.2/cv2.2.2/Conv", "/model.22/cv3.2/cv3.2.2/Conv"]


def konfiguracja_nms(wzor, rozmiar, klasy=1, prog=0.05, iou=0.6, maks=300):
    """Konfiguracja NMS z Model Zoo dopasowana do naszego modelu.

    Po co: wzor z Model Zoo jest dla obrazu 640, 80 klas COCO, progu 0,2 i 100 ramek -
           nasz model ma wejscie 1280, 1 klase, a chwastow na kafelku bywa wiecej.

    Argumenty:
        wzor    - yolov8n_nms_config.json z Model Zoo
        rozmiar - bok wejscia sieci, np. 1280
        klasy   - liczba klas, 1 (chwast)
        prog    - prog pewnosci NMS, np. 0.05
                  (ponizej - wykrycie znika juz w HEF, zanim zobaczy je nasz kod)
        iou     - prog IoU NMS, np. 0.6
        maks    - najwyzej tyle ramek na klase, np. 300

    Zwraca:
        dict - konfiguracja zapisywana do <nazwa>_nms.json

    Uzywana w:
        main() (ten plik), tests/test_compile_hef.py
    """
    c = json.loads(Path(wzor).read_text())
    c["image_dims"] = [rozmiar, rozmiar]
    c["classes"] = klasy
    c["nms_scores_th"] = prog
    c["nms_iou_th"] = iou
    c["max_proposals_per_class"] = maks
    return c


def skrypt_modelu(wzor_alls, sciezka_nms):
    """Skrypt modelu z Model Zoo z podmieniona sciezka do konfiguracji NMS.

    Po co: skrypt .alls (normalizacja /255, aktywacje wyjsc, NMS) bierzemy gotowy z Model Zoo,
           zmieniamy w nim tylko plik konfiguracji NMS na nasz.

    Argumenty:
        wzor_alls  - yolov8n.alls z Model Zoo
        sciezka_nms - nasz plik <nazwa>_nms.json

    Zwraca:
        str - tresc skryptu modelu; blad, gdy nms_postprocess nie wystepuje dokladnie raz

    Uzywana w:
        main() (ten plik), tests/test_compile_hef.py
    """
    s = Path(wzor_alls).read_text()
    s, n = re.subn(r'nms_postprocess\("[^"]*"', f'nms_postprocess("{sciezka_nms}"', s)
    if n != 1:
        raise ValueError(f"w {wzor_alls} nie ma jednego nms_postprocess (znaleziono {n})")
    return s


def zdjecia_kalibracyjne(katalog, rozmiar, liczba):
    """(N, H, W, 3) float32, RGB, 0-255 - normalizacja /255 jest w skrypcie modelu.

    Po co: zdjecia, na ktorych kompilator mierzy zakresy wartosci w sieci przy kwantyzacji INT8.

    Argumenty:
        katalog - folder z models.cli kalibracja --rozmiar N, np. "kalibracja_1280"
        rozmiar - wymagany bok zdjec, np. 1280 (inny rozmiar = blad z podpowiedzia)
        liczba  - najwyzej tyle zdjec, np. 128

    Zwraca:
        numpy (N, rozmiar, rozmiar, 3) float32, np. ksztalt (128, 1280, 1280, 3)

    Uzywana w:
        main() - etap optimize; tests/test_compile_hef.py
    """
    import cv2
    import numpy as np
    pliki = sorted(p for p in Path(katalog).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:liczba]
    if not pliki:
        raise FileNotFoundError(f"brak zdjec w {katalog}")
    dane = np.zeros((len(pliki), rozmiar, rozmiar, 3), np.float32)
    for i, p in enumerate(pliki):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (rozmiar, rozmiar):
            raise ValueError(f"{p.name}: {img.shape[1]}x{img.shape[0]}, oczekiwano {rozmiar}x{rozmiar} "
                             f"(models.cli kalibracja --rozmiar {rozmiar})")
        dane[i] = img
    return dane


def main(argv=None):
    """Cala kompilacja: konfiguracja NMS -> parse -> optimize -> compile -> plik .hef.

    Argumenty:
        argv - opcje jak w terminalu albo None (sys.argv):
               onnx (plik .onnx), kalibracja (folder), -o (plik .hef), --rozmiar (1280),
               --liczba (128), --prog-nms (0.05), --maks-ramek (300)

    Zwraca:
        0; zapisuje .hef, _nms.json i trzy pliki .har

    Uzywana w:
        python compile_hef.py ... w WSL (scripts/HAILO.md, krok 2)
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("kalibracja", help="katalog zdjec z models.cli kalibracja --rozmiar N")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--rozmiar", type=int, default=1280)
    ap.add_argument("--liczba", type=int, default=128, help="zdjec kalibracyjnych (bez GPU kompilator i tak bierze 64)")
    ap.add_argument("--prog-nms", type=float, default=0.05)
    ap.add_argument("--maks-ramek", type=int, default=300)
    a = ap.parse_args(argv)

    # pakiety Hailo importowane dopiero tutaj - funkcje wyzej daja sie testowac bez nich
    import hailo_model_zoo
    from hailo_sdk_client import ClientRunner

    cfg = Path(hailo_model_zoo.__file__).parent / "cfg"
    nms_wzor = cfg / "postprocess_config" / "yolov8n_nms_config.json"
    alls_wzor = cfg / "alls" / "generic" / "yolov8n.alls"
    for p in (nms_wzor, alls_wzor):
        if not p.exists():
            raise FileNotFoundError(f"brak {p} - dograj cfg z gałęzi v2.17 hailo_model_zoo (patrz dokument, 4.5)")

    # nasza konfiguracja NMS obok wyniku + skrypt modelu z podmieniona sciezka do niej
    wyj = Path(a.output)
    nazwa = wyj.stem
    nms_plik = wyj.with_name(nazwa + "_nms.json").resolve()
    nms_plik.write_text(json.dumps(konfiguracja_nms(nms_wzor, a.rozmiar, prog=a.prog_nms, maks=a.maks_ramek), indent=2))
    alls = skrypt_modelu(alls_wzor, nms_plik)
    print(f"  NMS: {nms_plik}  (obraz {a.rozmiar}, 1 klasa, prog {a.prog_nms}, do {a.maks_ramek} ramek)")

    runner = ClientRunner(hw_arch="hailo8")     # kompilator dla ukladu Hailo-8
    print("  1/3 parse")
    # nazwa sieci jak w Model Zoo: konfiguracja NMS odwoluje sie do warstw "yolov8n/conv41" itd.
    runner.translate_onnx_model(a.onnx, "yolov8n", start_node_names=["images"], end_node_names=KONCOWE,
                                net_input_shapes={"images": [1, 3, a.rozmiar, a.rozmiar]})
    runner.save_har(str(wyj.with_suffix(".parsed.har")))

    print("  2/3 optimize (kwantyzacja)")
    runner.load_model_script(alls)              # normalizacja /255, aktywacje, NMS
    runner.optimize(zdjecia_kalibracyjne(a.kalibracja, a.rozmiar, a.liczba))
    runner.save_har(str(wyj.with_suffix(".quantized.har")))

    print("  3/3 compile")
    hef = runner.compile()                      # bajty pliku .hef
    wyj.write_bytes(hef)
    runner.save_har(str(wyj.with_suffix(".compiled.har")))
    print(f"  HEF: {wyj} ({len(hef) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())