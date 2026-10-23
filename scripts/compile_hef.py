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
"""

import argparse
import json
import re
from pathlib import Path

KONCOWE = ["/model.22/cv2.0/cv2.0.2/Conv", "/model.22/cv3.0/cv3.0.2/Conv",
           "/model.22/cv2.1/cv2.1.2/Conv", "/model.22/cv3.1/cv3.1.2/Conv",
           "/model.22/cv2.2/cv2.2.2/Conv", "/model.22/cv3.2/cv3.2.2/Conv"]


def konfiguracja_nms(wzor, rozmiar, klasy=1, prog=0.05, iou=0.6, maks=300):
    """Konfiguracja NMS z Model Zoo dopasowana do naszego modelu."""
    c = json.loads(Path(wzor).read_text())
    c["image_dims"] = [rozmiar, rozmiar]
    c["classes"] = klasy
    c["nms_scores_th"] = prog
    c["nms_iou_th"] = iou
    c["max_proposals_per_class"] = maks
    return c


def skrypt_modelu(wzor_alls, sciezka_nms):
    """Skrypt modelu z Model Zoo z podmieniona sciezka do konfiguracji NMS."""
    s = Path(wzor_alls).read_text()
    s, n = re.subn(r'nms_postprocess\("[^"]*"', f'nms_postprocess("{sciezka_nms}"', s)
    if n != 1:
        raise ValueError(f"w {wzor_alls} nie ma jednego nms_postprocess (znaleziono {n})")
    return s


def zdjecia_kalibracyjne(katalog, rozmiar, liczba):
    """(N, H, W, 3) float32, RGB, 0-255 - normalizacja /255 jest w skrypcie modelu."""
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("kalibracja", help="katalog zdjec z models.cli kalibracja --rozmiar N")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--rozmiar", type=int, default=1280)
    ap.add_argument("--liczba", type=int, default=128, help="zdjec kalibracyjnych (bez GPU kompilator i tak bierze 64)")
    ap.add_argument("--prog-nms", type=float, default=0.05)
    ap.add_argument("--maks-ramek", type=int, default=300)
    a = ap.parse_args(argv)

    import hailo_model_zoo
    from hailo_sdk_client import ClientRunner

    cfg = Path(hailo_model_zoo.__file__).parent / "cfg"
    nms_wzor = cfg / "postprocess_config" / "yolov8n_nms_config.json"
    alls_wzor = cfg / "alls" / "generic" / "yolov8n.alls"
    for p in (nms_wzor, alls_wzor):
        if not p.exists():
            raise FileNotFoundError(f"brak {p} - dograj cfg z gałęzi v2.17 hailo_model_zoo (patrz dokument, 4.5)")

    wyj = Path(a.output)
    nazwa = wyj.stem
    nms_plik = wyj.with_name(nazwa + "_nms.json").resolve()
    nms_plik.write_text(json.dumps(konfiguracja_nms(nms_wzor, a.rozmiar, prog=a.prog_nms, maks=a.maks_ramek), indent=2))
    alls = skrypt_modelu(alls_wzor, nms_plik)
    print(f"  NMS: {nms_plik}  (obraz {a.rozmiar}, 1 klasa, prog {a.prog_nms}, do {a.maks_ramek} ramek)")

    runner = ClientRunner(hw_arch="hailo8")
    print("  1/3 parse")
    runner.translate_onnx_model(a.onnx, "fieldeye", start_node_names=["images"], end_node_names=KONCOWE,
                                net_input_shapes={"images": [1, 3, a.rozmiar, a.rozmiar]})
    runner.save_har(str(wyj.with_suffix(".parsed.har")))

    print("  2/3 optimize (kwantyzacja)")
    runner.load_model_script(alls)
    runner.optimize(zdjecia_kalibracyjne(a.kalibracja, a.rozmiar, a.liczba))
    runner.save_har(str(wyj.with_suffix(".quantized.har")))

    print("  3/3 compile")
    hef = runner.compile()
    wyj.write_bytes(hef)
    runner.save_har(str(wyj.with_suffix(".compiled.har")))
    print(f"  HEF: {wyj} ({len(hef) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())