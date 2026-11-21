"""Wybor urzadzenia obliczeniowego - ten sam kod na Windowsie i na Macu.

Ultralytics sam wybiera tylko karty NVIDIA (CUDA). Na Macu z Apple Silicon,
bez jawnego device="mps", trening szedlby na procesorze. Ta funkcja wybiera
najlepsze dostepne urzadzenie: CUDA -> MPS (Apple) -> CPU.

Modul trzeba zaimportowac PRZED torch: ustawia PYTORCH_ENABLE_MPS_FALLBACK,
dzieki ktoremu operacje nieobslugiwane przez MPS licza sie na procesorze,
zamiast przerywac trening bledem.

Uzywany w:
    training.py, evaluation.py, cli.py (srodowisko), mission/detector.py (YoloDetector), app.py
"""

import os
import platform

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def wybierz(urzadzenie=None):
    """Wybiera urzadzenie do treningu i detekcji: CUDA -> MPS (Apple) -> CPU.

    Po co: ultralytics sam wybiera tylko karte NVIDIA - na Macu bez tego liczylby na procesorze.

    Argumenty:
        urzadzenie - None / "" / "auto" = wybierz najlepsze; inna wartosc (np. "cpu", "0")
                     przechodzi bez zmian

    Zwraca:
        str: "0" (pierwsza karta NVIDIA), "mps" (Mac) albo "cpu"

    Uzywana w:
        training.py trenuj_etap(), evaluation.py (ocen, predykcje), opis() (ten plik),
        mission/detector.py YoloDetector, app.py
    """
    if urzadzenie not in (None, "", "auto"):
        return urzadzenie
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def opis():
    """Co jest zainstalowane i co zostanie uzyte - do diagnozy srodowiska.

    Zwraca:
        dict, np. {"system": "Darwin arm64", "python": "3.14.0", "torch": "2.x", "cuda": False,
                   "mps": True, "ultralytics": "8.x", "wybrane_urzadzenie": "mps", "mps_fallback": "1"}

    Uzywana w:
        models/cli.py cmd_srodowisko() - komenda "python -m models.cli srodowisko"
    """
    info = {"system": f"{platform.system()} {platform.machine()}",
            "python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["karta"] = torch.cuda.get_device_name(0)
        mps = getattr(torch.backends, "mps", None)
        info["mps"] = bool(mps and mps.is_available())
    except ImportError:
        info["torch"] = "BRAK - zainstaluj: pip install -r requirements.txt"
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = "BRAK"
    info["wybrane_urzadzenie"] = wybierz()
    info["mps_fallback"] = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    return info
