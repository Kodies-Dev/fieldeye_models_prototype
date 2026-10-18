"""Wybor urzadzenia obliczeniowego - ten sam kod na Windowsie i na Macu.

Ultralytics sam wybiera tylko karty NVIDIA (CUDA). Na Macu z Apple Silicon,
bez jawnego device="mps", trening szedlby na procesorze. Ta funkcja wybiera
najlepsze dostepne urzadzenie: CUDA -> MPS (Apple) -> CPU.

Modul trzeba zaimportowac PRZED torch: ustawia PYTORCH_ENABLE_MPS_FALLBACK,
dzieki ktoremu operacje nieobslugiwane przez MPS licza sie na procesorze,
zamiast przerywac trening bledem.
"""

import os
import platform

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def wybierz(urzadzenie=None):
    """None / "" / "auto" -> najlepsze dostepne. Inna wartosc przechodzi bez zmian."""
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
    """Co jest zainstalowane i co zostanie uzyte - do diagnozy srodowiska."""
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
