"""mission - pelny przebieg lotu: detekcja i mapowanie w locie, mozaika po wyladowaniu.

Podmoduly sa importowane dopiero przy pierwszym uzyciu (PEP 562). Dzieki temu
`fieldeye-detect` nie wczytuje geolokalizacji (pyproj), mozaiki i reszty misji,
a binarka zbudowana z niego zawiera tylko to, czego potrzebuje.
"""

import importlib

__all__ = ["MissionConfig", "zbuduj_detektor", "geo", "in_flight", "post_flight", "telemetry"]
_PODMODULY = {"geo", "in_flight", "post_flight", "telemetry"}


def __getattr__(nazwa):
    if nazwa in _PODMODULY:
        return importlib.import_module(f".{nazwa}", __name__)
    if nazwa == "MissionConfig":
        from .config import MissionConfig
        return MissionConfig
    if nazwa == "zbuduj_detektor":
        from .detector import zbuduj_detektor
        return zbuduj_detektor
    raise AttributeError(f"module {__name__!r} has no attribute {nazwa!r}")
