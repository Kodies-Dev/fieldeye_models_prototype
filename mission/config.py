"""Konfiguracja misji - wszystkie parametry w jednym miejscu.

Wczytywana z pliku YAML/JSON albo uzywana z wartosciami domyslnymi.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class CameraConfig:
    """Parametry optyki - potrzebne do przeliczenia pikseli na metry.

    Wartosci domyslne: modul kamery Raspberry Pi HQ (sensor IMX477).
    OGNISKOWA RZECZYWISTA, nie ekwiwalent 35 mm.
    """
    szer_matrycy_mm: float = 6.287
    ogniskowa_mm: float = 6.0


@dataclass
class FlightConfig:
    """Co wiemy o locie, gdy brakuje tego w metadanych zdjecia."""
    agl_domyslne: float = 30.0        # wysokosc nad terenem [m]
    wysokosc_terenu_npm: float = 0.0  # do przeliczenia wysokosci GPS na AGL
    kurs_domyslny: float = 0.0        # gdy brak kompasu
    korekta_kursu: float = 0.0        # montaz kamery wzgledem osi drona [st.]


@dataclass
class DetectionConfig:
    model: str = None                 # .pt, .onnx albo .hef (Hailo); None = atrapa testowa
    # prog i wejscie, przy ktorych model docelowy byl uczony i oceniany:
    # prog dobrany na walidacji (0,15), wejscie modelu wg_sudars (1280).
    # Dla modelu 640 podaj wejscie_sieci: 640; przy .hef wejscie jest wbudowane w plik.
    prog_pewnosci: float = 0.15
    wejscie_sieci: int = 1280
    kafelkowanie: bool = True
    kafelek: int = 896
    zakladka: float = 0.15
    # klasy uznawane za chwast (reszta, np. uprawa, nie trafia na mape zabiegowa)
    klasy_chwastow: tuple = ("chwast", "weed", "atrapa")


@dataclass
class MapConfig:
    promien_scalania_m: float = 0.35   # ta sama roslina z sasiednich zdjec
    min_pewnosc_na_mape: float = 0.30


@dataclass
class MissionConfig:
    kamera: CameraConfig = field(default_factory=CameraConfig)
    lot: FlightConfig = field(default_factory=FlightConfig)
    detekcja: DetectionConfig = field(default_factory=DetectionConfig)
    mapa: MapConfig = field(default_factory=MapConfig)

    # skroty - zeby nie pisac cfg.lot.agl_domyslne w kazdym miejscu
    @property
    def agl_domyslne(self):
        return self.lot.agl_domyslne

    @property
    def wysokosc_terenu_npm(self):
        return self.lot.wysokosc_terenu_npm

    @property
    def kurs_domyslny(self):
        return self.lot.kurs_domyslny

    @property
    def korekta_kursu(self):
        return self.lot.korekta_kursu

    @classmethod
    def wczytaj(cls, sciezka=None):
        if sciezka is None:
            return cls()
        sciezka = Path(sciezka)
        tekst = sciezka.read_text(encoding="utf-8")
        if sciezka.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise RuntimeError("brak PyYAML - uzyj pliku .json")
            dane = yaml.safe_load(tekst) or {}
        else:
            dane = json.loads(tekst)
        return cls(kamera=CameraConfig(**dane.get("kamera", {})),
                   lot=FlightConfig(**dane.get("lot", {})),
                   detekcja=DetectionConfig(**dane.get("detekcja", {})),
                   mapa=MapConfig(**dane.get("mapa", {})))

    def slownik(self):
        return asdict(self)
