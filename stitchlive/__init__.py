"""stitchlive - skladanie zdjec z drona w czasie rzeczywistym na bazie OpenCV.

Pliki:
    global_stitcher.py - GlobalStitcher: pary + globalne wyrownanie (domyslna metoda, app.py i misja)
    detail.py          - DetailStitcher: pipeline cv2.detail (porownanie, zakladka 3 app.py)
    basic.py           - BasicStitcher: cv2.Stitcher od nowa po kazdym zdjeciu (punkt odniesienia)
    source.py          - zrodla zdjec: obserwacja folderu (watch) albo caly folder (batch)
    flight_sim.py      - symulator lotu: pole -> kadry z zakloceniami + prawda.json
    camera.py          - model kamery drona (Arducam 64MP): zasieg, GSD, kadr, kafelek
    field.py           - sztuczne pole do testow
    evaluation.py      - ocena mozaiki wzgledem oryginalu (skala, pokrycie, PSNR, werdykt)
    cli.py             - python -m stitchlive.cli batch / watch / info / rozloz / ocen / demo
"""

__version__ = "0.1.0"

from .basic import BasicStitcher
from .detail import DetailStitcher
from .global_stitcher import GlobalStitcher
from .source import FolderSource, BatchSource

__all__ = ["BasicStitcher", "DetailStitcher", "GlobalStitcher",
           "FolderSource", "BatchSource"]
