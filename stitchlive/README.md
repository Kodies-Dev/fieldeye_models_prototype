# stitchlive/ — kod programu

Pakiet Pythona z całą logiką. GUI (`../app.py`) i wiersz poleceń (`cli.py`)
tylko go wywołują.

## Moduły

| Plik | Rola | Najważniejsze elementy |
|---|---|---|
| `global_stitcher.py` | **domyślna metoda składania** | `GlobalStitcher` — pary kadrów z OpenCV (ORB + RANSAC), położenie wszystkich kadrów z jednego układu równań liniowych |
| `detail.py` | metoda `detail` | `DetailStitcher` — pipeline `cv2.detail`, model afiniczny, okno przesuwne |
| `basic.py` | metoda `basic` | `BasicStitcher` — opakowanie `cv2.Stitcher` w trybie SCANS |
| `flight_sim.py` | symulator lotu | `SimSettings`, `rozloz()` — cięcie jednego zdjęcia na kadry (kwadrat albo prostokąt 4:3), trasa lub losowo, z zakłóceniami i rozmyciem ruchem |
| `camera.py` | model kamery drona | `CameraModel`, `opis_lotu()` — ile pola widzi zdjęcie z danej wysokości, kadr w pikselach źródła, kafelek 1,5 m, rozmycie ruchem; Arducam 64MP (FIELDEYE) i Raspberry Pi HQ |
| `evaluation.py` | ocena wyniku | `ocen()` — dopasowuje mozaikę do oryginału i liczy PSNR, pokrycie, skalę |
| `field.py` | dane testowe | `zbuduj_pole()` — sztuczne pole z rzędami, chwastami i kamieniami |
| `source.py` | źródła zdjęć | `FolderSource` (obserwuje katalog w wątku), `BatchSource` |
| `cli.py` | wiersz poleceń | komendy `batch`, `watch`, `info`, `rozloz`, `ocen`, `demo` |

## Wspólny interfejs składaczy

Każdy składacz ma te same metody, więc GUI i CLI mogą je podmieniać:

```python
s = GlobalStitcher(work_megapix=0.6, renderuj_co=3)
for sciezka in zdjecia:
    if s.dodaj(sciezka):        # True = mozaika zaktualizowana
        podglad(s.mozaika)
s.zakoncz()                     # tylko global: końcowe renderowanie
s.raport_czasu()                # [{"zdjec": n, "czas_s": t, "ok": bool}, ...]
```

## Przepływ danych

```
camera.opis_lotu()   ->  rozmiar kadru, kafelek, rozmycie      (tryb kamery drona)
flight_sim.rozloz()  ->  images/*.jpg + prawda.json
                             |
source.*Source       ->  kolejka  ->  *Stitcher.dodaj()  ->  mozaika
                                                              |
                                  evaluation.ocen(mozaika, oryginał, prawda)
```

## Kamera drona

```python
from stitchlive.camera import ARDUCAM_64MP_16MP as kam, opis_lotu
kam.zasieg_m(5.0)                                   # (7.25, 5.44) m pola na zdjęciu
opis_lotu(kam, 5.0, 2.5, 3.0, 1/1000, 0.7)          # kadr 2902 x 2176 px źródła, kafelek 600 px, rozmycie 1,2 px
```

W wierszu poleceń: `python -m stitchlive.cli rozloz --wysokosc 5 ...`, w interfejsie:
zakładka 4, pole „Kamera drona”. Szczegóły algorytmów: README w katalogu głównym.