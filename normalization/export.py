"""Manifest -> gotowy zbior w formacie YOLO.

Jedyne miejsce, w ktorym powstaje format YOLO. Wszystkie zbiory zrodlowe
przeszly juz przez manifest, wiec ten kod jest napisany raz i nie zalezy
od tego, skad dane pochodza.
"""

import json
import logging
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


_PAMIEC = {"klucz": None, "obraz": None}


def _wczytaj_obraz(rekord):
    """Zdjecie rekordu jako BGR.

    WeedsGalore trzyma kanaly w osobnych plikach (_R, _G, _B) i bywaja
    16-bitowe - skladamy je i sprowadzamy do 8 bitow.

    Ostatnie zdjecie zostaje w pamieci: kafelki jednego zdjecia ida po kolei,
    wiec duze PNG (EWIS1: ok. 40 MB) sa rozpakowywane raz, a nie raz na kafelek.
    Zwracana jest kopia, bo wycinki kafelkow moga byc pozniej modyfikowane.
    """
    klucz = (str(rekord.obraz), getattr(rekord, "skala", 1.0) or 1.0)
    if _PAMIEC["klucz"] == klucz and _PAMIEC["obraz"] is not None:
        return _PAMIEC["obraz"].copy()
    img = _wczytaj_obraz_z_dysku(rekord)
    _PAMIEC["klucz"], _PAMIEC["obraz"] = klucz, img
    return None if img is None else img.copy()


def _wczytaj_obraz_z_dysku(rekord):
    img = _wczytaj_bez_skali(rekord)
    f = getattr(rekord, "skala", 1.0) or 1.0
    if img is not None and f != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))),
                         interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    return img


def _wczytaj_bez_skali(rekord):
    p = Path(rekord.obraz)
    if p.exists():
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    kanaly = []
    for k in ("B", "G", "R"):                       # OpenCV trzyma BGR
        c = cv2.imread(f"{rekord.obraz}_{k}.png", cv2.IMREAD_UNCHANGED)
        if c is None:
            return None
        kanaly.append(c if c.ndim == 2 else c[:, :, 0])
    img = np.dstack(kanaly)
    if img.dtype == np.uint8:
        return img
    # WSPOLNA skala dla trzech kanalow. Rozciaganie kazdego osobno zmienia
    # proporcje kolorow zaleznie od zawartosci zdjecia - przy gestej
    # roslinnosci gleba robila sie rozowa. Percentyle zamiast min/max, zeby
    # pojedyncze przepalone piksele nie spychaly reszty do ciemnosci.
    f = img.astype(np.float32)
    lo, hi = np.percentile(f, 0.5), np.percentile(f, 99.5)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def eksportuj(rekordy, katalog, nazwy_klas=None, kopiuj_obrazy=True, jakosc=92):
    """Zapisuje zbior w ukladzie, ktorego oczekuje Ultralytics.

        dataset/
          images/train/  images/val/  images/test/
          labels/train/  labels/val/  labels/test/
          data.yaml

    Etykiety YOLO: jedna linia na obiekt, wartosci znormalizowane do 0-1:
        <klasa> <x_srodka> <y_srodka> <szerokosc> <wysokosc>
    """
    katalog = Path(katalog)
    if nazwy_klas is None:
        nazwy_klas = sorted({o.klasa for r in rekordy for o in r.obiekty})
    idx = {n: i for i, n in enumerate(nazwy_klas)}

    for split in ("train", "val", "test"):
        (katalog / "images" / split).mkdir(parents=True, exist_ok=True)
        (katalog / "labels" / split).mkdir(parents=True, exist_ok=True)

    licznik, pominiete = Counter(), 0
    for r in rekordy:
        split = r.split or "train"
        nazwa = r.id.replace("/", "__")

        if kopiuj_obrazy:
            img = _wczytaj_obraz(r)
            if img is None:
                log.warning("brak obrazu: %s", r.obraz)
                pominiete += 1
                continue
            if r.kafelek:                           # rekord jest kafelkiem - wytnij go
                tx, ty = r.kafelek
                img = img[ty:ty + r.wys, tx:tx + r.szer]
            cv2.imwrite(str(katalog / "images" / split / f"{nazwa}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, jakosc])

        wiersze = []
        for o in r.obiekty:
            if o.klasa not in idx:
                continue
            x0, y0, x1, y1 = o.xyxy
            cx, cy = (x0 + x1) / 2 / r.szer, (y0 + y1) / 2 / r.wys
            w, h = (x1 - x0) / r.szer, (y1 - y0) / r.wys
            # przyciecie do 0-1: ramka na krawedzi moze minimalnie wystawac
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
            if w <= 0 or h <= 0:
                continue
            wiersze.append(f"{idx[o.klasa]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            licznik[o.klasa] += 1

        # plik etykiet powstaje ZAWSZE, takze pusty - to przyklad negatywny
        (katalog / "labels" / split / f"{nazwa}.txt").write_text(
            "\n".join(wiersze) + ("\n" if wiersze else ""), encoding="utf-8")

    _data_yaml(katalog, nazwy_klas)
    if pominiete:
        log.warning("Pominieto %d rekordow bez obrazu", pominiete)
    return {"klasy": nazwy_klas, "obiektow": dict(licznik), "pominiete": pominiete}


def _data_yaml(katalog, nazwy):
    linie = ["# wygenerowane przez normalization/export.py",
             f"path: {Path(katalog).resolve()}",
             "train: images/train", "val: images/val", "test: images/test",
             "names:"]
    linie += [f"  {i}: {n}" for i, n in enumerate(nazwy)]
    (katalog / "data.yaml").write_text("\n".join(linie) + "\n", encoding="utf-8")


def zapisz_info(katalog, info):
    """Zapis parametrow budowy - bez tego po miesiacu nie odtworzysz,
    na czym trenowany byl model, ktory dal najlepszy wynik."""
    info = {**info, "data": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    p = Path(katalog) / "build_info.json"
    p.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def podglad_ramek(rekordy, katalog, ile=12, seed=0):
    """Rysuje ramki na probce zdjec - DO OBEJRZENIA WZROKIEM.

    Blad w konwersji (przesuniecie o pol piksela, pomylona kolejnosc osi,
    zla wartosc maski) jest niewidoczny w metrykach, a psuje caly trening.
    Ten kwadrans ogladania oszczedza dni.
    """
    katalog = Path(katalog)
    katalog.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    z_obiektami = [r for r in rekordy if r.obiekty]
    if not z_obiektami:
        return []
    wybrane = [z_obiektami[i] for i in
               rng.choice(len(z_obiektami), min(ile, len(z_obiektami)), replace=False)]

    out = []
    for r in wybrane:
        img = _wczytaj_obraz(r)
        if img is None:
            continue
        if r.kafelek:
            tx, ty = r.kafelek
            img = img[ty:ty + r.wys, tx:tx + r.szer]
        img = img.copy()
        for o in r.obiekty:
            x0, y0, x1, y1 = (int(v) for v in o.xyxy)
            kol = (60, 220, 60) if "uprawa" in o.klasa else (230, 40, 160)
            cv2.rectangle(img, (x0, y0), (x1, y1), kol, max(1, img.shape[1] // 400))
            cv2.putText(img, o.klasa[:18], (x0, max(12, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, kol, 1, cv2.LINE_AA)
        p = katalog / f"{r.id.replace('/', '__')}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append(p)
    return out