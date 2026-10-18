"""Miniaturowe imitacje czterech zbiorow - w DOKLADNIE tych samych formatach.

Po co: zeby dalo sie uruchomic i przetestowac caly potok normalizacji przed
pobraniem kilkunastu gigabajtow danych.

UWAGA: liczby wychodzace z imitacji nic nie mowia o prawdziwych zbiorach.
Sluza wylacznie do sprawdzenia, ze kod dziala i formaty sie zgadzaja.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np


def _tlo(h, w, rng):
    img = np.zeros((h, w, 3), np.float32)
    img[:] = (45, 60, 80)                       # BGR - brazowa gleba
    img += rng.normal(0, 10, (h, w, 1))
    return img


def _roslina(img, maska, cx, cy, r, kolor, wartosc, rng):
    for _ in range(6):
        dx, dy = rng.normal(0, r * 0.3, 2)
        rr = max(2, int(r * rng.uniform(0.45, 0.7)))
        c = (int(cx + dx), int(cy + dy))
        cv2.circle(img, c, rr, kolor, -1)
        if maska is not None:
            cv2.circle(maska, c, rr, wartosc, -1)


def _xml(sciezka, plik, w, h, obiekty):
    L = [f"<annotation><filename>{plik}</filename>",
         f"<size><width>{w}</width><height>{h}</height><depth>3</depth></size>"]
    for nazwa, (x0, y0, x1, y1) in obiekty:
        L.append(f"<object><name>{nazwa}</name><bndbox>"
                 f"<xmin>{x0}</xmin><ymin>{y0}</ymin>"
                 f"<xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox></object>")
    L.append("</annotation>")
    Path(sciezka).write_text("\n".join(L))


def weedsgalore(cel, rng):
    """Kanaly w osobnych plikach + maski semantyczne i instancji + splits/."""
    cel = Path(cel)
    dni = {"2023-05-25": 4, "2023-06-15": 3}
    ids = []
    for dzien, n in dni.items():
        for sub in ("images", "semantics", "instances"):
            (cel / dzien / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            oid = f"{dzien}_{i:04d}"
            ids.append(oid)
            h = w = 600
            img, sem = _tlo(h, w, rng), np.zeros((h, w), np.uint8)
            for rz in range(70, h, 140):                  # rzedy kukurydzy (klasa 1)
                for x in range(30, w, 60):
                    _roslina(img, sem, x, rz + rng.normal(0, 4), 18, (50, 150, 60), 1, rng)
            for _ in range(int(rng.integers(6, 14))):     # chwasty (klasy 2-5)
                _roslina(img, sem, rng.uniform(15, w - 15), rng.uniform(15, h - 15),
                         rng.uniform(7, 15), (60, 190, 110), int(rng.integers(2, 6)), rng)

            ins = np.zeros((h, w), np.uint16)             # numer kazdej rosliny
            nr = 0
            for v in range(1, 6):
                n_sk, lab = cv2.connectedComponents((sem == v).astype(np.uint8))
                for k in range(1, n_sk):
                    nr += 1
                    ins[lab == k] = nr

            img = np.clip(img, 0, 255).astype(np.uint8)
            b, g, r = cv2.split(img)
            for nazwa, kan in (("R", r), ("G", g), ("B", b),
                               ("NIR", np.clip(g * 1.3, 0, 255).astype(np.uint8)),
                               ("RE", np.clip(g * 1.1, 0, 255).astype(np.uint8))):
                cv2.imwrite(str(cel / dzien / "images" / f"{oid}_{nazwa}.png"), kan)
            cv2.imwrite(str(cel / dzien / "semantics" / f"{oid}.png"), sem)
            cv2.imwrite(str(cel / dzien / "instances" / f"{oid}.png"), ins)

    (cel / "splits").mkdir(exist_ok=True)
    (cel / "splits" / "train.txt").write_text("\n".join(ids[:4]) + "\n")
    (cel / "splits" / "val.txt").write_text(ids[4] + "\n")
    (cel / "splits" / "test.txt").write_text("\n".join(ids[5:]) + "\n")


def cofly(cel, rng):
    """Maski w labels_1d i labels; czesc zdjec BEZ chwastow (przyklady negatywne)."""
    cel = Path(cel)
    for sub in ("images", "labels", "labels_1d"):
        (cel / sub).mkdir(parents=True, exist_ok=True)
    kolory = {1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 255)}   # BGR
    for i in range(1, 9):
        nazwa = (f"ID_{i:05d}_UAV_[Lat={39.54 + i*1e-4:.6f},Lon={22.64 + i*1e-4:.6f},"
                 f"Alt={rng.uniform(4.6, 5.2):.3f}].png" if i % 2 else f"ID_{i:05d}.png")
        h, w = 360, 640
        img = _tlo(h, w, rng)
        for y in range(60, h, 120):
            for x in range(30, w, 60):
                _roslina(img, None, x, y, 16, (60, 140, 70), 0, rng)
        if i > 5:                                     # bez maski = brak chwastow
            cv2.imwrite(str(cel / "images" / nazwa), np.clip(img, 0, 255).astype(np.uint8))
            continue
        m = np.zeros((h, w), np.uint8)
        for _ in range(int(rng.integers(2, 6))):
            _roslina(img, m, rng.uniform(20, w - 20), rng.uniform(20, h - 20),
                     rng.uniform(10, 22), (70, 190, 120), int(rng.choice([1, 2, 3])), rng)
        cv2.imwrite(str(cel / "images" / nazwa), np.clip(img, 0, 255).astype(np.uint8))
        cv2.imwrite(str(cel / "labels_1d" / nazwa), m)
        kol = np.zeros((h, w, 3), np.uint8)
        for v, c in kolory.items():
            kol[m == v] = c
        cv2.imwrite(str(cel / "labels" / nazwa), kol)


def sudars(cel, rng):
    """Pascal VOC, skroty gatunkow, rozne rozdzielczosci, przewaga chwastow."""
    cel = Path(cel)
    (cel / "raw images").mkdir(parents=True, exist_ok=True)
    (cel / "annotations").mkdir(parents=True, exist_ok=True)
    chwasty = ["CA", "GA", "TA", "CB", "MI", "PC", "VA", "GP", "Galinsoga parviflora"]
    uprawy = ["BV", "DC", "RSS"]
    for i in range(10):
        w, h = [(640, 480), (480, 384), (560, 420)][i % 3]
        img = _tlo(h, w, rng)
        ob = []
        for nazwa in ([str(rng.choice(uprawy))] if i % 3 == 0 else []) + \
                     [str(rng.choice(chwasty)) for _ in range(int(rng.integers(3, 8)))]:
            r = rng.uniform(15, 45)
            cx, cy = rng.uniform(r, w - r), rng.uniform(r, h - r)
            _roslina(img, None, cx, cy, r, (60, 170, 90), 0, rng)
            ob.append((nazwa, (int(cx - r), int(cy - r), int(cx + r), int(cy + r))))
        if i == 9:
            ob.append(("xyz_nieznana", (10, 10, 40, 40)))      # do raportu nieznanych
        plik = f"ave-{i:04d}.jpg"
        cv2.imwrite(str(cel / "raw images" / plik), np.clip(img, 0, 255).astype(np.uint8))
        _xml(cel / "annotations" / f"ave-{i:04d}.xml", plik, w, h, ob)


def weedmaize(cel, rng):
    """Pascal VOC z kodami EPPO i faza wzrostu; gotowy podzial w katalogach."""
    cel = Path(cel)
    etyk = ["ZEAMX_V1", "ZEAMX_V3", "SOLNI_Vc", "CYPRO_min", "ECHCG_Vt"]
    for split, n in (("train_set", 6), ("validation_set", 2), ("test_set", 2)):
        (cel / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            w, h = 1000, 750
            img = _tlo(h, w, rng)
            ob = []
            for _ in range(int(rng.integers(6, 14))):
                nazwa = str(rng.choice(etyk))
                r = rng.uniform(18, 50) if nazwa.startswith("ZEAMX") else rng.uniform(8, 28)
                cx, cy = rng.uniform(r, w - r), rng.uniform(r, h - r)
                _roslina(img, None, cx, cy, r, (60, 160, 80), 0, rng)
                ob.append((nazwa, (int(cx - r), int(cy - r), int(cx + r), int(cy + r))))
            plik = f"{split[:3]}_{i:03d}.jpg"
            cv2.imwrite(str(cel / split / plik), np.clip(img, 0, 255).astype(np.uint8))
            _xml(cel / split / f"{split[:3]}_{i:03d}.xml", plik, w, h, ob)


def wlasne(cel, rng, kopia_z=None):
    """Dane wlasne w formacie YOLO. kopia_z: sciezka zdjecia do zduplikowania
    (do sprawdzenia, czy deduplikacja dziala miedzy zbiorami)."""
    cel = Path(cel)
    (cel / "misja_01").mkdir(parents=True, exist_ok=True)
    (cel / "classes.txt").write_text("chwast\nkukurydza\n")
    for i in range(4):
        w, h = 1200, 900
        img = _tlo(h, w, rng)
        wiersze = []
        for _ in range(int(rng.integers(3, 8))):
            r = rng.uniform(12, 30)
            cx, cy = rng.uniform(r, w - r), rng.uniform(r, h - r)
            _roslina(img, None, cx, cy, r, (60, 185, 105), 0, rng)
            wiersze.append(f"0 {cx/w:.6f} {cy/h:.6f} {2*r/w:.6f} {2*r/h:.6f}")
        cv2.imwrite(str(cel / "misja_01" / f"IMG_{i:04d}.jpg"),
                    np.clip(img, 0, 255).astype(np.uint8))
        (cel / "misja_01" / f"IMG_{i:04d}.txt").write_text("\n".join(wiersze) + "\n")

    if kopia_z and Path(kopia_z).exists():
        shutil.copy(kopia_z, cel / "misja_01" / "IMG_0099.jpg")
        (cel / "misja_01" / "IMG_0099.txt").write_text("0 0.5 0.5 0.1 0.1\n")


def utworz(katalog="dane_przykladowe", seed=0, z_duplikatem=True):
    """Tworzy wszystkie imitacje. Zwraca slownik sciezek dla zbuduj()."""
    katalog = Path(katalog)
    if katalog.exists():
        shutil.rmtree(katalog)
    rng = np.random.default_rng(seed)
    sciezki = {
        "weedsgalore": katalog / "weedsgalore-dataset",
        "cofly": katalog / "CoFly-WeedDB",
        "sudars": katalog / "sudars" / "dataset",
        "weedmaize": katalog / "weedmaize",
        "wlasne": katalog / "wlasne",
    }
    weedsgalore(sciezki["weedsgalore"], rng)
    cofly(sciezki["cofly"], rng)
    sudars(sciezki["sudars"], rng)
    weedmaize(sciezki["weedmaize"], rng)
    dubel = (sciezki["sudars"] / "raw images" / "ave-0000.jpg") if z_duplikatem else None
    wlasne(sciezki["wlasne"], rng, dubel)
    return {k: str(v) for k, v in sciezki.items()}
