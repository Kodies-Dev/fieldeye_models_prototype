"""Kroki potoku miedzy adapterami a eksportem: deduplikacja, podzial, kafelki.

Kazda funkcja bierze liste rekordow manifestu i zwraca liste rekordow.

Kolejnosc wywolan w build.py zbuduj():
    deduplikuj()          - prawie identyczne zdjecia odpadaja (skrot percepcyjny phash)
    zastosuj_tryb_klas()  - gatunki -> klasy treningowe (u nas "chwasty": jedna klasa)
    podziel()             - train/val/test po GRUPACH; sprawdz_podzial() - kontrola przeciekow
    przeskaluj()          - skala zdjec (u nas EWIS1 x0.47)
    kafelkuj_wszystkie()  - duze zdjecia -> kafelki 896 px + dobor pustych kafelkow

Wszystko dzieje sie na liczbach w Record (rozmiar, ramki, split) - same zdjecia
sa ciete i skalowane dopiero przy zapisie w export.py.
"""

import hashlib
import logging
from collections import defaultdict

import cv2
import numpy as np

from .manifest import LabeledObject, Record

log = logging.getLogger(__name__)

# priorytet przy duplikacie: zostaje rekord ze zbioru wyzej na liscie
# (zbiorow spoza listy, np. ewis1, dostaja najnizszy priorytet)
PRIORYTET = ["wlasne", "weedsgalore", "sudars"]


# =====================================================================
# 1. DEDUPLIKACJA
# =====================================================================

def phash(sciezka, rozmiar=32):
    """Skrot percepcyjny (DCT) - 64 bity opisujace WYGLAD obrazu.

    Zwykla suma kontrolna nie wystarczy: po przeskalowaniu albo rekompresji
    plik jest inny, a obraz ten sam. Skrot percepcyjny to wychwyci.

    Realny przypadek: zbior "weed crop aerial" z Roboflow 100 pochodzi
    z danych Sudarsa - te same zdjecia pod inna nazwa.

    Jak dziala: zdjecie w skali szarosci -> 32x32 -> DCT (rozklad na "fale"
    jasnosci) -> 8x8 najwolniejszych fal -> bit 1 tam, gdzie wartosc > mediany.

    Argumenty:
        sciezka - plik zdjecia
        rozmiar - do jakiego kwadratu zmniejszyc przed DCT, np. 32

    Zwraca:
        str z 64 znakow "0"/"1", np. "1011000...", albo None, gdy nie da sie wczytac

    Uzywana w:
        dolicz_phash() (ten plik)
    """
    img = cv2.imread(str(sciezka), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (rozmiar, rozmiar), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(img))[:8, :8]
    mediana = np.median(dct[1:, 1:])       # pomijamy skladowa stala (jasnosc)
    return "".join("01"[int(b)] for b in (dct > mediana).flatten())


def _sciezka_obrazu(r):
    """WeedsGalore trzyma prefiks bez kanalu - dokladamy _R.png.

    Argumenty:
        r - Record

    Zwraca:
        Path do istniejacego pliku zdjecia (dla WeedsGalore: kanal R)

    Uzywana w:
        dolicz_phash() (ten plik)
    """
    from pathlib import Path
    p = Path(r.obraz)
    return p if p.exists() else Path(str(p) + "_R.png")


def dolicz_phash(rekordy):
    """Liczy phash dla rekordow, ktore go jeszcze nie maja.

    Argumenty:
        rekordy - lista Record

    Zwraca:
        ta sama lista (pole r.phash uzupelnione w miejscu)

    Uzywana w:
        deduplikuj() (ten plik)
    """
    for r in rekordy:
        if r.phash is None:
            r.phash = phash(_sciezka_obrazu(r))
    return rekordy


def _odleglosc(a, b):
    """Odleglosc Hamminga: na ilu pozycjach dwa skroty sie roznia.

    Argumenty:
        a, b - skroty phash, np. "1011..." i "1001..."

    Zwraca:
        int 0-64; 0 = identyczny wyglad

    Uzywana w:
        deduplikuj() (ten plik)
    """
    return sum(x != y for x, y in zip(a, b))


def deduplikuj(rekordy, prog=5):
    """Usuwa zdjecia wygladajace tak samo jak juz przyjete.

    prog = ile bitow z 64 moze sie roznic. 0 to identyczny obraz,
    5 to ten sam obraz po rekompresji albo lekkim przeskalowaniu.

    Zeby nie porownywac kazdy z kazdym (przy kilkudziesieciu tysiacach zdjec
    to roznica miedzy sekundami a godzinami), skrot dzielimy na prog + 1 kawalkow.
    Dwa skroty rozne na najwyzej `prog` bitach maja co najmniej jeden kawalek
    identyczny (zasada szufladkowa), wiec wystarczy porownac zdjecie z tymi,
    ktore maja taki sam ktorys kawalek - zaden duplikat nie umknie.

    Po co: duplikat w treningu i w tescie zawyza wynik - dlatego krok jest PRZED podzialem.

    Argumenty:
        rekordy - lista Record ze wszystkich adapterow
        prog    - maks. liczba roznych bitow, np. 5

    Zwraca:
        (zostaje, odrzucone) - lista Record bez duplikatow i lista slownikow,
        np. [{"id": "sudars/x", "duplikat_z": "weedsgalore/...", "roznych_bitow": 2}]

    Uzywana w:
        build.py zbuduj() - krok 2; odrzucone -> duplikaty.json
    """
    dolicz_phash(rekordy)
    kolejnosc = {z: i for i, z in enumerate(PRIORYTET)}
    posort = sorted(rekordy, key=lambda r: kolejnosc.get(r.zrodlo, 99))   # wazniejsze zbiory najpierw

    n = prog + 1                           # ile kawalkow skrotu; prog 5 -> 6 kawalkow po 10-11 bitow
    granice = [64 * i // n for i in range(n + 1)]
    kubelki = defaultdict(list)            # (numer kawalka, kawalek) -> przyjete rekordy
    zostaje, odrzucone = [], []
    for r in posort:
        if r.phash is None:
            zostaje.append(r)
            continue
        klucze = [(i, r.phash[granice[i]:granice[i + 1]]) for i in range(n)]
        # kandydaci: przyjete rekordy z tym samym ktoryms kawalkiem (kazdy raz, w kolejnosci przyjecia)
        kandydaci = {id(c): c for k in klucze for c in kubelki.get(k, [])}.values()
        dubel = next((c for c in kandydaci if _odleglosc(r.phash, c.phash) <= prog), None)
        if dubel is not None:
            odrzucone.append({"id": r.id, "duplikat_z": dubel.id,
                              "roznych_bitow": _odleglosc(r.phash, dubel.phash)})
        else:
            for k in klucze:
                kubelki[k].append(r)
            zostaje.append(r)

    if odrzucone:
        log.info("Deduplikacja: odrzucono %d zdjec", len(odrzucone))
    return zostaje, odrzucone


# =====================================================================
# 2. PODZIAL train / val / test
# =====================================================================

def podziel(rekordy, proporcje=(0.70, 0.15, 0.15), sol="fieldeye-v1",
            zachowaj_istniejace=True, tylko_wlasne_do_testu=False):
    """Przydziela kazdemu rekordowi split - NA POZIOMIE GRUPY, nie zdjecia.

    Dlaczego deterministycznie przez skrot, a nie losowo: dodanie nowej misji
    nie przetasowuje istniejacego podzialu. Ten sam identyfikator grupy zawsze
    daje ten sam split.

    tylko_wlasne_do_testu: zbior testowy wylacznie z danych wlasnych. Wynik
    na zbiorze testowym zlozonym z danych publicznych nie mowi nic o
    skutecznosci na docelowym polu.

    Jak dziala: sha256("fieldeye-v1:" + grupa) -> liczba 0-1; ponizej train -> train,
    ponizej train+val -> val, reszta -> test. Rekordy z gotowym splitem (WeedsGalore
    z podzialu autorow, EWIS1 z test_dni) zostaja bez zmian.

    Argumenty:
        rekordy     - lista Record
        proporcje   - (train, val, test), np. (0.70, 0.15, 0.15) albo (1, 0, 0)
        sol         - dopisek do skrotu; zmiana = zupelnie inny podzial
        zachowaj_istniejace - True = nie ruszaj rekordow, ktore maja juz split
        tylko_wlasne_do_testu - True = test tylko z danych wlasnych

    Zwraca:
        ta sama lista (r.split ustawione w miejscu)

    Uzywana w:
        build.py zbuduj() - krok 4
    """
    grupy = {}                            # grupa -> split (raz policzony dla calej grupy)
    for r in rekordy:
        if zachowaj_istniejace and r.split:
            r.split_oficjalny = True      # oficjalny podzial autorow zbioru
            continue
        g = r.grupa or r.id
        if g not in grupy:
            h = hashlib.sha256(f"{sol}:{g}".encode()).digest()
            x = int.from_bytes(h[:8], "big") / 2 ** 64    # 8 bajtow skrotu -> liczba 0-1
            tr, va, _ = proporcje
            grupy[g] = "train" if x < tr else ("val" if x < tr + va else "test")
        r.split = grupy[g]

    if tylko_wlasne_do_testu:
        for r in rekordy:
            if r.split == "test" and r.zrodlo != "wlasne":
                r.split = "train"

    return rekordy


def sprawdz_podzial(rekordy):
    """Kontrola przeciekow: czy grupa nie wystepuje w dwoch czesciach podzialu.

    To jest ten blad, ktory daje zawyzone mAP i nie widac go w metrykach.

    Rozrozniamy dwa przypadki, bo znacza co innego:

      bledy       - grupa, ktora MY dzielilismy, trafila do dwoch czesci.
                    To blad w kodzie, nie powinien wystapic nigdy.

      oficjalne   - grupa rozjechana przez oficjalny podzial autorow zbioru.
                    Tak jest w WeedsGalore: podzial jest warstwowany po datach
                    lotu, wiec kazda data ma kafelki w train, val i test.
                    To jedno pole fotografowane cztery razy - trzeba sprawdzic,
                    czy te same MIEJSCA pola nie trafiaja do dwoch czesci.
                    Alternatywa: --wlasny-podzial, ktory ignoruje podzial
                    autorow i dzieli po grupach (kosztem porownywalnosci
                    z wynikami z publikacji).

    Argumenty:
        rekordy - lista Record po podziel()

    Zwraca:
        dict, np. {"bledy": {}, "oficjalne": {"weedsgalore_2023-05-25": ["test", "train", "val"]}}

    Uzywana w:
        build.py zbuduj() - wynik idzie do logu, na terminal i do build_info.json
    """
    gdzie = defaultdict(set)              # grupa -> zbior czesci, w ktorych wystepuje
    oficjalne = set()
    for r in rekordy:
        if r.grupa and r.split:
            gdzie[r.grupa].add(r.split)
            if getattr(r, "split_oficjalny", False):
                oficjalne.add(r.grupa)
    rozjechane = {g: sorted(s) for g, s in gdzie.items() if len(s) > 1}
    return {"bledy": {g: s for g, s in rozjechane.items() if g not in oficjalne},
            "oficjalne": {g: s for g, s in rozjechane.items() if g in oficjalne}}


# =====================================================================
# 3. KAFELKOWANIE
# =====================================================================

def kafelkuj(rekord, kafelek=896, zakladka=0.1, min_widocznosc=0.4,
             zwroc_puste_osobno=False,
             udzial_pustych=0.15, rng=None):
    """Tnie duze zdjecie na kafelki i przelicza ramki do ich ukladu.

    Po co: zdjecie z kapsuly ma ~4000 px, wejscie sieci 640 px. Zmniejszenie
    calosci szesciokrotnie robi z chwastu o boku 40 px obiekt 6-pikselowy,
    ktorego nie znajdzie zaden detektor. Kafelkowanie zachowuje rozdzielczosc.

    min_widocznosc: ramka przycieta na krawedzi kafelka do mniej niz tego
    udzialu pierwotnej powierzchni jest odrzucana. Bez tego progu na
    krawedziach powstaja szczatkowe ramki uczace model blednych wzorcow.

    udzial_pustych: kafelki bez zadnego obiektu to PRZYKLADY NEGATYWNE
    (sama uprawa, gola gleba, resztki pozniwne). Bez nich model reaguje na
    kazda nieregularnosc. Zostawiamy losowo czesc z nich.

    Argumenty:
        rekord         - Record jednego zdjecia (rozmiar i ramki juz po skali)
        kafelek        - bok kafelka w px, np. 896
        zakladka       - zakladka sasiednich kafelkow 0-1, np. 0.1
        min_widocznosc - minimalna czesc ramki, ktora musi zostac w kafelku, np. 0.4
        zwroc_puste_osobno - True = zwroc (z_obiektami, puste) zamiast jednej listy
        udzial_pustych - ile pustych dobrac przy pojedynczym wywolaniu, np. 0.15
        rng            - generator losowy numpy albo None

    Zwraca:
        lista Record kafelkow (kafelek=[x, y], szer=wys=896, ramki w ukladzie kafelka)
        albo (z_obiektami, puste), gdy zwroc_puste_osobno=True;
        zdjecie mniejsze niz kafelek (np. WeedsGalore 600 px) zostaje w calosci

    Uzywana w:
        kafelkuj_wszystkie() (ten plik), tests/test_normalization.py
    """
    rng = rng or np.random.default_rng(0)
    W, H = rekord.szer, rekord.wys
    if max(W, H) <= kafelek:
        # zdjecie mniejsze niz kafelek - bez ciecia (np. WeedsGalore: 600 px)
        if zwroc_puste_osobno:
            # zdjecie bez obiektow traktujemy jak pusty kafelek - podlega
            # temu samemu globalnemu limitowi przykladow negatywnych
            return ([rekord], []) if rekord.obiekty else ([], [rekord])
        return [rekord]

    # pozycje lewych gornych rogow kafelkow: co krok px, ostatni dosuniety do krawedzi
    krok = int(kafelek * (1 - zakladka))
    poz_x = sorted({min(x, W - kafelek) for x in range(0, W, krok)} | {W - kafelek})
    poz_y = sorted({min(y, H - kafelek) for y in range(0, H, krok)} | {H - kafelek})
    poz_x = [max(0, x) for x in poz_x]
    poz_y = [max(0, y) for y in poz_y]

    z_obiektami, puste = [], []
    for ty in poz_y:
        for tx in poz_x:
            obiekty = []
            for o in rekord.obiekty:
                x0, y0, x1, y1 = o.xyxy
                pole0 = max(1e-6, (x1 - x0) * (y1 - y0))
                # czesc ramki, ktora miesci sie w kafelku
                cx0, cy0 = max(x0, tx), max(y0, ty)
                cx1, cy1 = min(x1, tx + kafelek), min(y1, ty + kafelek)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue                      # ramka calkiem poza kafelkiem
                if (cx1 - cx0) * (cy1 - cy0) >= min_widocznosc * pole0:
                    # ramka w ukladzie kafelka (odejmujemy jego rog)
                    obiekty.append(LabeledObject(klasa=o.klasa,
                                          xyxy=[cx0 - tx, cy0 - ty, cx1 - tx, cy1 - ty],
                                          zrodlowa_nazwa=o.zrodlowa_nazwa, faza=o.faza))

            # kafelek dziedziczy wszystko po zdjeciu, w tym split i grupe
            nowy = Record(id=f"{rekord.id}_k{tx}_{ty}", obraz=rekord.obraz,
                          szer=kafelek, wys=kafelek, zrodlo=rekord.zrodlo,
                          grupa=rekord.grupa, domena=rekord.domena, agl_m=rekord.agl_m,
                          gsd_mm=rekord.gsd_mm, phash=rekord.phash,
                          wyczerpujace=rekord.wyczerpujace, split=rekord.split,
                          kafelek=[tx, ty], skala=rekord.skala, obiekty=obiekty)
            (z_obiektami if obiekty else puste).append(nowy)

    if zwroc_puste_osobno:
        return z_obiektami, puste

    # dobor pustych na poziomie jednego zdjecia - uzywane tylko przy
    # bezposrednim wywolaniu kafelkuj(); kafelkuj_wszystkie() dobiera je
    # globalnie, co daje wlasciwy udzial w calym zbiorze
    if puste and udzial_pustych > 0:
        ile = max(1, int(len(z_obiektami) * udzial_pustych)) if z_obiektami else len(puste)
        wybrane = rng.choice(len(puste), size=min(ile, len(puste)), replace=False)
        z_obiektami += [puste[i] for i in wybrane]
    return z_obiektami


def kafelkuj_wszystkie(rekordy, udzial_pustych=0.15, **kw):
    """Kafelkuje wszystkie zdjecia i dobiera puste kafelki GLOBALNIE.

    Udzial liczony na calym zbiorze, nie na pojedynczym zdjeciu. Inaczej
    zdjecie z jednym obiektem wymuszaloby jeden pusty kafelek i realny
    udzial pustych skakalby do 50%, zamiast zadanych kilkunastu procent.

    Puste kafelki to przyklady negatywne - sama uprawa, gola gleba, resztki
    pozniwne. Bez nich model reaguje na kazda nieregularnosc.

    Argumenty:
        rekordy        - lista Record (cale zdjecia, po podziale i skali)
        udzial_pustych - jaka czesc WYNIKU maja stanowic puste kafelki, np. 0.15
        **kw           - przekazywane do kafelkuj(): kafelek, zakladka, ...; seed - ziarno losowania

    Zwraca:
        lista Record kafelkow: wszystkie z obiektami + losowo dobrane puste

    Uzywana w:
        build.py zbuduj() - krok 5
    """
    rng = np.random.default_rng(kw.pop("seed", 0))    # seed nie idzie do kafelkuj()
    z_obiektami, puste = [], []
    for r in rekordy:
        a, b = kafelkuj(r, rng=rng, zwroc_puste_osobno=True, **kw)
        z_obiektami += a
        puste += b

    dobrane = []
    if puste and udzial_pustych > 0:
        # ile pustych, zeby stanowily udzial_pustych CALEGO wyniku
        ile = int(round(len(z_obiektami) * udzial_pustych / max(1e-9, 1 - udzial_pustych)))
        ile = min(ile, len(puste))
        if ile:
            wybrane = rng.choice(len(puste), size=ile, replace=False)
            dobrane = [puste[i] for i in wybrane]

    out = z_obiektami + dobrane
    log.info("Kafelkowanie: %d zdjec -> %d kafelkow (%d z obiektami, %d pustych)",
             len(rekordy), len(out), len(z_obiektami), len(dobrane))
    return out


# =====================================================================
# 4. SKALA, LIMIT ZDJEC, MAPOWANIE KLAS NA ETYKIETY TRENINGOWE
# =====================================================================

def przeskaluj(rekordy, skale, min_bok=1.0):
    """Zmniejsza (albo powieksza) zdjecia wybranych zrodel: {zrodlo: wspolczynnik}.

    Po co: zdjecia z ziemi albo z niska (Sudars, EWIS1) sa dokladniejsze niz z
    drona - chwasty maja na nich kilka razy wiecej pikseli. Zmniejszenie do GSD
    drona daje modelowi siewki takiej wielkosci, jaka zobaczy w locie.
    Wspolczynnik = GSD zrodla / GSD docelowe; u nas EWIS1 1,18 / 2,5 = 0,47.

    Zmienia tylko liczby w rekordzie (rozmiar, ramki, GSD); sam obraz jest
    zmniejszany przy eksporcie. Ramki mniejsze niz min_bok px po zmniejszeniu
    sa odrzucane - nie da sie ich juz rozpoznac.

    Argumenty:
        rekordy - lista Record
        skale   - slownik zrodlo -> wspolczynnik, np. {"ewis1": 0.47}
        min_bok - minimalny bok ramki po skali w px, np. 1.0

    Zwraca:
        int - ile ramek odrzucono (rekordy zmieniane w miejscu)

    Uzywana w:
        build.py zbuduj() - krok 4b, przed kafelkowaniem (--skala)
    """
    odrzucone = 0
    for r in rekordy:
        f = skale.get(r.zrodlo)
        if not f or f == 1.0:
            continue
        r.szer, r.wys = max(1, round(r.szer * f)), max(1, round(r.wys * f))
        r.skala = (r.skala or 1.0) * f        # zapamietane dla export.py (zmniejszy zdjecie)
        if r.gsd_mm:
            r.gsd_mm = r.gsd_mm / f           # 1,18 / 0,47 = ok. 2,5 mm/px
        nowe = []
        for o in r.obiekty:
            x0, y0, x1, y1 = (v * f for v in o.xyxy)
            if x1 - x0 >= min_bok and y1 - y0 >= min_bok:
                o.xyxy = [x0, y0, x1, y1]
                nowe.append(o)
            else:
                odrzucone += 1
        r.obiekty = nowe
    if odrzucone:
        log.info("przeskalowanie: odrzucono %d ramek mniejszych niz %.0f px", odrzucone, min_bok)
    return odrzucone


def zastosuj_tryb_klas(rekordy, tax, tryb="binarnie"):
    """Zamienia klucze gatunkow na docelowe etykiety i usuwa odrzucone obiekty.

    tryb: binarnie (chwast/uprawa), chwasty (same chwasty), gatunki.

    Po co: w trybie "chwasty" (model koncowy) kazdy chwast dostaje klase "chwast",
           a ramki upraw znikaja - kukurydza staje sie tlem.

    Argumenty:
        rekordy - lista Record
        tax     - Taxonomy (etykieta(): gatunek -> klasa albo None)
        tryb    - "binarnie", "chwasty" albo "gatunki"

    Zwraca:
        ta sama lista (r.obiekty zmienione w miejscu)

    Uzywana w:
        build.py zbuduj() - krok 3 (--tryb-klas)
    """
    for r in rekordy:
        nowe = []
        for o in r.obiekty:
            et = tax.etykieta(o.klasa, tryb)
            if et is None:
                continue                      # np. kukurydza w trybie "chwasty" -> tlo
            o.klasa = et
            nowe.append(o)
        r.obiekty = nowe
    return rekordy
