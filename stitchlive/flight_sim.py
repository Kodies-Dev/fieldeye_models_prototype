"""Rozklad jednego duzego zdjecia na kadry - symulacja lotu drona.

Bierzemy obraz calego pola (ortomozaike, zdjecie satelitarne, dowolne duze
zdjecie) i wycinamy z niego kadry tak, jakby robil je dron. Potem mozemy porownac

Dwa tryby:
  trasa   - przelot "kosiarka" (linie tam i z powrotem), z losowym
            odchyleniem pozycji, kursu i wysokosci. Najbardziej realistyczny.
  losowy  - losowe miejsca ciecia. Kazdy kolejny kadr lezy w losowym
            kierunku od poprzedniego, ale zawsze z zadanym minimalnym
            pokryciem, zeby dalo sie go dopasowac.

Zaklocenia, ktore mozna wlaczyc:
  obrot      - odchylenie kursu drona (stopnie)
  skala      - zmiana wysokosci lotu (kadr obejmuje wiekszy/mniejszy teren)
  ekspozycja - zmiana jasnosci miedzy kadrami
  rozmycie   - rozmycie ruchowe od drgan
  szum       - szum matrycy
"""

import json
import logging
import math
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
# Wszystkie parametry symulacji
# Zapisywane w prawda.json (asdict), zeby wiadomo bylo, jak powstaly kadry.
class SimSettings:
    tryb: str = "trasa"          # trasa | losowy
    kadr: int = 640              # szerokosc zapisanego kadru w px (wzdluz kierunku lotu)
    kadr_wys: int = 0            # wysokosc kadru w px; 0 = kwadrat (kadr x kadr)
    zasieg: float = 0.0          # ile px zrodla obejmuje szerokosc kadru; 0 = tyle co kadr
    overlap: float = 0.65        # pokrycie wzdluz linii (trasa) / minimalne (losowy)
    sidelap: float = 0.5         # pokrycie miedzy liniami (trasa)
    liczba: int = 0              # ile kadrow (losowy); 0 = automatycznie
    jitter: float = 0.08         # odchylenie pozycji jako ulamek kroku
    obrot: float = 0.0           # odchylenie standardowe kursu w stopniach
    obrot_linii: bool = False    # co druga linia obrocona o 180 st. (jak prawdziwy dron)
    skala: float = 0.0           # odchylenie standardowe wysokosci (ulamek) - czyli gora dol dronu
    ekspozycja: float = 0.04     # losowa zmiana jasnoci kadru
    rozmycie: float = 0.0        # prawdopodobienstwo rozmycia kadru
    szum: float = 0.0            # odchylenie standardowe szumu (poziomy jasnosci) w pikselach osobno
    winieta: float = 0.0         # przyciemnienie rogow kadru, 0-1 (typowe dla kamer dronowych)
    # rozmycie od ruchu drona w czasie naswietlania, w px kadru, w KAZDYM kadrze:
    # predkosc * czas naswietlania / GSD (stitchlive.camera.CameraModel.rozmycie_px)
    rozmycie_ruch_px: float = 0.0
    seed: int = 7


def wymiary(u):
    """(szerokosc, wysokosc) kadru w px."""
    return u.kadr, (u.kadr_wys or u.kadr)


# ---------------- geometria wycinania ----------------
"""funkcja pod cv2.warpAffine - funkcja do wycinania obroconych kadrow
 przyjmuje macierz 2x3 (np.ndarray, float64):
            [[a, -b, tx],      a, b   - obrot i skala (s = zasieg/kadr, #maciere obrotu
             [b,  a, ty]]               a = s*cos(kat), b = s*sin(kat))
                               tx, ty - miejsce na polu lewego gornego rogu kadru
"""
def macierz_kadru(cx, cy, kat_st, zasieg, kadr, kadr_wys=None):
    """Liczy przepis na wyciecie jednego kadru z duzego zdjecia pola.

    Po co: symulujemy drona, ktory robi zdjecia w roznych miejscach, lekko
    obrocone i z roznej wysokosci. Zwykle wycinanie zrodlo[y0:y1, x0:x1]
    daje tylko prosty prostokat - nie obroci i nie przeskaluje kadru.
    Ta macierz zapisuje polozenie, obrot i skale w 6 liczbach, ktore
    rozumie cv2.warpAffine (uzywa jej wytnij()). Ta sama macierz daje tez
    rogi kadru do prawda.json - wiemy dokladnie, skad pochodzi kadr.

    Argumenty:
        cx, cy   - punkt zrodla, nad ktorym jest dron = srodek kadru (px)
        kat_st   - obrot kadru w stopniach (kurs drona)
        zasieg   - ile px zrodla obejmuje szerokosc kadru; wieksze = dron
                   wyzej, kadr obejmuje wiecej terenu
        kadr     - szerokosc zapisanego kadru (px) do tego dopausjemy zasieg
        kadr_wys - wysokosc kadru (px); None = kwadrat

    Zwraca:
        macierz 2x3 (np.ndarray float64) [[a, -b, tx], lewy gorny rog kadru
                                          [b,  a, ty]] prwy gorny og kadru
        Dla kazdego piksela kadru (x, y) mowi, ktory piksel zrodla wziac:
            x_zrodla = a*x - b*y + tx
            y_zrodla = b*x + a*y + ty
        a, b  - obrot i skala (a = s*cos(kat), b = s*sin(kat), s = zasieg/kadr)
        tx, ty - przesuniecie, tak zeby srodek kadru trafil w (cx, cy)

    Przyklad: cx=1000, cy=800, kat_st=0, zasieg=480, kadr=480
        -> [[1, 0, 760], [0, 1, 560]]: bez obrotu i skali, kadr to wycinek
           zrodla od (760, 560) do (1240, 1040)
    """
    # s - ile pikseli zrodla przypada na piksel kadru; a, b - skladowe obrotu ze skala.
    # Wyrazy wolne dobrane tak, zeby srodek kadru (h, h) trafil w (cx, cy) zrodla.
    s = zasieg / kadr
    t = math.radians(kat_st)
    a, b = s * math.cos(t), s * math.sin(t)
    hx, hy = kadr / 2.0, (kadr_wys or kadr) / 2.0
    return np.array([[a, -b, cx - a * hx + b * hy],
                     [b,  a, cy - b * hx - a * hy]], dtype=np.float64) #zwraca macierz pod cv warp aby wyciac obrocony kawalek ze zdj

#m macierz poprzednia, szerokosc wysokosc kadry
def narozniki_kadru(M, kadr, kadr_wys=None):
    """Liczy, gdzie na polu leza 4 rogi kadru.

    Po co: zeby wiedziec, skad dokladnie pochodzi kadr (prawda). Rogi trafiaja
    do prawda.json, na podglad trasy (czerwona ramka) i do oceny mozaiki.

    Argumenty:
        M        - macierz 2x3 z macierz_kadru() (piksel kadru -> piksel pola)
        kadr     - szerokosc kadru (px), np. 480
        kadr_wys - wysokosc kadru (px); None = kwadrat

    Zwraca:
        lista 4 punktow [x, y] w px pola, kolejno: lewy gorny, prawy gorny,
        prawy dolny, lewy dolny;"""
    w, h = kadr, (kadr_wys or kadr)
    pts = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) #rogi kadru od lewy gorny do prawy dolny 1 potem na koncu by dalo sie przemnozyc przez przesuniecie tx
    return (pts @ M.T).tolist() #zwracamy pts razy macierz transponowana dosajemy po prostu koordynaty wszystkich puntkow rogw kadru


""" Wycina kadr z duzego zdjecia pola wedlug macierzy M.

  zrodlo   - duze zdjecie pola (BGR)
  M        - macierz 2x3 z macierz_kadru():
  kadr     - szerokosc kadru (px)
  kadr_wys - wysokosc kadru (px); None = kwadrat 
  zwraca obraz kadru za pomoca funkcji open cv warAffine"""
def wytnij(zrodlo, M, kadr, kadr_wys=None):
    return cv2.warpAffine(zrodlo, M, (kadr, kadr_wys or kadr),
                          flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


# ---------------- planowanie polozen ----------------
#zasieg mówi ile zdj zrobione mialo pixeli u.kadr do ilu zostalo scisniete
def marginesy(zasieg, u):
    """Liczy najwieksza odleglosc srodka kadru od krawedzi pola, osobno w poziomie i w pionie.

    Po co: obrocony kadr i kadr z wyzszego lotu siegaja dalej niz pol boku.
    Srodek musi byc dalej od krawedzi, zeby kadr nie wystawal poza zdjecie.
    Liczy realny najgorszy przypadek (3 odchylenia standardowe):
        kat = 3 * obrot (najwyzej 45 st. - przy 45 kadr zajmuje najwiecej miejsca) losujemy z rozkladu nrmlanego wczesniej z przedzialu 0 - obrot
        z   = zasieg * (1 + 3 * skala), najwyzej * 1.33 najgorzsy realny przypadek
        prostokat obejmujacy obrocony kadr: szer = w|cos| + h|sin|, wys = w|sin| + h|cos| z trojkatow
        margines = polowa tego prostokata + 2 px zapasu

    Argumenty:
        zasieg - ile px pola obejmuje szerokosc kadru (bez zaklocen), np. 480
        u      - SimSettings (uzywa: kadr, kadr_wys, obrot, skala)

    Zwraca:
        (margines_poziomy, margines_pionowy) w px pola;
        np. kadr 480, obrot 3 st., skala 0.03 -> (301.3, 301.3)


    """
    kw, kh = wymiary(u) #wymiary prostokata,kadru
    kat = math.radians(min(45.0, 3 * u.obrot)) if u.obrot else 0.0 #kat miedzy kadrem z zewnetrznym prostkatem
    z = zasieg * (min(1.33, 1 + 3 * u.skala) if u.skala else 1.0) #szerokosc kadru przy najwyzszym locie
    zw, zh = z, z * kh / kw #obydwa boki
    c, s_ = abs(math.cos(kat)), abs(math.sin(kat)) #sinus i cosinus
    return (zw * c + zh * s_) / 2.0 + 2, (zw * s_ + zh * c) / 2.0 + 2 #najgorsze odleglosci x i y


def margines(zasieg, u):
    """
    zwraca bezpieczna odleglosc od krawedzi czyli bierze najwieksza liczbe z x i y
    """
    return max(marginesy(zasieg, u))


def plan_trasa(W, H, zasieg, u, rng):
    """Planuje lot : punkty zdjec w poziomych liniach, tam i z powrotem.

        "dron" lata pasami, z zadanym pokryciem
        miedzy kolejnymi zdjeciami (overlap) i miedzy pasami (sidelap). Pokrycie jest
        potrzebne, zeby skladanie mialo wspolne fragmenty do dopasowania.
        Jak liczy:
            krok_x = zasieg * (1 - overlap)   - co tyle px kolejne zdjecie w linii
            krok_y = wys. kadru * (1 - sidelap) - odstep miedzy liniami
            linie od marginesu do marginesu (marginesy()), co druga w przeciwna strone,
            kazdy punkt losowo przesuniety (jitter) - dron nie trafia idealnie.

        Argumenty:
            W, H   - szerokosc i wysokosc zdjecia pola (px)
            zasieg - ile px pola obejmuje szerokosc kadru, np. 480
            u      - SimSettings (uzywa: kadr, kadr_wys, overlap, sidelap, jitter,
                     obrot, skala, obrot_linii)
            rng    - generator losowy (np.random.default_rng(seed)) - ten sam seed
                     = ta sama trasa

        Zwraca:
            lista krotek (x, y, kurs, nr_linii) - po jednej na zdjecie, w kolejnosci lotu:
                x, y     - zaplanowany srodek kadru na polu (px, z jitterem)
                kurs     - 0 albo 180 st. (180 w co drugiej linii, gdy obrot_linii=True)
                nr_linii - numer pasa 0, 1, 2...
            np. pole 2400x1600, kadr 480, overlap 0.65, sidelap 0.5:
                krok_x 168, krok_y 240, margines 301 -> 11 kadrow x 5 linii = 55

        Rzuca:
            ValueError - gdy pole jest mniejsze niz dwa marginesy (kadr sie nie miesci)

        Uzywana w:
            rozloz() (flight_sim.py) - gdy u.tryb == "trasa"; dla kazdego punktu
            wycina kadr; nr_linii trafia do prawda.json ("linia").
        """
    kw, kh = wymiary(u)
    krok_x = zasieg * (1 - u.overlap)
    krok_y = zasieg * kh / kw * (1 - u.sidelap)      # odstep linii z wysokosci kadru w terenie
    mx, my = marginesy(zasieg, u)

    if W < 2 * mx or H < 2 * my:
        raise ValueError(f"Obraz {W}x{H} za maly na kadr obejmujacy "
                         f"{zasieg:.0f}x{zasieg * kh / kw:.0f} px")

    xs = np.arange(mx, W - mx + 1, krok_x)
    ys = np.arange(my, H - my + 1, krok_y)
    if len(xs) < 2:
        xs = np.array([mx, W - mx])

    plan = []
    for i, y in enumerate(ys):
        # co druga linia w przeciwna strone - dron zawraca na koncu pasa
        linia = xs if i % 2 == 0 else xs[::-1]
        kurs = 180.0 if (u.obrot_linii and i % 2 == 1) else 0.0
        for x in linia:
            # Plan to lista (x, y, kurs, numer_linii). Jitter - dron nigdy nie trafia
            # idealnie w zaplanowany punkt.
            plan.append((x + rng.normal(0, u.jitter * krok_x),
                         y + rng.normal(0, u.jitter * krok_y),
                         kurs, i))
    return plan

def plan_losowy(W, H, zasieg, u, rng):
    """Losowe miejsca ciecia z gwarancja pokrycia miedzy kolejnymi kadrami.

    Kolejny kadr lezy w losowym kierunku od poprzedniego, w odleglosci
    dajacej pokrycie co najmniej u.overlap. Gdy wypadlby poza obraz,
    losujemy kierunek ponownie.
    """
    marg = margines(zasieg, u)
    if W < 2 * marg or H < 2 * marg:
        raise ValueError(f"Obraz {W}x{H} za maly na kadr obejmujacy {zasieg:.0f} px")
    wolne = (W - 2 * marg) * (H - 2 * marg)
    if wolne < (zasieg * 2) ** 2:
        log.warning("Malo miejsca na trase (%.0fx%.0f px) - kadry beda sie "
                    "skupiac w jednym miejscu. Uzyj wiekszego obrazu albo "
                    "mniejszego --kadr / --obrot.", W - 2 * marg, H - 2 * marg)

    # Domyslna liczba kadrow: tyle, zeby obszar obrazu byl pokryty ok. 2,5 razy.
    n = u.liczba or max(8, int((W * H) / (zasieg * zasieg) * 2.5))
    # najdalszy krok, przy ktorym pokrycie z poprzednim kadrem jest jeszcze >= overlap
    max_krok = zasieg * (1 - u.overlap)

    x, y = rng.uniform(marg, W - marg), rng.uniform(marg, H - marg)
    plan = [(x, y, 0.0, 0)]
    kierunek = rng.uniform(0, 2 * math.pi)

    for _ in range(n - 1):
        for proba in range(50):
            # kierunek zmienia sie plynnie - jak dron, ktory skreca, a nie teleportuje sie
            kierunek += rng.normal(0, 0.35)
            # odleglosc 60-100% maksymalnej - zawsze z wymaganym pokryciem
            d = rng.uniform(0.6, 1.0) * max_krok
            nx, ny = x + d * math.cos(kierunek), y + d * math.sin(kierunek)
            if marg <= nx <= W - marg and marg <= ny <= H - marg:
                break
            kierunek += math.pi / 2          # odbicie od krawedzi
        else:
            # 50 prob bez skutku (zakleszczenie w rogu) - kadr w tym samym miejscu
            nx, ny = x, y
        x, y = nx, ny
        plan.append((x, y, 0.0, 0))
    return plan


# ---------------- zaklocenia ----------------

def _maska_winiety(h, w, sila):
    """Buduje maske winiety: jasnosc 1.0 w srodku kadru, malejaca ku rogom.

       Po co: prawdziwy obiektyw daje ciemniejsze rogi zdjecia (winietowanie,
       vignetting). Kadry z symulatora maja wygladac jak z kamery, a ciemne rogi
       utrudniaja skladanie (te same miejsca pola maja inna jasnosc na sasiednich
       kadrach). Wzor 1 - sila * r^2 to uproszczenie fizyki (ok. cos^4 kata
       padania) - dla malych katow kwadrat odleglosci jest dobrym przyblizeniem.

       Argumenty:
           h, w - wysokosc i szerokosc kadru (px)
           sila - jak mocno sciemnic rogi (0 = brak, 0.3 = rogi o 30% ciemniejsze);
                  z suwaka "Winietowanie" (u.winieta)

       Zwraca:
           np.ndarray (h, w, 1), float - mnoznik jasnosci dla kazdego piksela:
               srodek 1.0, srodek krawedzi 1 - sila/2, rog 1 - sila
               np. sila 0.3: srodek 1.0, srodek krawedzi 0.85, rog 0.7
           Ostatni wymiar 1 - zeby jedna maska mnozyla wszystkie 3 kanaly BGR.

       Uzywana w:
           zaklocenia() (flight_sim.py) - kadr mnozony przez maske:
           img * _maska_winiety(h, w, u.winieta)
       """
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((x - w / 2) / (w / 2), (y - h / 2) / (h / 2)) / np.sqrt(2)   # 0 srodek, 1 rog
    return (1.0 - sila * r ** 2)[:, :, None]


def _rozmyj_ruchem(img, dlugosc):
    """Rozmywa kadr smuga o zadanej dlugosci (px, takze ulamkowej) wzdluz kierunku lotu.

    Po co: dron leci, a migawka jest otwarta przez chwile - kazdy punkt pola
    rozmazuje sie na zdjeciu w smuge o dlugosci predkosc * czas_naswietlania / GSD
    (liczy to camera.py). W odroznieniu od losowego `rozmycie` ta smuga jest
    w KAZDYM kadrze i ma dokladnie wyliczona dlugosc. Skrajne piksele jadra
    dostaja wage czesciowa, wiec 2,5 px rozmywa naprawde o 2,5 px, a nie o 2 albo 3.

    Argumenty:
        img     - kadr (BGR, uint8)
        dlugosc - dlugosc smugi w px, np. 2.5 (u.rozmycie_ruch_px)

    Zwraca:
        rozmyty kadr - ten sam ksztalt i typ co img (cv2.filter2D, -1 = typ bez zmian)

    Jak liczy (przyklad dlugosc = 2.5):
        n = 3 piksele jadra (zaokraglone w gore, zawsze nieparzyste - jadro ma
            srodek, wiec obraz sie nie przesuwa)
        wagi [1, 1, 1] -> skrajne zmniejszone o (3 - 2.5)/2 = 0.25 -> [0.75, 1, 0.75]
        po podzieleniu przez sume (2.5): [0.3, 0.4, 0.3] - kazdy piksel staje sie
        srednia wazona 3 sasiadow w poziomie
        (dlugosc 4: n = 5, wagi [0.5, 1, 1, 1, 0.5] / 4)

    Uzywana w:
        zaklocenia() (flight_sim.py) - gdy u.rozmycie_ruch_px >= 1 px;
        wartosc liczy camera.opis_lotu() w zakladce 4 (app.py)
        i cli.py (rozloz --wysokosc). W zakladkach 1-3 jest 0 - nieuzywana.
    """
    n = int(math.ceil(dlugosc))
    n += 1 - n % 2                                   # nieparzysta szerokosc jadra
    wagi = np.ones(n, np.float32)
    nadmiar = (n - dlugosc) / 2.0
    wagi[0] -= nadmiar
    wagi[-1] -= nadmiar
    jadro = np.zeros((n, n), np.float32)
    jadro[n // 2, :] = wagi / wagi.sum()
    return cv2.filter2D(img, -1, jadro)


# Naklada na wyciety kadr zaklocenia typowe dla kamery drona. Kolejnosc ma
# znaczenie: najpierw optyka (winieta), potem ekspozycja, ruch, na koncu szum
# matrycy - tak jak w prawdziwym torze obrazu.
def zaklocenia(img, u, rng):
    """Dodaje do wycietego kadru zaklocenia prawdziwej kamery, w kolejnosci jak w torze obrazu.

        Po co: kadr wyciety z oryginalu jest "za czysty". Prawdziwe zdjecia z drona
        maja ciemne rogi, rozna jasnosc, bywaja rozmazane i zaszumione - a skladanie
        musi sobie z tym radzic. Kolejnosc jak w kamerze: najpierw optyka
        (winieta), potem ekspozycja, ruch (rozmycie), na koncu matryca (szum).

        Kolejne kroki (kazdy tylko gdy jego ustawienie > 0):
            1. winieta       - rogi ciemniejsze: kadr * maska (1 - winieta * r^2)
            2. ekspozycja    - caly kadr * jeden losowy mnoznik 1 + N(0, ekspozycja)
            3. rozmycie_ruch_px - smuga z fizyki lotu w KAZDYM kadrze (zakladka 4)
            4. rozmycie      - z prawdopodobienstwem u.rozmycie losowa smuga 5-12 px
                               (drgania, podmuch wiatru)
            5. szum          - do kazdego piksela i kanalu dodany N(0, szum)
        Przy mnozeniu/dodawaniu kadr idzie na float32 i z powrotem przez clip(0, 255),
        zeby wartosci nie "zawinely sie" w uint8 (np. 260 -> 4).

        Argumenty:
            img - wyciety kadr (BGR, uint8), z wytnij()
            u   - SimSettings (uzywa: winieta, ekspozycja, rozmycie_ruch_px, rozmycie, szum)
            rng - generator losowy; ten sam seed = te same zaklocenia

        Zwraca:
            kadr z zakloceniami - ten sam ksztalt, uint8, BGR; gdy wszystkie
            ustawienia = 0 (zestaw "Czyste ciecie"), kadr bez zmian

        Uzywana w:
            rozloz() (flight_sim.py) - zaraz po wytnij(), przed zapisem IMG_xxxx.jpg
        """
    if u.winieta:
        img = np.clip(img.astype(np.float32) * _maska_winiety(*img.shape[:2], u.winieta),
                      0, 255).astype(np.uint8)
    if u.ekspozycja:
        # jeden losowy wspolczynnik jasnosci na caly kadr
        g = 1.0 + rng.normal(0, u.ekspozycja)
        img = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)
    if u.rozmycie_ruch_px and u.rozmycie_ruch_px >= 1.0:
        img = _rozmyj_ruchem(img, u.rozmycie_ruch_px)
    if u.rozmycie and rng.random() < u.rozmycie:
        k = int(rng.integers(5, 13))
        # Jadro rozmycia ruchowego: pozioma linia jedynek (usrednianie k pikseli w poziomie).
        jadro = np.zeros((k, k), np.float32)
        jadro[k // 2, :] = 1.0 / k           # rozmycie wzdluz kierunku lotu
        img = cv2.filter2D(img, -1, jadro)
    if u.szum:
        # Szum niezalezny w kazdym pikselu i kanale - jak szum matrycy przy slabym swietle.
        img = np.clip(img.astype(np.float32) + rng.normal(0, u.szum, img.shape),
                      0, 255).astype(np.uint8)
    return img


# ---------------- glowna funkcja ----------------

def wczytaj_zrodlo(sciezka, max_wymiar=0):
    """Wczytuje zdjecie pola z dysku i w razie potrzeby je zmniejsza.

        Po co: wersja z terminala tego, co w aplikacji robia wczytaj_z_uploadu()
        + zmniejsz() (app.py). Duze zdjecie z drona (ponad 4000 px) zmniejszone
        kilkukrotnie przyspiesza ciecie, skladanie i ocene.
        INTER_AREA - przy zmniejszaniu usrednia piksele, bez mory na rzedach uprawy.

        Argumenty:
            sciezka    - sciezka do pliku obrazu (str albo Path), np. "pole.jpg"
            max_wymiar - najwiekszy dozwolony dluzszy bok w px; 0 = bez zmniejszania.
                         Proporcje zostaja, obraz nigdy nie jest powiekszany.

        Zwraca:
            np.ndarray (wysokosc, szerokosc, 3), uint8, BGR
            przykladowo 4000x3000 przy max_wymiar=2400 -> 2400x1800



        Uzywana w:
            cli.py (polecenie rozloz) - zrodlo do rozloz() przy
            python -m stitchlive.cli rozloz ZDJECIE ... --max-wymiar 2400.
            Aplikacja (app.py) jej nie uzywa.
        """
    img = cv2.imread(str(sciezka), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Nie mozna wczytac obrazu: {sciezka}")
    if max_wymiar and max(img.shape[:2]) > max_wymiar:
        k = max_wymiar / max(img.shape[:2])
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        log.info("Zrodlo zmniejszone do %dx%d", img.shape[1], img.shape[0])
    return img

#funckja pelna symulujcaca
def rozloz(zrodlo, outdir, u=None, na_zywo=None, odstep=0.0):
    """Symuluje lot drona: tnie zdjecie pola na kadry, zapisuje je razem z prawda.

    Po co: glowna funkcja symulatora - laczy wszystkie poprzednie. Z jednego
    duzego zdjecia robi serie kadrow takich, jakie zrobilby dron (przesuniete,
    obrocone, z roznej wysokosci, z zakloceniami), i zapisuje, skad dokladnie
    pochodzi kazdy kadr. Te kadry sklada potem zakladka 3, a prawda sluzy do
    oceny, czy mozaika zgadza sie z polem.

    Kroki:
        1. rng z u.seed - ten sam seed = te same kadry przy kazdym uruchomieniu
        2. czysci outdir/images (stare kadry nie mieszaja sie z nowymi)
        3. zasieg = u.zasieg albo u.kadr (0 = tyle co kadr)
        4. plan lotu: plan_trasa() albo plan_losowy() (u.tryb) + marginesy
        5. dla kazdego punktu planu:
             - srodek przyciety do marginesow (jitter mogl go wypchnac)
             - kat = kurs + N(0, obrot); zasieg z = zasieg * (1 + N(0, skala)),
               przyciety do +-3 sigma (najwyzej +33%) - tak jak w marginesach
             - macierz_kadru() -> wytnij() -> zaklocenia() -> IMG_0001.jpg (jakosc 94)
             - narozniki_kadru() -> wpis do prawdy + czerwona ramka na podgladzie
        6. zolta linia trasy na podgladzie, zapis plikow, log.info
        7. opcjonalnie: wypuszczanie kadrow co `odstep` s do folderu na_zywo

    Argumenty:
        zrodlo  - zdjecie pola (np.ndarray, BGR)
        outdir  - folder wynikow (str albo Path); powstaje, jesli nie ma
        u       - SimSettings; None = wartosci domyslne
        na_zywo - folder, do ktorego kadry trafiaja stopniowo, jak z lecacego
                  drona (test trybu watch w cli.py); None = nie wypuszczaj
        odstep  - co ile sekund wypuszczac kolejny kadr do na_zywo

    Zwraca:
        slownik prawdy (to samo, co zapisane w prawda.json):
            {"zrodlo_px": [W, H],
             "ustawienia": {...wszystkie pola SimSettings...},
             "kadry": [{"plik": "IMG_0001.jpg",
                        "srodek": [x, y],         # srodek kadru na polu (px)
                        "kat_st": 2.11,           # obrot kadru (stopnie)
                        "zasieg_px": 487.5,       # ile px pola obejmuje szerokosc kadru
                        "kadr_px": [480, 480],    # rozmiar zapisanego kadru
                        "linia": 0,               # numer pasa (trasa); w losowym 0
                        "naroza": [[x, y] x 4]},  # rogi kadru na polu
                       ...]}

    Zapisuje w outdir:
        images/IMG_0001.jpg ...  - kadry
        prawda.json              - jak wyzej
        oryginal.jpg             - zrodlo (do oceny w zakladce 3)
        podglad_trasy.jpg        - czerwone ramki kadrow + zolta linia trasy

    Uzywana w:
        app.py (zakladka 1)    - "Rozloz zdjecie"
        app.py (zakladka 4)    - lot z detekcja
        mission/drone_sim.py   - symulowany dron z telemetria
        stitchlive/cli.py      - polecenia rozloz i demo z terminala
    """
    u = u or SimSettings()
    # jedno ziarno na cala symulacje = te same kadry przy kazdym uruchomieniu
    rng = np.random.default_rng(u.seed)
    outdir = Path(outdir)
    imgdir = outdir / "images"
    if imgdir.exists():
        shutil.rmtree(imgdir)
    imgdir.mkdir(parents=True)

    H, W = zrodlo.shape[:2] #wysokosc szerokosc zrodla (duzego zdj)
    zasieg = u.zasieg or u.kadr #ile widac pola (np bazowo 480 px ze zdj 4000x4000)

    # najpierw plan (gdzie ciac), potem wlasciwe ciecie
    plan = (plan_losowy if u.tryb == "losowy" else plan_trasa)(W, H, zasieg, u, rng)
    kw, kh = wymiary(u)
    if u.tryb == "losowy":
        mx = my = margines(zasieg, u)  # tryb losowy: ten sam margines w obu kierunkach
    else:
        mx, my = marginesy(zasieg, u) #zwraca nam jak daleko musi byc srodek kadru aby nie wyleciec poza zdj

    kadry = []
    podglad = zrodlo.copy()
    for i, (cx, cy, kurs, linia) in enumerate(plan, start=1): # srodek kadru x,y kierunek drona numer pasa
        cx = float(np.clip(cx, mx, W - mx)) #pilnowanie bysmy nie wyszli poza zdj jezeli wychodzi to mx lub w-mx
        cy = float(np.clip(cy, my, H - my))
        # kurs linii + losowe myszkowanie drona
        kat = kurs + (rng.normal(0, u.obrot) if u.obrot else 0.0) #losowe odchylenie drona
        # Zmiana wysokosci = zmiana zasiegu kadru w terenie. Przycinamy do zakresu
        # uwzglednionego w marginesie, zeby kadr nie wyszedl poza obraz.
        z = zasieg * (1.0 + (rng.normal(0, u.skala) if u.skala else 0.0)) #dron leci w gore w dol wiec idzi wiecej mniej pixeli ze zdj
        z_max = min(1.33, 1 + 3 * u.skala) if u.skala else 1.0 #o ile najweiyzej moze poleciec dron
        z = float(np.clip(z, zasieg * max(0.6, 2 - z_max), zasieg * z_max)) #przycina do bezpeiczengo zakresu kadr

        M = macierz_kadru(cx, cy, kat, z, kw, kh) #macierz od open cv pod warpAffine
        img = zaklocenia(wytnij(zrodlo, M, kw, kh), u, rng) #wycina z zakloceniami

        nazwa = f"IMG_{i:04d}.jpg"
        cv2.imwrite(str(imgdir / nazwa), img, [cv2.IMWRITE_JPEG_QUALITY, 94])

        nar = narozniki_kadru(M, kw, kh)
        # tablica prawda o kadrze: gdzie byl srodek, jaki kat i zasieg, gdzie narozniki.
        # Na tej podstawie ocena sprawdza, czy mozaika zgadza sie z oryginalem.
        kadry.append({"plik": nazwa, "srodek": [round(cx, 1), round(cy, 1)],
                      "kat_st": round(kat, 2), "zasieg_px": round(z, 1), "kadr_px": [kw, kh],
                      "linia": linia, "naroza": [[round(a, 1), round(b, 1)] for a, b in nar]})

        pts = np.array(nar, np.int32).reshape(-1, 1, 2) #zamiana naroznikow z [4,2] na [4,1,2] tego wymaga open cv
        cv2.polylines(podglad, [pts], True, (0, 0, 255), 2) #obramowanie zdj
        cv2.putText(podglad, str(i), (int(cx) - 10, int(cy) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) #numer zdj

    # linia trasy - kolejnosc wykonywania zdjec
    trasa = np.array([k["srodek"] for k in kadry], np.int32).reshape(-1, 1, 2) #pod open cv
    cv2.polylines(podglad, [trasa], False, (0, 255, 255), 2) #laczymy po srodkach

    cv2.imwrite(str(outdir / "oryginal.jpg"), zrodlo, [cv2.IMWRITE_JPEG_QUALITY, 92])#czytse zdj
    cv2.imwrite(str(outdir / "podglad_trasy.jpg"), podglad, [cv2.IMWRITE_JPEG_QUALITY, 85]) #zdj

    prawda = {"zrodlo_px": [W, H], "ustawienia": asdict(u), "kadry": kadry}
    (outdir / "prawda.json").write_text(json.dumps(prawda, indent=1), encoding="utf-8")
    log.info("Rozlozono na %d kadrow -> %s", len(kadry), imgdir)

    if na_zywo:
        _wypuszczaj(imgdir, Path(na_zywo), odstep)
    return prawda


def _wypuszczaj(imgdir, cel, odstep):
    """Kopiuje kadry do katalogu obserwowanego, po jednym, jak w locie.

    Zapis do pliku tymczasowego i zmiana nazwy - obserwator nigdy nie
    zobaczy pliku w polowie zapisu.
    """
    cel.mkdir(parents=True, exist_ok=True)
    for p in sorted(imgdir.glob("*.jpg")):
        tmp = cel / f".tmp_{p.name}"
        shutil.copy(p, tmp)
        # zmiana nazwy jest atomowa - plik pojawia sie w calosci albo wcale
        tmp.rename(cel / p.name)
        log.info("  lot: %s", p.name)
        if odstep:
            time.sleep(odstep)