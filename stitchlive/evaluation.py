"""Ocena mozaiki wzgledem oryginalu, z ktorego zostala rozlozona.

Mozaika ma wlasny, umowny uklad wspolrzednych - nie da sie jej porownac
z oryginalem piksel w piksel. Najpierw dopasowujemy ja do oryginalu
(cechy SIFT, awaryjnie ORB; RANSAC; przeksztalcenie podobienstwa), potem liczymy:

  zgodne_pary  - ile dopasowan potwierdza przeksztalcenie (wiarygodnosc)
  skala        - ~1.0 oznacza, ze mozaika ma skale oryginalu
  pokrycie     - jaka czesc obszaru przeleconego trafila do mozaiki
  blad_sredni  - srednia roznica jasnosci w czesci wspolnej (0-255)
  psnr_db      - to samo jako PSNR; powyzej ~25 dB to dobra zgodnosc
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)



def _szary(img, max_wymiar):
    """Przygotowuje obraz do szukania punktow: szarosc i zmniejszenie.

        Po co: ocena porownuje mozaike z oryginalem przez punkty charakterystyczne
        (SIFT / ORB), a te szukaja zmian jasnosci - kolor nie jest potrzebny.
        Mozaika i oryginal maja po kilka tysiecy px; zmniejszenie do max_wymiar
        przyspiesza szukanie punktow wielokrotnie, a do oceny wystarcza.
        Nigdy nie powieksza (k najwyzej 1). INTER_AREA - przy zmniejszaniu usrednia
        piksele, bez mory na rzedach uprawy.

        Argumenty:
            img        - obraz BGR (mozaika albo wycinek oryginalu)
            max_wymiar - najwiekszy dozwolony dluzszy bok w px (domyslnie 1600 z ocen())

        Zwraca:
            (g, k):
              g - obraz szary (uint8), dluzszy bok najwyzej max_wymiar
              k - wspolczynnik zmniejszenia (1.0 = bez zmian), np. mozaika
                  2163 x 1552 przy 1600 -> k = 0.74
              k jest potrzebne, zeby wspolrzedne punktow znalezionych na
              zmniejszonym obrazie przeliczyc z powrotem na pelny rozmiar (/ k).

        Uzywana w:
            ocen() (evaluation.py) - dla mozaiki i dla wycinka oryginalu;
            potem: src = punkty_mozaiki / km, dst = punkty_oryginalu / ko
        """
    k = min(1.0, max_wymiar / max(img.shape[:2]))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if k < 1.0:
        g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return g, k


def _pary_wzajemne(d1, d2, norm):
    """Dopasowuje punkty mozaiki do punktow oryginalu - tylko pary pewne w obie strony.

    Po co: rzedy uprawy sa powtarzalne, wiec wiele punktow mozaiki wyglada
    jak jeden i ten sam punkt oryginalu. Bez kontroli kilkadziesiat punktow
    mozaiki "trafia" w jeden punkt oryginalu, a RANSAC wybiera wtedy model
    zdegenerowany - sciagniecie calej mozaiki do jednego punktu (skala ~0).
    Dwa sita:
        1. test proporcji (0.8) - najlepszy kandydat wyraznie lepszy od drugiego
        2. wzajemnosc - punkt A mozaiki wybral B w oryginale I punkt B oryginalu,
           szukajac w mozaice, tez wybral A (cross-check)
    Test proporcji 0.8 jest lagodniejszy niz 0.75 w skladaczu, bo drugie sito
    i tak wycina niejednoznaczne pary.

    Argumenty:
        d1   - deskryptory punktow mozaiki
        d2   - deskryptory punktow (wycinka) oryginalu
        norm - jak porownywac: cv2.NORM_L2 (SIFT) albo cv2.NORM_HAMMING (ORB)

    Zwraca:
        lista cv2.DMatch - tylko pary wzajemne; queryIdx = numer punktu mozaiki,
        trainIdx = numer punktu oryginalu

    Uzywana w:
        ocen() (evaluation.py) - najpierw dla SIFT, gdy par < 40 - jeszcze dla ORB;
        wygrywa zestaw z wieksza liczba par
    """
    bf = cv2.BFMatcher(norm)
    # najlepsze dopasowanie w przeciwna strone: punkt oryginalu -> punkt mozaiki
    wstecz = {m.queryIdx: m.trainIdx for m in bf.match(d2, d1)}
    pary = []
    for para in bf.knnMatch(d1, d2, k=2):
        if len(para) == 2 and para[0].distance < 0.8 * para[1].distance:
            # para wzajemna: A wybralo B i B wybralo A
            if wstecz.get(para[0].trainIdx) == para[0].queryIdx:
                pary.append(para[0])
    return pary


def _obszar_lotu(prawda, W, H, zapas=0.05):
    """Wyznacza prostokat oryginalu, nad ktorym faktycznie przelecial dron.

        Po co: ocena porownuje mozaike tylko z PRZELECIANYM fragmentem pola.
        Na calym oryginale powtarzalne rzedy uprawy daja mnostwo niejednoznacznych
        punktow i test proporcji odrzucilby prawie wszystkie pary; poza tym
        fragmentow, nad ktorymi dron nie lecial, i tak nie ma na mozaice.
        Jak liczy: wszystkie rogi wszystkich kadrow z prawda.json -> najmniejsze
        i najwieksze x, y (obwiednia) -> poszerzenie o zapas z kazdej strony ->
        przyciecie do granic obrazu.

        Argumenty:
            prawda - slownik z prawda.json (uzywa: prawda["kadry"][...]["naroza"])
            W, H   - szerokosc i wysokosc oryginalu (px)
            zapas  - poszerzenie z kazdej strony jako ulamek rozmiaru obszaru (0.05 = 5%)

        Zwraca:
            (x0, y0, x1, y1) - liczby calkowite, prostokat w px oryginalu
            np. lot z zakladki 1 (55 kadrow = 220 rogow, pole 2400 x 1600):
                rogi od (45, 28) do (2243, 1551) + 5% -> (0, 0, 2353, 1600)

        Uzywana w:
            ocen() (evaluation.py) - wycinek = oryginal[y0:y1, x0:x1]; punkty wycinka
            przesuwane potem o (x0, y0) z powrotem do ukladu oryginalu
        """
    pts = np.array([p for k in prawda["kadry"] for p in k["naroza"]], float)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    dx, dy = (x1 - x0) * zapas, (y1 - y0) * zapas
    return (int(max(0, x0 - dx)), int(max(0, y0 - dy)),
            int(min(W, x1 + dx)), int(min(H, y1 + dy)))


def ocen(mozaika, oryginal, prawda=None, outdir=None, max_wymiar=1600):
    """Sprawdza, jak dobrze mozaika odtwarza oryginalne pole.

        Po co: zakladka 3 sklada kadry z symulacji, a my znamy oryginal, z ktorego
        je wycieto - mozna wiec zmierzyc jakosc skladania. Mozaika ma jednak wlasny
        uklad wspolrzednych, wiec nie da sie porownac jej piksel w piksel - najpierw
        trzeba ja do oryginalu dopasowac.
        Kroki:
            1. z prawdy: tylko przeleciany fragment oryginalu (_obszar_lotu)
            2. oba obrazy szare i zmniejszone do max_wymiar (_szary)
            3. punkty SIFT (8000), pary wzajemne (_pary_wzajemne); gdy < 40 par -
               drugi raz ORB o niskim progu; wygrywa zestaw z wieksza liczba par
            4. RANSAC: przeksztalcenie mozaika -> oryginal (obrot + skala + przesuniecie);
               skala < 0.2 = dopasowanie zdegenerowane -> ok False
            5. mozaika przeniesiona do ukladu oryginalu (warpAffine) + maski:
               gdzie sa dane mozaiki (m_moz) i gdzie lecial dron (m_lot, z rogow kadrow)
            6. miary na czesci wspolnej: pokrycie, sredni blad jasnosci, PSNR
            7. opcjonalnie zapis podgladow do outdir

        Argumenty:
            mozaika    - mozaika (BGR, uint8) z GlobalStitcher / Detail / Basic
            oryginal   - zdjecie pola, z ktorego wycieto kadry (BGR)
            prawda     - slownik z prawda.json; None = porownanie z calym oryginalem
                         (gorzej - powtarzalne rzedy, brak maski przelotu)
            outdir     - folder na podglady i ocena.json; None = bez zapisu
            max_wymiar - do ilu px zmniejszac obrazy przy szukaniu punktow (1600)

        Zwraca:
            slownik:
              ok          - True / False (False + "powod", gdy nie da sie dopasowac)
              zgodne_pary - ile par potwierdza dopasowanie (wiarygodnosc)
              skala       - ~1.0 = mozaika w skali oryginalu
              pokrycie    - jaka czesc przelecianego terenu jest na mozaice (0-1)
              blad_sredni - srednia roznica jasnosci w czesci wspolnej (0-255)
              psnr_db     - zgodnosc pikseli w dB; powyzej ~25 dobrze
              werdykt     - napis z werdykt()
            np. lot z zakladki 1 (55 kadrow, global):
              {"ok": True, "zgodne_pary": 235, "skala": 1.0185, "pokrycie": 0.9991,
               "blad_sredni": 14.74, "psnr_db": 23.27,
               "werdykt": "DOBRZE - drobne niedokladnosci na szwach"}
        Zapisuje w outdir (gdy podany):
            ocena_nalozenie.jpg - przygaszony oryginal z nalozona mozaika
            ocena_roznica.jpg   - mapa roznic (niebieski = zgodnie, czerwony = duza roznica)
            ocena.json          - slownik wyniku

        Uzywana w:
            app.py, zakladka 3 (linia 424) - ocena po skladaniu; liczby i werdykt
                                             pod mozaika, podglady w zakladce
            cli.py - polecenie ocen
        """
    ox, oy = 0, 0
    wycinek = oryginal
    if prawda and prawda.get("kadry"):
        x0, y0, x1, y1 = _obszar_lotu(prawda, oryginal.shape[1], oryginal.shape[0])
        wycinek = oryginal[y0:y1, x0:x1]
        ox, oy = x0, y0

    gm, km = _szary(mozaika, max_wymiar)
    go, ko = _szary(wycinek, max_wymiar)

    # Najpierw SIFT (odporny na lekkie rozmycie mozaiki od mieszania kadrow),
    # a gdy znajdzie za malo par - ORB o niskim progu, ktory na drobnej
    # teksturze gleby znajduje wielokrotnie wiecej punktow.
    pary = kp1 = kp2 = None
    for det, norm in ((cv2.SIFT_create(nfeatures=8000), cv2.NORM_L2),
                      (cv2.ORB_create(nfeatures=8000, fastThreshold=5), cv2.NORM_HAMMING)):
        k1, d1 = det.detectAndCompute(gm, None)
        k2, d2 = det.detectAndCompute(go, None)
        if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
            continue
        p = _pary_wzajemne(d1, d2, norm)
        if pary is None or len(p) > len(pary):
            pary, kp1, kp2 = p, k1, k2
        if len(pary) >= 40:
            break
    if pary is None:
        return {"ok": False, "powod": "brak cech na obrazach"}
    if len(pary) < 12:
        return {"ok": False, "powod": f"za malo dopasowan ({len(pary)})"}

    # Wspolrzedne z powrotem do pelnej rozdzielczosci (/k), a punkty oryginalu
    # przesuniete o poczatek wycinka (ox, oy) - bo szukalismy tylko w obszarze lotu.
    src = np.float32([kp1[m.queryIdx].pt for m in pary]) / km
    dst = np.float32([kp2[m.trainIdx].pt for m in pary]) / ko + np.float32([ox, oy])
    # przeksztalcenie mozaika -> oryginal
    A, maska = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                           ransacReprojThreshold=4.0)
    if A is None:
        return {"ok": False, "powod": "nie udalo sie dopasowac mozaiki do oryginalu"}

    zgodne = int(maska.sum())
    skala = float(np.hypot(A[0, 0], A[1, 0]))
    # Skala bliska zera = RANSAC wybral model 'sciagnij wszystko w jeden punkt',
    # ktory formalnie pasuje do wielu blednych par. To nie jest prawdziwe dopasowanie.
    if skala < 0.2:
        return {"ok": False, "powod": "dopasowanie zdegenerowane - mozaika nie "
                "przypomina oryginalu", "zgodne_pary": zgodne}

    # mozaika przeniesiona do ukladu oryginalu
    H, W = oryginal.shape[:2]
    przen = cv2.warpAffine(mozaika, A, (W, H), flags=cv2.INTER_LINEAR)
    m_moz = cv2.warpAffine(np.full(mozaika.shape[:2], 255, np.uint8), A, (W, H),
                           flags=cv2.INTER_NEAREST) > 0
    # czarne tlo mozaiki (brak danych) nie liczy sie do porownania
    m_moz &= przen.sum(axis=2) > 0           # czarne tlo mozaiki to brak danych

    # obszar, ktory faktycznie zostal przeleciany - z prawdy
    if prawda:
        m_lot = np.zeros((H, W), np.uint8)
        for k in prawda["kadry"]:
            cv2.fillPoly(m_lot, [np.array(k["naroza"], np.int32)], 255)
        m_lot = m_lot > 0
    else:
        m_lot = np.ones((H, W), bool)

    wspolne = m_moz & m_lot
    pokrycie = float(wspolne.sum() / max(1, m_lot.sum()))

    # Roznica jasnosci piksel po pikselu, usredniona po 3 kanalach. int16, bo przy
    # uint8 odejmowanie 'zawinieloby sie' (np. 10 - 20 = 246).
    roznica = np.abs(przen.astype(np.int16) - oryginal.astype(np.int16)).mean(axis=2)
    blad = float(roznica[wspolne].mean()) if wspolne.any() else float("nan")
    mse = float((roznica[wspolne] ** 2).mean()) if wspolne.any() else float("nan")
    # PSNR = 10 * log10(MAX^2 / MSE). MAX = 255 dla 8 bitow. Kazde +6 dB to
    # mniej wiecej dwukrotnie mniejszy sredni blad.
    psnr = float(10 * np.log10(255 ** 2 / mse)) if mse and mse > 0 else float("inf")

    wynik = {
        "ok": True,
        "zgodne_pary": zgodne,
        "skala": round(skala, 4),
        "pokrycie": round(pokrycie, 4),
        "blad_sredni": round(blad, 2),
        "psnr_db": round(psnr, 2),
        "werdykt": werdykt(zgodne, skala, pokrycie, psnr),
    }

    if outdir:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        # podglad: oryginal przygaszony, mozaika nalozona, roznica obok
        # podglad: przygaszony oryginal, na nim mozaika - od razu widac przesuniecia
        tlo = (oryginal * 0.35).astype(np.uint8)
        tlo[m_moz] = przen[m_moz]
        # mapa roznic w kolorach (niebieski = zgodnie, czerwony = duza roznica)
        mapa = cv2.applyColorMap(np.clip(roznica * 4, 0, 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)
        mapa[~wspolne] = 0
        cv2.imwrite(str(outdir / "ocena_nalozenie.jpg"), tlo, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(outdir / "ocena_roznica.jpg"), mapa, [cv2.IMWRITE_JPEG_QUALITY, 85])
        (outdir / "ocena.json").write_text(json.dumps(wynik, indent=2), encoding="utf-8")
    return wynik


def werdykt(zgodne, skala, pokrycie, psnr):
    """Zamienia liczby z oceny mozaiki na jedno zdanie: ZLE / CZESCIOWO / SLABO / DOBRZE / BARDZO DOBRZE.

    Po co: zeby w aplikacji od razu bylo widac, czy mozaika wyszla, bez czytania liczb.
           Werdykt opiera sie glownie na skali i PSNR. Liczba par mowi tylko, na ile ocena
           jest pewna - mala mozaika albo powtarzalne rzedy daja malo par nawet przy
           poprawnym zlozeniu.

    Argumenty:
        zgodne   - liczba par punktow mozaika <-> oryginal po RANSAC, np. 235
        skala    - skala mozaiki wzgledem oryginalu (1.0 = idealnie), np. 1.0185
        pokrycie - jaka czesc obszaru lotu pokrywa mozaika, 0-1, np. 0.9991
        psnr     - podobienstwo mozaiki do oryginalu w dB (wiecej = lepiej), np. 23.27

    Zwraca:
        str, np. "DOBRZE - drobne niedokladnosci na szwach"
        (z dopiskiem "  (niska pewnosc oceny - malo par)", gdy zgodne < 20)

    Uzywana w:
        ocen() (stitchlive/evaluation.py) - pole "werdykt" w wyniku oceny
    """
    if zgodne < 8 or not (0.85 < skala < 1.15):
        return "ZLE - mozaika nie odpowiada oryginalowi"
    dopisek = "  (niska pewnosc oceny - malo par)" if zgodne < 20 else ""
    if pokrycie < 0.7:
        return "CZESCIOWO - czesc kadrow nie trafila do mozaiki" + dopisek
    if psnr < 20:
        return "SLABO - widoczne przesuniecia lub zjawy" + dopisek
    if psnr < 25:
        return "DOBRZE - drobne niedokladnosci na szwach" + dopisek
    return "BARDZO DOBRZE" + dopisek