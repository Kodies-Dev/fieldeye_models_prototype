"""Skladacz 'global' - pary z OpenCV, pozycje kadrow z jednego ukladu rownan.

Dlaczego osobna metoda
----------------------
Pipeline cv2.detail (i cv2.Stitcher w trybie SCANS) przy kadrach obroconych
wzgledem siebie o kilka stopni daje wyraznie rozmyta mozaike, mimo ze
pojedyncze pary kadrow dopasowuja sie z dokladnoscia subpikselowa.
Bez obrotu problem nie wystepuje - same przesuniecia sumuja sie poprawnie.

Tutaj pary liczy OpenCV (ORB + test proporcji + RANSAC), a polozenie
wszystkich kadrow wyznaczamy naraz, metoda najmniejszych kwadratow.

Model: przeksztalcenie podobienstwa (obrot + skala + przesuniecie)

    S(p) = [a -b] p + [tx]
           [b  a]     [ty]

jest LINIOWE w parametrach (a, b, tx, ty). Kazda para punktow p (kadr i)
i q (kadr j), ktore przedstawiaja ten sam punkt terenu, daje dwa rownania
liniowe S_i(p) - S_j(q) = 0. Kadr 0 ustalamy jako uklad odniesienia.
Rozwiazanie jest globalne - blad nie kumuluje sie wzdluz trasy.
"""

import logging
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)


# =====================================================================
# Glowna klasa metody 'global'. Uzycie:
#     s = GlobalStitcher()
#     for sciezka in zdjecia:
#         s.dodaj(sciezka)        # True, gdy mozaika zostala odswiezona
#     s.zakoncz()                 # koncowe renderowanie
#     s.mozaika                   # gotowy obraz (tablica BGR)
# =====================================================================
class GlobalStitcher:


    def __init__(self, work_megapix=0.6, zakres_par=0, min_zgodnych=15,
                 pkt_na_pare=40, wyrownaj_ekspozycje=True, szwy=False,
                 renderuj_co=1, detektor="SIFT", okno=0, cechy=1500,
                 bloki_ekspozycji=False):
        """

                Argumenty:
                    work_megapix  - rozdzielczosc robocza w Mpx; kadry sa do niej zmniejszane
                                    (mniej = szybciej, wiecej = dokladniej); suwak 0.1-2.0, dom. 0.6
                    zakres_par    - z iloma poprzednimi kadrami dopasowywac nowy; 0 = ze wszystkimi
                                    (dokladniej, ale wolniej przy dlugich lotach)
                    min_zgodnych  - najmniej par punktow, zeby uznac dwa kadry za polaczone (15)
                    pkt_na_pare   - ile punktow z kazdej pary trafia do ukladu rownan (40);
                                    wiecej nie poprawia wyniku, a spowalnia
                    wyrownaj_ekspozycje - wyrownanie jasnosci kadrow przy rysowaniu (GainCompensator)
                    szwy          - wyznaczanie szwow (GraphCut): ladniej, wolniej; checkbox w zakladce 3
                    renderuj_co   - co ile kadrow przeliczac i rysowac mozaike (1 = po kazdym);
                                    suwak "Odswiezaj mozaike co N kadrow", dom. 3
                    detektor      - "ORB" (szybki, binarny) albo "SIFT" (wolniejszy, dokladniejszy)
                    okno          - NIEUZYWANY tutaj; jest tylko po to, zeby wszystkie skladacze
                                    przyjmowaly te same argumenty (uzywa go DetailStitcher)
                    cechy         - ile punktow charakterystycznych szukac na kadrze (1500)
                    bloki_ekspozycji - blokowe wyrownanie jasnosci (dokladniejsze, duzo wolniejsze)


                Uzywana w:
                    app.py, zakladka 3 (linia 384) - GlobalStitcher(work_megapix, zakres_par,
                                                      szwy, renderuj_co) z suwakow
                    app.py, zakladka 4 (linia 711) - renderuj_co=10**6 (rysuje tylko na koncu)
                    cli.py (linia 29)              - polecenia batch / watch / demo

                    Stan skladacza (atrybuty) - rosnie z kazdym dodanym kadrem:
            sciezki       sciezki dodanych plikow, w kolejnosci dodania (indeks = numer kadru)
            obrazy        kadry zmniejszone do rozdzielczosci roboczej (do rysowania mozaiki)
            kp            punkty charakterystyczne kazdego kadru: tablica (N, 2) wspolrzednych x, y
            des           deskryptory tych punktow (opis otoczenia punktu, do porownywania kadrow)
            pary          dopasowane kadry: (i, j) -> (punkty_i, punkty_j) - te same miejsca pola
                          na kadrze i oraz j; tylko pary, ktore przeszly RANSAC
            S             polozenie kadrow na mozaice po _rozwiaz(): {numer_kadru: (a, b, tx, ty)};
                          kadr bez polaczenia z reszta nie ma wpisu; None = jeszcze nie liczone
            mozaika       gotowa mozaika (BGR, uint8); None = jeszcze nie narysowana
            historia      lista (liczba_kadrow, czas_dodaj_s, czy_odswiezono) - do raport_czasu()
            ostatni_czas  ile sekund trwalo ostatnie dodaj()
            _licznik      ile razy wywolano dodaj(); mozaika odswiezana, gdy licznik % renderuj_co == 0
            _aktualna     czy mozaika uwzglednia wszystkie kadry; False = zostaly nienarysowane
                          (renderuj_co > 1) i zakoncz() musi narysowac jeszcze raz
                """
        self.work_megapix = work_megapix
        self.zakres_par = zakres_par
        self.min_zgodnych = min_zgodnych
        self.pkt_na_pare = pkt_na_pare
        self.wyrownaj = wyrownaj_ekspozycje
        self.szwy = szwy
        self.renderuj_co = max(1, renderuj_co)
        self.okno = okno
        self.bloki = bloki_ekspozycji


        self.det = (cv2.ORB_create(nfeatures=cechy) if detektor == "ORB"
                    else cv2.SIFT_create(nfeatures=cechy))
        # zapasowy detektor o nizszym progu - dla kadrow rozmytych (drgania, ruch),
        # na ktorych domyslny prog FAST nie znajduje prawie niczego
        self.det_czuly = (cv2.ORB_create(nfeatures=cechy, fastThreshold=5) if detektor == "ORB"
                          else cv2.SIFT_create(nfeatures=cechy, contrastThreshold=0.01))
        # CLAHE: lokalne wyrownanie kontrastu w siatce 8x8 fragmentow. clipLimit
        # ogranicza wzmocnienie, zeby nie podbic szumu na jednolitej glebie.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Miara podobienstwa deskryptorow: ORB daje deskryptory binarne (porownanie
        # = liczba roznych bitow, Hamming), SIFT - wektory liczb (odleglosc euklidesowa).
        self.norm = cv2.NORM_HAMMING if detektor == "ORB" else cv2.NORM_L2

        # Stan skladacza - rosnie z kazdym kadrem:
        #   obrazy  kadry w rozdzielczosci roboczej (potrzebne do renderowania)
        #   kp      wspolrzedne punktow charakterystycznych kazdego kadru
        #   des     ich deskryptory (do porownywania miedzy kadrami)
        self.sciezki, self.obrazy, self.kp, self.des = [], [], [], []
        self.pary = {}          # (i, j) -> (pkt_i, pkt_j) punkty zgodne
        self.S = None           # parametry podobienstwa, ksztalt (n, 4)
        self.mozaika = None
        self.historia = []
        self.ostatni_czas = 0.0
        # ile razy wywolano dodaj() - steruje odswiezaniem co renderuj_co
        self._licznik = 0
        # czy mozaika uwzglednia wszystkie dotychczasowe kadry
        self._aktualna = True

    # ---------------- dodawanie ----------------


    def _przeskaluj(self, img):
        """Zmniejsza kadr tak, zeby mial okolo work_megapix milionow pikseli.

                Po co: szukanie punktow, dopasowanie par i rysowanie mozaiki sa tym
                wolniejsze, im wiekszy kadr. Do skladania wystarcza ok. 0,6 Mpx -
                polozenie kadrow i tak liczy sie z dokladnoscia ulamka piksela.
                Pierwiastek, bo liczba pikseli rosnie z KWADRATEM skali: zeby miec
                X razy mniej pikseli, kazdy bok zmniejszamy sqrt(X) razy.
                INTER_AREA - przy zmniejszaniu  usrednia piksele,

                Argumenty:
                    img - kadr (BGR, uint8) w pelnej rozdzielczosci

                Zwraca:
                    kadr zmniejszony (albo ten sam, jesli juz jest mniejszy), BGR, uint8:
                        480 x 480   (0,23 Mpx) -> bez zmian, k = 1
                        2902 x 2176 (6,3 Mpx)  -> 895 x 671, k = 0,31
                        4624 x 3472 (16 Mpx)   -> 894 x 671, k = 0,19 (pelna matryca kamery)

                Uzywana w:
                    dodaj() (global_stitcher.py) - kazdy nowy kadr jest najpierw
                    zmniejszany, potem szukane sa na nim punkty i trafia do self.obrazy
                """
        k = min(1.0, np.sqrt(self.work_megapix * 1e6 / (img.shape[0] * img.shape[1]))) #skala
        return img if k >= 1.0 else cv2.resize(img, None, fx=k, fy=k,
                                               interpolation=cv2.INTER_AREA)

    # Dodaje jeden kadr. Kolejnosc: wczytanie -> punkty charakterystyczne ->
    # dopasowanie do wczesniejszych kadrow -> (co renderuj_co) rozwiazanie ukladu
    # i narysowanie mozaiki. Zwraca True, gdy mozaika zostala odswiezona.
    def dodaj(self, sciezka):
        """Dodaje jeden kadr: szuka na nim punktow, dopasowuje do wczesniejszych, co jakis czas odswieza mozaike.

                Po co: glowna metoda skladacza - wołana dla kazdego kadru po kolei, jak
                zdjecia z lecacego drona. Mozaika rosnie przyrostowo.
                Kroki:
                    1. wczytanie i zmniejszenie do ok. work_megapix (_przeskaluj)
                    2. CLAHE na szarym obrazie - podbija kontrast w ciemnych rogach (winieta),
                       czyli tam, gdzie kadry na siebie zachodza
                    3. punkty charakterystyczne + deskryptory (ORB); gdy mniej niz 200 -
                       druga proba czulszym detektorem (kadr rozmyty)
                    4. mniej niz min_zgodnych punktow -> kadr pominiety
                    5. zapis kadru, punktow (same x, y) i deskryptorow do stanu
                    6. dopasowanie do poprzednich kadrow (_dopasuj_pare); z zakres_par > 0
                       tylko do ostatnich N
                    7. co renderuj_co kadrow: polozenie wszystkich kadrow (_rozwiaz)
                       i narysowanie mozaiki (_renderuj)

                Argumenty:
                    sciezka - sciezka do pliku kadru; musi byc Path (kod uzywa sciezka.name)

                Zwraca:
                    True  - mozaika zostala w tym wywolaniu odswiezona (self.mozaika aktualna)
                    False - mozaika NIE odswiezona, czyli jedno z:
                              - nie da sie wczytac pliku albo za malo punktow (kadr pominiety)
                              - to nie jest kadr odswiezania (renderuj_co > 1) - kadr JEST
                                dodany, tylko jeszcze nienarysowany
                              - _rozwiaz() albo _renderuj() sie nie udaly
                    np. 55 kadrow, renderuj_co 3 -> True przy kadrach 3, 6, ... 54 (18 razy);
                    reszte dorysowuje zakoncz()

                Uzywana w:
                    app.py, zakladka 3 (linia 399) - dla kazdego kadru; przy True pokazuje mozaike
                    app.py, zakladka 4 (linia 736) - lot z detekcja
                    cli.py - polecenia batch / watch / demo
                """
        img = cv2.imread(str(sciezka))
        if img is None:
            log.warning("Nie mozna wczytac: %s", sciezka)
            return False
        img = self._przeskaluj(img)
        self._licznik += 1

        t0 = time.time()

        szary = self.clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) #szary moz byc bo detektor szuka zmian jasnoci
        # detectAndCompute: jednoczesnie znajduje punkty (kp) i liczy ich deskryptory (des).
        kp, des = self.det.detectAndCompute(szary, None)
        #z czulszym detektorem
        if des is None or len(kp) < 200:
            kp2, des2 = self.det_czuly.detectAndCompute(szary, None)
            if des2 is not None and len(kp2) > (0 if kp is None else len(kp)):
                log.debug("%s: kadr malo ostry, prog obnizony (%d -> %d punktow)",
                          sciezka.name, 0 if kp is None else len(kp), len(kp2))
                kp, des = kp2, des2
        # Nawet po drugiej probie za malo - kadr jest bezuzyteczny, pomijamy go.
        if des is None or len(kp) < self.min_zgodnych:
            log.warning("Pominieto %s: za malo punktow (%d)", sciezka.name,
                        0 if kp is None else len(kp))
            return False

        n = len(self.obrazy)
        self.sciezki.append(sciezka)
        self.obrazy.append(img)

        self.kp.append(np.float32([k.pt for k in kp])) #lista wspołrzędnych
        self.des.append(des) #deskryptory punktow

        # z iloma wczesnijszymi porownujemy jesli 0 to ze wszytkimi domylnei
        od = max(0, n - self.zakres_par) if self.zakres_par else 0
        nowe = 0
        for j in range(od, n):
            if self._dopasuj_pare(j, n):
                nowe += 1 #prubuje dopasowac pare do kazdego wczesniejszego
        # Kadr bez zadnej pary nie trafi do mozaiki (chyba ze kolejne kadry go polacza).
        if n > 0 and nowe == 0:
            log.warning("%s: brak polaczenia z wczesniejszymi kadrami", sciezka.name)

        self._aktualna = False
        # Odswiezamy mozaike tylko co renderuj_co kadrow - samo dopasowanie par jest
        # tanie, rozwiazanie ukladu i renderowanie juz nie.
        if len(self.obrazy) < 2 or self._licznik % self.renderuj_co:
            return False

        ok = self._rozwiaz() and self._renderuj() #liczy polozenie kadrow i rysuje z nich mozaike
        self._aktualna = ok
        self.ostatni_czas = time.time() - t0
        self.historia.append((len(self.obrazy), round(self.ostatni_czas, 2), ok)) #ile kadrow czas czy sie udalo
        return ok

    # Probuje dopasowac kadr i z kadrem j. Przy sukcesie zapisuje pary punktow
    # w self.pary[(i, j)] i zwraca True.
    def _dopasuj_pare(self, i, j):
        """Sprawdza, czy kadry i oraz j pokazuja wspolny kawalek pola; jesli tak - zapisuje pare punktow.

                Po co: z takich par _rozwiaz() liczy potem polozenie wszystkich kadrow.
                Para musi byc pewna - jedna bledna para potrafi rozjechac cala mozaike,
                dlatego jest kilka sit:
                    1. knnMatch (k=2)  - dla kazdego punktu z i dwa najpodobniejsze z j
                    2. test proporcji  - najlepszy wyraznie (o 25%) lepszy od drugiego;
                                         odrzuca niejednoznaczne miejsca (powtarzalne rzedy)
                    3. RANSAC          - szuka obrotu + skali + przesuniecia, z ktorym zgadza
                                         sie najwiecej par (do 3 px); reszta to pomylki
                    4. kontrola        - co najmniej min_zgodnych par zgodnych, co najmniej
                                         20% "dobrych", skala miedzy 0.7 a 1.4 (dron nie zmienia
                                         wysokosci o 40% miedzy kadrami)
                Z par zgodnych zostaje najwyzej pkt_na_pare punktow (rownomierna probka) -
                wiecej nie poprawia ukladu rownan, a spowalnia.

                Argumenty:
                    i, j - numery kadrow (indeksy w self.kp / self.des), i < j

                Zwraca:
                    True  - kadry pasuja; self.pary[(i, j)] = (p, q): tablice (K, 2) tych samych
                            miejsc pola na kadrze i (p) i j (q), K <= pkt_na_pare
                    False - nie pasuja (za malo punktow, RANSAC nic nie znalazl, zla skala)

                    Przyklad (kadry 1 i 2 z symulacji, sasiednie):
                        1500 i 1217 punktow -> 54 po tescie proporcji -> 23 zgodne z RANSAC,
                        skala 1.07, obrot -10.9 st. -> True
                    Kadry 1 i 30 (inne miejsce pola):
                        33 przypadkowe "dobre" -> tylko 4 zgodne -> False

                Uzywana w:
                    dodaj() (global_stitcher.py) - nowy kadr kolejno z kazdym wczesniejszym
                """

        # (BFMatcher = porownanie kazdy z kazdym, knn k=2 = dwoch najlepszych).
        m = cv2.BFMatcher(self.norm).knnMatch(self.des[i], self.des[j], k=2) #zwraca dwa najprawdopodobiejszych kandydatow punktow z j dla jednego punktu z i DMatch(queryIdx=0,   trainIdx=1202, distance=70)
        # Test proporcji: para tylko wtedy, gdy najlepszy kandydat jest wyraznie
        # (o 25%) lepszy od drugiego. Odrzuca niejednoznaczne miejsca, np. powtarzalne rzedy.
        dobre = []
        for q in m:  # q = (najlepszy, drugi najlepszy punkt) dla jednego punktu z kadru i
            if len(q) != 2:  # czasem jest tylko 1 kandydat - nie da sie porownac
                continue
            a, b = q  # a = najlepszy, b = drugi
            if a.distance < 0.75 * b.distance: #jak bardzo roznia sie deskryptory i od j
                dobre.append(a)  # zostaje tylko najlepszy
        if len(dobre) < self.min_zgodnych:
            return False

        # wspolrzedne dopasowanych punktow: p w kadrze i, q w kadrze j
        p = self.kp[i][[d.queryIdx for d in dobre]] #punkty z i
        q = self.kp[j][[d.trainIdx for d in dobre]] #dopasowane punkty z j
        # RANSAC: szuka przeksztalcenia podobienstwa (obrot + skala + przesuniecie)
        # wspieranego przez jak najwiecej par. maska = ktore pary sie z nim zgadzaja
        # z dokladnoscia do 3 px (to one sa 'zgodne').
        E, maska = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC,
                                               ransacReprojThreshold=3.0) #E to jest macierz 2 x 3 mowiaca jak przeliczyc i na j
        #     E = [[ s*cos  -s*sin  tx ],      np. kadry 1-2:  [[ 1.052   0.202  -227.6 ],
        #          [ s*sin   s*cos  ty ]]
        #maska mowi ktore pary zgodne ktore nie

        if E is None:
            return False
        maska = maska.ravel().astype(bool) #splaczza tablice na jednowymiarowwaca
        zgodne = int(maska.sum()) #liczy ile pkt bylo zgodnych

        # zabezpieczenia przed bledna para: za malo poparcia albo absurdalna skala
        # Skala z macierzy: E = [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]], wiec
        # s = sqrt(E00^2 + E10^2). Dron nie zmienia wysokosci o 40% miedzy kadrami -
        # taka skala oznacza bledne dopasowanie.
        skala = float(np.hypot(E[0, 0], E[1, 0]))
        if zgodne < self.min_zgodnych or zgodne < 0.2 * len(dobre) or not 0.7 < skala < 1.4:
            return False

        # zostawiamy tylko pary zgodne z RANSAC
        p, q = p[maska], q[maska]
        if len(p) > self.pkt_na_pare: #najwiecej par punktow z jednej pary kadrow domyslie 40                 # rownomierna probka punktow
            wyb = np.linspace(0, len(p) - 1, self.pkt_na_pare).astype(int) #bierzerównomiernie 40 pkt
            p, q = p[wyb], q[wyb]
        self.pary[(i, j)] = (p, q) #zwraca pary pkt kadrów (0, 1):  (p, q)
        return True

    # ---------------- globalne wyrownanie ----------------

    # Graf: wezly = kadry, krawedzie = udane pary. Szukamy najwiekszej spojnej
    # skladowej przeszukiwaniem w glab (stos). Tylko te kadry da sie ze soba
    # powiazac - reszta nie ma wspolnego ukladu wspolrzednych.
    def _skladowa(self):
        """Znajduje najwieksza grupe kadrow polaczonych ze soba parami.

        Po co: polozenie da sie policzyc tylko dla kadrow, ktore lacza sie
        (bezposrednio albo przez inne kadry) w jedna calosc. Kadr bez polaczenia
        (rozmyty, zrobiony przy starcie, nad innym miejscem) nie ma do czego sie
        "przyczepic" - nie trafi do mozaiki. Nie zakladamy, ze kadr 0 jest
        polaczony z reszta; ukladem odniesienia jest pierwszy kadr najwiekszej grupy.
        Jak liczy: kadry to wezly grafu, pary z self.pary to krawedzie;
        przeszukiwanie w glab (DFS, stos) zbiera kazda grupe polaczonych
        wezlow (skladowa spojna) i zostawia najwieksza.

        Argumenty:
            brak (korzysta z self.obrazy i self.pary)

        Zwraca:
            lista numerow kadrow najwiekszej grupy, posortowana rosnaco
            np. pary (0,1), (1,2), (3,4) przy 5 kadrach:
                grupy {0, 1, 2} i {3, 4} -> [0, 1, 2]
            w symulacji z zakladki 1 (55 kadrow): wszystkie, [0, 1, ..., 54]

        Uzywana w:
            _rozwiaz() (global_stitcher.py) - liczy polozenie tylko tych kadrow
        """
        n = len(self.obrazy)
        sasiedzi = {i: set() for i in range(n)} #pusty slownik obrazow z brakiem wartosci
        for i, j in self.pary: #przechodzi przez kadry
            sasiedzi[i].add(j) #kadry sa poloczone
            sasiedzi[j].add(i)
        najlepsza, widziane = [], set() #widziane mowi ze kadr trafil juz do jakiejss grupy
        for start in range(n):
            if start in widziane:
                continue
            grupa, stos = {start}, [start] #grupa to grupa kadrow ze soba zwiazanych, stos to te co aktualnie suzkamy czy maja jeszce nowych sasaidow
            while stos:
                for x in sasiedzi[stos.pop()]: #sprawdzamy wszystkich sasaidow tego gosia
                    if x not in grupa:
                        grupa.add(x) #dodajemy tych sasiadow jesli sa nowi
                        stos.append(x) #spradzamy sasiadow sasiadow
            widziane |= grupa # union zeby potem nei sprawdzac tych z wezyka
            if len(grupa) > len(najlepsza):
                najlepsza = sorted(grupa)
        return najlepsza


    def _rozwiaz(self):
        """Liczy polozenie WSZYSTKICH kadrow naraz ktore akutalnie sa dostepne - z jednego ukladu rownan.

        Po co: serce metody "global". Kazdy kadr ma 4 niewiadome (a, b, tx, ty) -
        obrot, skale i przesuniecie na mozaice (ta sama postac co w macierz_kadru).
        Niewiadome kadru (a, b, tx, ty):
            a = s * cos(kat), b = s * sin(kat)  - obrot i skala RAZEM (
                skala = sqrt(a^2 + b^2), kat = atan2(b, a)
                bez obrotu: b = 0, a = skala
            tx, ty                              - przesuniecie kadru na mozaice (px)
            np. kadr 1: a=0.916, b=0.178 -> skala 0.93, kat 11 st.

        Po co zapis przez a, b, a nie od razu kat i skala: z a, b rownania sa
        LINIOWE (a*x - b*y + tx), wiec caly uklad rozwiazuje jedno lstsq.
        Z katem jako niewiadoma pojawilyby sie cos(kat) i sin(kat) - uklad
        nieliniowy, trzeba by go rozwiazywac iteracyjnie, wolniej i mniej pewnie.
        Kazda para punktow z self.pary (p na kadrze i, q na kadrze j - ten sam
        punkt pola) daje 2 rownania: S_i(p) - S_j(q) = 0, czyli "po przeliczeniu
        na mozaike oba punkty maja wypasc w tym samym miejscu". Rownan jest
        duzo wiecej niz niewiadomych, wiec rozwiazanie metoda najmniejszych
        kwadratow (least squares) - najlepsze dopasowanie do wszystkich par naraz.
        Dzieki temu blad NIE narasta wzdluz trasy (jak przy dokladaniu kadr po kadrze).

        Kroki:
            1. najwieksza grupa polaczonych kadrow (_skladowa); jej pierwszy kadr
               = uklad odniesienia (a=1, b=0, tx=0, ty=0)
            2. do 8 przebiegow:
                 - kotwica skali i obrotu (_kotwica) - zeby sprzeczne pary nie
                   "zgodzily sie" przez sciesnienie kadrow do zera
                 - uklad rownan A x = b (_uklad) i najmniejsze kwadraty (lstsq)
                 - blad kazdej pary (_blad_pary): jak daleko od siebie laduja
                   jej punkty na mozaice
                 - najgorsza para <= 3 px -> koniec; inaczej odrzucenie par
                   z bledem > max(3, polowa najgorszego) i liczenie od nowa;
                   gdy graf sie rozpadnie - zostaje najwieksza grupa
            3. kontrola skali: kadr ze skala spoza 0.7-1.4 pominiety

        Argumenty:
            brak (korzysta z self.pary, self.obrazy)

        Zwraca:
            True  - umieszczono co najmniej 2 kadry; wynik w self.S
            False - za malo polaczonych kadrow
        Ustawia:
            self.S      {numer_kadru: (a, b, tx, ty)} - polozenie kazdego kadru na mozaice
            self.blad_px mediana bledu par w px (jakosc dopasowania, do raportu)

            Przyklad - lot z zakladki 1 (55 kadrow, 128 par):
                uklad 8222 rownan x 216 niewiadomych (54 kadry x 4), 1 przebieg
                (zadna para nie przekroczyla 3 px), mediana bledu 0.66 px;
                kadr 0: (1, 0, 0, 0), kadr 1: (0.916, 0.178, 223.8, -35.3)
                -> skala 0.93, obrot 11 st., przesuniecie (224, -35) px

        Uzywana w:
            dodaj() (global_stitcher.py) - co renderuj_co kadrow, przed _renderuj()
            zakoncz() (global_stitcher.py) - ostatnie przeliczenie po wszystkich kadrach
        """
        wezly = self._skladowa()
        if len(wezly) < 2: #wiecej niz 2 zdj
            return False
        # pozycja: numer kadru -> numer kolumny w ukladzie rownan. Pierwszy kadr
        # grupy dostaje pozycje 0 - to uklad odniesienia.
        pozycja = {k: i for i, k in enumerate(wezly)} #miejsce: numer kadru
        pary = {k: v for k, v in self.pary.items() if k[0] in pozycja and k[1] in pozycja} #bierze pary numery kadrow oraz ich punkty  wspolne jezeli nazeza do pozycji

        # Waga kotwicy (rownan "skala i obrot blisko oszacowania z _kotwica()").
        # Musi byc rzedu wspolczynnikow rownan punktow (setki px): za mala - kotwica
        # ignorowana i sprzeczne pary scisna kadry do zera; za duza - przeglosuje punkty.
        # Liczymy: 2 x mediana dluzszego boku kadru, np. kadry 480 px -> 960.
        waga = 2.0 * float(np.median([max(self.obrazy[k].shape[:2]) for k in wezly])) #jak wazna jest kotwica a kotwica to wstepne oszacowanie  skali i obrotu kazdego kadru

        bledy, S = {}, None # blad kazdej pary po rozwiazaniu , wynik układu równań: położenie każdego kadru na mozaice. rozwiązanie układu:  (a, b, tx, ty) dla kazdego kadru
        for przebieg in range(8):          #8 prób rozwiązania układu, bez najgorszych par z poprzedniej proby

            kotwica = self._kotwica(pary, wezly) #slownik numer kadru i jego oszacowany skala i obrot
            #   Zwraca: slownik {numer_kadru: a0 + b0*j} bez kadru odniesienia;
            #     a0 = skala*cos(kat) (.real), b0 = skala*sin(kat) (.imag)
            #     np. 1: 0.916+0.178j -> skala 0.93, kat 11 st.
            A, bvec = self._uklad(pary, pozycja, len(wezly), kotwica, waga)
            #   -----Jak wyglada A----
            #   czym jest A to macierz hdzie kolumny to niewiadome a wiersze to rownania opisujace tylko 2 kadry na wiersz jak sie do siebie odnosza
            #   KOLUMNY = niewiadome; kazdy kadr poza odniesieniem ma 4 kolumny:
            #             a1  b1  tx1 ty1 | a2  b2  tx2 ty2 | ...   (55 kadrow -> 54 * 4 = 216)
            #             '---- kadr 1 ---'   '---- kadr 2 ---'
            #
            #   WIERSZE = rownania. Para punktow p = (100, 50) na kadrze 1 i q = (30, 60)
            #             na kadrze 2 (to samo miejsce pola) daje 2 wiersze - "po przeliczeniu
            #             na mozaike p i q wypadaja w tym samym miejscu", osobno x i y:
            #
            #                  a1   b1  tx1 ty1   a2   b2  tx2 ty2   kadry 3..54
            #     wiersz x  [ 100, -50,  1,  0,  -30,  60, -1,  0,   0 ... 0 ]  = 0
            #     wiersz y  [  50, 100,  0,  1,  -60, -30,  0, -1,   0 ... 0 ]  = 0

            #   Do czego dalej:
            #     A, bvec --lstsq--> x (216 liczb) --> S: (a, b, tx, ty) kazdego kadru
            #     --> self.S --> _renderuj(): z (a, b, tx, ty) macierz 2x3 dla
            #     cv2.warpAffine, ktory kladzie kadr na mozaike.
            # Zwraca:
            #     A    - macierz wspolczynnikow: wiersz = jedno rownanie,
            #            kolumna = jedna niewiadoma (a, b, tx, ty kazdego kadru poza odniesieniem)
            #     bvec - prawa strona kazdego rownania (zera dla punktow, chyba ze
            #            w parze jest kadr odniesienia; waga*a0, waga*b0 dla kotwicy)
            #   ----- Jak wyglada bvec -----
            #   Czym jest: wektor "ile ma wyjsc" kazde rownanie z A - prawa strona ukladu
            #   A @ x = bvec. Jedna liczba na kazdy wiersz A (ta sama kolejnosc).
            #   Prawie wszedzie 0 ("punkty dwoch kadrow maja sie pokryc - roznica 0");
            #   rozne od zera tylko tam, gdzie wchodzi cos ZNANEGO:
            #     - kadr odniesienia (jego polozenie znamy, wiec ląduje po prawej stronie),
            #     - kotwica (docelowe a0, b0 razy waga).
            #   ----- Jak wyglada bvec -----
            #   bvec - wektor (1 wymiar), jedna liczba na kazdy wiersz A: prawa strona
            #   rownania. Ksztalt (8222,) - tyle, ile wierszy ma A.
            #
            #   1. Para punktow miedzy dwoma "zwyklymi" kadrami (np. 1 i 2) -> 0:
            #        100*a1 - 50*b1 + tx1 - 30*a2 + 60*b2 - tx2 = 0
            #
            #   2. Para z kadrem odniesienia (0) - jego (a, b, tx, ty) = (1, 0, 0, 0) sa ZNANE,
            #      wiec jego czesc nie jest niewiadoma, tylko liczba przeniesiona na prawa strone.
            #      np. p = (100, 50) na kadrze 0, q = (30, 60) na kadrze 1:
            #        wiersz x:  1*100 - 0*50 + 0  - (30*a1 - 60*b1 + tx1) = 0
            #                   ->  -30*a1 + 60*b1 - tx1 = -100      bvec = -100
            #        wiersz y:  0*100 + 1*50 + 0  - (60*a1 + 30*b1 + ty1) = 0
            #                   ->  -60*a1 - 30*b1 - ty1 = -50       bvec = -50
            #      (bez tego kadr 0 bylby niewiadoma i caly uklad moglby sie dowolnie
            #       przesunac / obrocic - kadr 0 ustala punkt zero mozaiki)
            #
            #   3. Wiersze kotwicy: waga * a0 i waga * b0
            #        np. kadr 1 (a0 = 0.916, b0 = 0.178, waga 960):  879.4  i  170.9
            #
            #   bvec = [0, 0, ..., -100, -50, ..., 0, 0, ..., 879.4, 170.9, ...]
            #   lstsq szuka x, przy ktorym A @ x jest jak najblizej bvec.
            #   Bez bvec rozwiazaniem byloby "wszystko = 0" (kadry sciagniete do punktu) -
            #   to znane liczby w bvec "zaczepiaja" mozaike i nadaja jej skale.
            x, *_ = np.linalg.lstsq(A, bvec, rcond=None) #obliczanie macierzy, roziazanie x czyli polozenie kadrow
            # kadr odniesienia nie jest niewiadoma - dokladamy go jako tozsamosc
            S = np.vstack([[1.0, 0.0, 0.0, 0.0], x.reshape(-1, 4)]) # plaski wektor zamieniony natable po 4 w w eirszu

            # Dla kazdej pary: jak daleko od siebie laduja jej punkty po przeksztalceniu.
            # Poprawna para - ulamek piksela; bledna (np. pomylony sasiedni rzad) - wiecej.
            bledy = {k: self._blad_pary(S[pozycja[k[0]]], S[pozycja[k[1]]], *v)
                     for k, v in pary.items()} #robi liste bledow par
            najgorszy = max(bledy.values()) if bledy else 0.0 # znajduje najwieksza
            if najgorszy <= 3.0:
                break #jezeli wsystkie pary pasuja do siebie blad ppniej 3 pixeli nie lecimy dalej
            # Najpierw najgorsze pary: dobre pary obok blednych tez maja podwyzszony
            # blad, dopoki bledne nie zostana usuniete.
            prog = max(3.0, 0.5 * najgorszy)
            log.debug("Odrzucam pary o bledzie > %.1f px", prog)
            for k in [k for k, e in bledy.items() if e > prog]:
                pary.pop(k)
            skl = self._skladowa_z(pary, wezly) #sprawdzamy czy graf po wyrzuceniu sie nie rozpadl
            if len(skl) < len(wezly):
                # graf sie rozpadl - zostaje najwieksza spojna grupa kadrow
                wezly = [k for k in wezly if k in skl]
                pozycja = {k: i for i, k in enumerate(wezly)} #nowa numeracja bez dziur
                pary = {k: v for k, v in pary.items() if k[0] in pozycja and k[1] in pozycja} #nowe pary
                if len(wezly) < 2:
                    return False


        dobre = [k for k in wezly if 0.7 <= float(np.hypot(*S[pozycja[k]][:2])) <= 1.4] #kadry o rozsadnej skali
        if len(dobre) < len(wezly):
            log.warning("Pomijam %d kadrow o nierealnej skali", len(wezly) - len(dobre))
        self.S = {k: S[pozycja[k]] for k in dobre}
        # mediana bledu par - miara jakosci dopasowania, trafia do raportu
        self.blad_px = float(np.median(list(bledy.values()))) if bledy else 0.0
        return len(self.S) >= 2

    def _kotwica(self, pary, wezly):
        """Wstepnie szacuje skale i obrot kazdego kadru wzgledem kadru odniesienia.

        Po co: daje _uklad() "kotwice" - rownania a ~ a0, b ~ b0 - zeby przy blednych
        parach kadry nie mogly sie skurczyc ani absurdalnie obrocic tylko po to,
        zeby ukryc blad. Tylko skala i obrot - przesuniecia liczy sam uklad.
        Jak liczy (drzewo najsilniejszych polaczen, algorytm Prima):
            1. kadr odniesienia: skala 1, obrot 0 -> c = 1 + 0j
            2. z par dotykajacych znanych kadrow bierze te z NAJWIEKSZA liczba
               punktow (najpewniejsza) - kolejka priorytetowa heapq
            3. dla tej pary liczy wzgledna skale i obrot T (estimateAffinePartial2D)
            4. sklada ze znanym kadrem: c_nowy = c_znany * c_T (mnozenie liczb
               zespolonych = mnozenie skal i dodawanie katow)
            5. dokłada pary nowego kadru do kolejki, powtarza az zna wszystkie kadry
        Laczy kadry lancuchem, wiec bledy sie w nim sumuja - dlatego to tylko
        miekka podpowiedz dla ukladu, a nie wynik.

        Argumenty:
            pary  - {(i, j): (p, q)} dopasowane punkty par kadrow
            wezly - kadry grupy; wezly[0] = kadr odniesienia

        Zwraca:
            {numer_kadru: a0 + b0*j} bez kadru odniesienia;
            a0 = skala*cos(kat), b0 = skala*sin(kat)
            np. {1: 0.916+0.178j, 2: 0.992-0.061j, ...} -> kadr 1: skala 0.93, kat 11 st.
            Kadr, dla ktorego estimateAffinePartial2D nic nie znalazl, nie ma wpisu.

        Uzywana w:
            _rozwiaz() (global_stitcher.py) - w kazdym przebiegu, przed _uklad()
        """
        import heapq
        c = {wezly[0]: complex(1.0, 0.0)}
        kolejka = []

        def krawedzie(k):
            for (i, j), (p, _) in pary.items():
                if k in (i, j):
                    heapq.heappush(kolejka, (-len(p), i, j))

        krawedzie(wezly[0])
        while kolejka and len(c) < len(wezly):
            _, i, j = heapq.heappop(kolejka)
            if (i in c) == (j in c):
                continue
            p, q = pary[(i, j)]
            # rownanie ukladu: S_i(p) = S_j(q); wzgledne T: p = T(q), wiec c_j = c_i * c_T
            T, _ = cv2.estimateAffinePartial2D(q, p)
            if T is None:
                continue
            cT = complex(T[0, 0], T[1, 0])
            if i in c:
                c[j] = c[i] * cT
                krawedzie(j)
            else:
                c[i] = c[j] / cT
                krawedzie(i)
        return {k: v for k, v in c.items() if k != wezly[0]}

    @staticmethod
    def _uklad(pary, pozycja, n, kotwica=None, waga=0.0):
        """Buduje uklad rownan A @ x = bvec na polozenie kadrow.

        Po co: zamienia wszystkie dopasowane pary punktow (i kotwice) na jedna
        macierz, ktora _rozwiaz() rozwiazuje jednym lstsq. Budowana wektorowo -
        wszystkie punkty pary naraz (tablice numpy), bez petli po punktach.
        Kazda para punktow p = (x1, y1) z kadru i i q = (x2, y2) z kadru j daje
        2 rownania "na mozaice p i q wypadaja w tym samym miejscu":
            x:  a_i*x1 - b_i*y1 + tx_i - (a_j*x2 - b_j*y2 + tx_j) = 0
            y:  b_i*x1 + a_i*y1 + ty_i - (b_j*x2 + a_j*y2 + ty_j) = 0
        Kadr odniesienia (miejsce 0) jest ZNANY (1, 0, 0, 0) - nie jest niewiadoma,
        jego wyrazy ida na prawa strone (bvec). Sama kara za odchylenie kadru 0
        nie wystarcza: wspolczynniki sa rzedu setek px, wiec i tak by sie odchylal.
        Kotwica: po 2 rownania na kadr - waga * a_k = waga * a0, waga * b_k = waga * b0.

        Argumenty:
            pary    - {(i, j): (p, q)} - p, q: tablice (m, 2) tych samych miejsc pola
            pozycja - {numer_kadru: miejsce w ukladzie}; miejsce 0 = odniesienie
            n       - liczba kadrow w ukladzie (z odniesieniem)
            kotwica - {numer_kadru: a0 + b0*j} z _kotwica(); None = bez kotwicy
            waga    - sila rownan kotwicy (np. 960)

        Zwraca:
            (A, bvec):
              A    - macierz (liczba_rownan, 4*(n-1)): wiersz = rownanie,
                     kolumny 4*(k-1)..4*k-1 = (a, b, tx, ty) kadru z miejsca k
              bvec - wektor (liczba_rownan,): prawa strona kazdego rownania
            np. 55 kadrow, 128 par: A (8222, 216), bvec (8222,)

        Uzywana w:
            _rozwiaz() (global_stitcher.py) - w kazdym przebiegu, potem lstsq
        """
        stale = np.array([1.0, 0.0, 0.0, 0.0])
        A_bl, b_bl = [], []
        for (i, j), (p, q) in pary.items():
            m = len(p)
            x1, y1, x2, y2 = p[:, 0], p[:, 1], q[:, 0], q[:, 1]
            # wspolczynniki przy (a, b, tx, ty) dla wiersza x i wiersza y
            Ci = np.empty((2 * m, 4))
            Ci[0::2] = np.column_stack([x1, -y1, np.ones(m), np.zeros(m)])
            Ci[1::2] = np.column_stack([y1, x1, np.zeros(m), np.ones(m)])
            Cj = np.empty((2 * m, 4))
            Cj[0::2] = -np.column_stack([x2, -y2, np.ones(m), np.zeros(m)])
            Cj[1::2] = -np.column_stack([y2, x2, np.zeros(m), np.ones(m)])

            # Blok macierzy dla tej pary: 2*m wierszy (x i y kazdego punktu),
            # 4*(n-1) kolumn (niewiadome wszystkich kadrow poza odniesieniem).
            B = np.zeros((2 * m, 4 * (n - 1)))
            b = np.zeros(2 * m)
            for C, k in ((Ci, pozycja[i]), (Cj, pozycja[j])):
                if k == 0:
                    b -= C @ stale                   # znany kadr -> prawa strona
                else:
                    B[:, 4 * (k - 1):4 * k] = C
            A_bl.append(B)
            b_bl.append(b)
        # miekka kotwica: waga * (a_k - a0_k) = 0, waga * (b_k - b0_k) = 0
        for k, c0 in (kotwica or {}).items():
            pk = pozycja.get(k, 0)
            if pk == 0:
                continue
            B = np.zeros((2, 4 * (n - 1)))
            B[0, 4 * (pk - 1)] = waga
            B[1, 4 * (pk - 1) + 1] = waga
            A_bl.append(B)
            b_bl.append(np.array([waga * c0.real, waga * c0.imag]))
        return np.vstack(A_bl), np.concatenate(b_bl)

    @staticmethod
    # Wersja _skladowa dla podanego zbioru par - do kontroli po odrzuceniu par.
    def _skladowa_z(pary, wezly):
        """Zbiera kadry polaczone (bezposrednio albo przez inne) z kadrem odniesienia.

                Po co: kontrola w _rozwiaz() po odrzuceniu blednych par - usunieta para
                mogla byc jedynym polaczeniem jakiegos kadru z reszta. Taki kadr nie ma
                sie do czego "przyczepic", wiec trzeba go wyrzucic z ukladu.
                Roznice wzgledem _skladowa():
                    - dostaje pary i kadry w argumentach (lokalna, juz przycieta kopie par),
                      a nie self.pary - stad @staticmethod (nie uzywa self)
                    - szuka tylko JEDNEJ grupy: tej z kadrem odniesienia wezly[0],
                      a nie najwiekszej sposrod wszystkich
                Jak liczy: graf (kadry = wezly, pary = krawedzie), przeszukiwanie w glab
                od wezly[0] ze stosem - tak samo jak w _skladowa().

                Argumenty:
                    pary  - {(i, j): (p, q)} pary, ktore zostaly po odrzuceniu blednych
                    wezly - kadry ukladu; wezly[0] = kadr odniesienia

                Zwraca:
                    zbior (set) numerow kadrow osiagalnych z kadru odniesienia, z nim samym;
                    np. pary (0,1), (1,2), (3,4), wezly [0..4] -> {0, 1, 2}
                    gdy nikt nie odpadl - wszystkie kadry z wezly

                Uzywana w:
                    _rozwiaz() (global_stitcher.py) - po odrzuceniu par; gdy zbior jest
                    mniejszy niz wezly, zostaja tylko kadry z tego zbioru
                """
        sasiedzi = {k: set() for k in wezly}
        for i, j in pary:
            sasiedzi[i].add(j)
            sasiedzi[j].add(i)
        widziane, stos = {wezly[0]}, [wezly[0]]
        while stos:
            for s in sasiedzi[stos.pop()]:
                if s not in widziane:
                    widziane.add(s)
                    stos.append(s)
        return widziane

    @staticmethod
    # Parametry (a, b, tx, ty) -> macierz 2x3 uzywana przez cv2.warpAffine.
    def _macierz(s):
        a, b, tx, ty = s
        return np.array([[a, -b, tx], [b, a, ty]], np.float64)

    def _blad_pary(self, si, sj, p, q):
        """Mierzy, jak dobrze polozenie dwoch kadrow zgadza sie z ich wspolnymi punktami.

                Po co: po rozwiazaniu ukladu sprawdzamy kazda pare kadrow. Gdyby wszystko
                bylo idealne, punkt p z kadru i i jego para q z kadru j wypadlyby na
                mozaice dokladnie w tym samym miejscu. Duzy blad = para klocaca sie
                z reszta (np. pomylony sasiedni rzad) - do odrzucenia.
                Jak liczy:
                    1. p przeliczone na mozaike polozeniem kadru i (si), q - kadru j (sj)
                    2. odleglosc miedzy kazda para przeliczonych punktow
                    3. mediana tych odleglosci - pojedynczy zly punkt jej nie zawyza
                       (srednia by zawyzyl)

                Argumenty:
                    si, sj - polozenie kadru i oraz j: (a, b, tx, ty) - wiersze z S
                    p, q   - tablice (m, 2): te same miejsca pola na kadrze i (p) i j (q)

                Zwraca:
                    float - mediana odleglosci w px na mozaice;
                    np. poprawna para 0.4-1.8 px, bledna - kilkadziesiat i wiecej

                Uzywana w:
                    _rozwiaz() (global_stitcher.py) - blad kazdej pary w kazdym przebiegu;
                    pary > 3 px sa odrzucane, mediana wszystkich trafia do self.blad_px
                """
        Pi = np.hstack([p, np.ones((len(p), 1))]) @ self._macierz(si).T
        Pj = np.hstack([q, np.ones((len(q), 1))]) @ self._macierz(sj).T
        return float(np.median(np.linalg.norm(Pi - Pj, axis=1)))

    # ---------------- renderowanie ----------------


    def _renderuj(self):
        """Rysuje mozaike z kadrow, kladac kazdy na miejsce policzone w self.S.

               Po co: _rozwiaz() mowi tylko GDZIE leza kadry; tu powstaje obraz mozaiki.
               Kroki:
                   1. dla kazdego kadru: macierz 2x3 z (a, b, tx, ty), obwiednia (prostokat,
                      ktory kadr zajmuje na mozaice), kadr przeksztalcony TYLKO do swojego
                      prostokata (nie do calego plotna - oszczednosc pamieci) + maska
                      (255 = dane kadru, 0 = puste rogi obroconego kadru)
                   2. wyrownanie jasnosci kadrow (GainCompensator) - jeden mnoznik na kadr,
                      zeby nie bylo "schodkow" od roznej ekspozycji
                   3. opcjonalnie szwy (GraphCutSeamFinder) - linie ciecia omijajace
                      roznice miedzy kadrami; ladniej, wolniej
                   4. mieszanie wielopasmowe (MultiBandBlender) - laczy kadry osobno dla
                      drobnych i grubych szczegolow: gladki szew, ostre detale

               Argumenty:
                   brak (korzysta z self.S, self.obrazy, self.wyrownaj, self.szwy, self.bloki)

               Zwraca:
                   True  - mozaika gotowa w self.mozaika (BGR, uint8),
                           np. 55 kadrow 480 px -> ok. 2163 x 1552 px
                   False - blender nic nie zwrocil

               Uzywana w:
                   dodaj() (global_stitcher.py)   - co renderuj_co kadrow, po _rozwiaz()
                   zakoncz() (global_stitcher.py) - ostatnie rysowanie po wszystkich kadrach
               """
        rogi, przeksztalcone, maski, rozmiary = [], [], [], []
        for k, s in self.S.items():
            img = self.obrazy[k]
            h, w = img.shape[:2]
            M = self._macierz(s)
            # Gdzie w mozaice laduja narozniki kadru - z nich obwiednia (x0, y0, x1, y1).
            nar = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ M.T
            x0, y0 = np.floor(nar.min(axis=0)).astype(int)
            x1, y1 = np.ceil(nar.max(axis=0)).astype(int)
            Mt = M.copy()
            # przesuniecie tak, zeby kadr trafil do swojego prostokata od (0, 0)
            Mt[0, 2] -= x0
            Mt[1, 2] -= y0
            rozm = (int(x1 - x0), int(y1 - y0))
            wimg = cv2.warpAffine(img, Mt, rozm, flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
            # Maska: gdzie w prostokacie sa dane kadru (255), a gdzie puste rogi (0).
            wmask = cv2.warpAffine(np.full((h, w), 255, np.uint8), Mt, rozm,
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            rogi.append((int(x0), int(y0)))
            przeksztalcone.append(wimg)
            maski.append(wmask)
            rozmiary.append(rozm)

        if self.wyrownaj:
            try:
                # GainCompensator: jeden wspolczynnik na kadr - przy 48 kadrach
                # 0,8 s zamiast 52 s dla wersji blokowej, PSNR praktycznie ten sam
                komp = (cv2.detail_BlocksGainCompensator() if self.bloki
                        else cv2.detail_GainCompensator())
                komp.feed(rogi, przeksztalcone, maski)
                for i in range(len(przeksztalcone)):
                    komp.apply(i, rogi[i], przeksztalcone[i], maski[i])
            except cv2.error as e:
                log.debug("Wyrownanie ekspozycji pominiete: %s", e)

        if self.szwy and len(przeksztalcone) > 1:
            try:
                f = cv2.detail_GraphCutSeamFinder("COST_COLOR")
                znal = f.find([p.astype(np.float32) for p in przeksztalcone], rogi, maski)
                maski = [m.get() if isinstance(m, cv2.UMat) else np.asarray(m) for m in znal]
            except cv2.error as e:
                log.debug("Szwy pominiete: %s", e)

        # Mieszanie wielopasmowe: laczy kadry osobno dla drobnych i grubych
        # szczegolow, wiec szew jest gladki, a detale nie sa rozmyte.
        # prepare() dostaje obszar calej mozaiki, feed() kolejne kadry z ich polozeniem.
        blender = cv2.detail_MultiBandBlender()
        blender.prepare(cv2.detail.resultRoi(corners=rogi, sizes=rozmiary))
        for wimg, m, r in zip(przeksztalcone, maski, rogi):
            # blender wymaga liczb calkowitych ze znakiem (int16)
            blender.feed(wimg.astype(np.int16), m, r)
        wynik, _ = blender.blend(None, None)
        if wynik is None:
            return False
        self.mozaika = np.clip(wynik, 0, 255).astype(np.uint8)
        return True

    # Lewy gorny rog mozaiki w ukladzie kadru odniesienia - ten sam, ktory
    # wyznacza resultRoi w _renderuj (minimum rogow wszystkich kadrow).
    def _poczatek_mozaiki(self):
        """Liczy, gdzie zaczyna sie obraz mozaiki - lewy gorny rog wszystkich kadrow.

                Po co: polozenia z self.S sa w ukladzie kadru odniesienia (jego lewy gorny
                rog = (0, 0)). Kadry lezace na lewo albo nad nim maja wspolrzedne UJEMNE.
                Obraz mozaiki (self.mozaika) zaczyna sie jednak od piksela (0, 0) - w punkcie
                najbardziej wysunietym w lewo i w gore. Ta funkcja zwraca ten punkt, zeby
                przeliczyc "wspolrzedne ukladu odniesienia" na "piksel w self.mozaika":
                    piksel_mozaiki = punkt - (x_start, y_start)
                Jak liczy: dla kazdego kadru 4 rogi przeliczone na mozaike (jak w _renderuj),
                z nich lewy gorny rog obwiedni; wynik = najmniejsze x i najmniejsze y.

                Argumenty:
                    brak (korzysta z self.S i self.obrazy)

                Zwraca:
                    (x_start, y_start) - liczby calkowite, zwykle <= 0
                    np. symulacja z zakladki 1: (0, -92) - ktorys kadr lezy 92 px wyzej
                    niz kadr odniesienia, wiec punkt (0, 0) ukladu to piksel (0, 92) mozaiki

                Uzywana w:
                    na_mozaike() (global_stitcher.py) - przeliczanie ramek chwastow
                    z kadru na piksele mozaiki (zakladka 4)
                """
        xs, ys = [], []
        for k, s in self.S.items():
            h, w = self.obrazy[k].shape[:2]
            nar = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ self._macierz(s).T
            x0, y0 = np.floor(nar.min(axis=0)).astype(int)
            xs.append(x0)
            ys.append(y0)
        return min(xs), min(ys)


    def na_mozaike(self, sciezka, punkty, rozmiar_oryg):
        """Przelicza punkty z oryginalnego kadru na piksele gotowej mozaiki.

                Po co: detekcja chwastow dziala na pelnym kadrze, a ramki chcemy narysowac
                na mozaice (zakladka 4). Trzeba uwzglednic trzy rzeczy naraz:
                    1. kadr w skladaczu jest zmniejszony do rozdzielczosci roboczej
                       (_przeskaluj) - punkty skalujemy tak samo
                    2. polozenie kadru na mozaice (obrot, skala, przesuniecie z self.S)
                    3. poczatek obrazu mozaiki (_poczatek_mozaiki) - wspolrzedne ukladu
                       odniesienia moga byc ujemne, piksele mozaiki zaczynaja sie od (0, 0)

                Argumenty:
                    sciezka      - plik kadru, tak jak byl podany do dodaj() (Path albo str)
                    punkty       - lista [x, y] w pikselach ORYGINALNEGO kadru,
                                   np. 4 rogi ramki chwastu [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                    rozmiar_oryg - (wysokosc, szerokosc) oryginalnego kadru, np. (2176, 2902)

                Zwraca:
                    np.ndarray (N, 2) - te same punkty w pikselach self.mozaika;
                    None - gdy brak mozaiki albo ten kadr nie wszedl do mozaiki
                    (nie dodany, za malo punktow, odciety od reszty, zla skala)

                Uzywana w:
                    app.py, zakladka 4 (linia 758) - ramki wykrytych chwastow z kazdego
                    kadru przeniesione na mozaike (rysuj_ramki, GIF lotu)
                """
        if not self.S:
            return None
        try:
            k = [str(x) for x in self.sciezki].index(str(sciezka))
        except ValueError:
            return None
        if k not in self.S:
            return None
        h0, w0 = rozmiar_oryg
        hk, wk = self.obrazy[k].shape[:2]
        pts = np.asarray(punkty, float) * [wk / w0, hk / h0]
        out = np.hstack([pts, np.ones((len(pts), 1))]) @ self._macierz(self.S[k]).T
        ox, oy = self._poczatek_mozaiki()
        return out - [ox, oy]

    # Wywolac po ostatnim kadrze. Potrzebne, bo przy renderuj_co > 1 ostatnie
    # kadry moga jeszcze nie byc na mozaice.
    def zakoncz(self):
        """Dorysowuje mozaike po ostatnim kadrze, jesli nie jest aktualna.

        Po co: przy renderuj_co > 1 dodaj() rysuje mozaike tylko co N kadrow.
        Jesli lot skonczyl sie "pomiedzy" (np. 55 kadrow przy renderuj_co 3 -
        ostatnie rysowanie przy kadrze 54), ostatnie kadry nie trafilyby do
        mozaiki. zakoncz() liczy uklad i rysuje jeszcze raz, ze wszystkimi kadrami.
        Gdy mozaika jest juz aktualna - nic nie robi (bez zbednego liczenia).

        Argumenty:
            brak

        Zwraca:
            True  - mozaika jest (aktualna albo wlasnie dorysowana)
            False - mozaiki nie ma (za malo kadrow albo _rozwiaz/_renderuj sie nie udaly)

        Uzywana w:
            app.py, zakladka 3 (linia 407) - po petli dodaj(); nowa mozaika trafia
                                             jako ostatnia klatka do GIF-a
            app.py, zakladka 4 (linia 749) - tu renderuj_co = 10**6, wiec dodaj()
                                             nigdy nie rysuje - cala mozaika powstaje tutaj
            cli.py - na koncu batch / watch / demo
        """
        if self._aktualna or len(self.obrazy) < 2:
            return self.mozaika is not None
        t0 = time.time()
        ok = self._rozwiaz() and self._renderuj()
        self.historia.append((len(self.obrazy), round(time.time() - t0, 2), ok))
        self._aktualna = ok
        return ok

    def raport(self):
        """Stan globalnego wyrownania - ile kadrow i par weszlo do mozaiki."""
        return {"kadrow_w_mozaice": len(self.S or {}),
                "kadrow_wczytanych": len(self.obrazy),
                "par": len(self.pary),
                "mediana_bledu_par_px": round(getattr(self, "blad_px", 0.0), 2)}

    def raport_czasu(self):
        return [{"zdjec": n, "czas_s": t, "ok": ok} for n, t, ok in self.historia]
