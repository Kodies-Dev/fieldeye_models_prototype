"""GUI do symulatora lotu i skladania zdjec.

Uruchomienie (z katalogu projektu):
    streamlit run app.py

Zakladki:
  1. Rozklad zdjecia  - tniesz jedno zdjecie na kadry, tak jakby robil je dron
  2. Kadry            - przegladasz i pobierasz wyciete kadry
  3. Skladanie        - symulacja laczenia kadr po kadrze, pobieranie wyniku
  4. Lot z detekcja  - symulowany lot z wykrywaniem chwastow i mozaika z chwastami

Jak dziala Streamlit: caly ten plik wykonuje sie od gory do dolu przy KAZDYM
kliknieciu lub ruchu suwaka. Dlatego wyniki, ktore maja przetrwac miedzy
kliknieciami (kadry, mozaika), trzymamy w st.session_state - slowniku, ktory
Streamlit zachowuje przez cala sesje przegladarki.
"""
import io
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from stitchlive.flight_sim import SimSettings, macierz_kadru, rozloz
from stitchlive.camera import ARDUCAM_64MP_16MP, GSD_WEEDSGALORE_MM, KAMERY, opis_lotu
from stitchlive.field import zbuduj_pole
from stitchlive.evaluation import ocen
from stitchlive.global_stitcher import GlobalStitcher
from stitchlive.detail import DetailStitcher
from stitchlive.basic import BasicStitcher
from models.device import wybierz
from mission.detector import DummyDetector, HailoDetector, TiledDetector, YoloDetector
from models.evaluation import pr_przy_progach
from models.plan import katalog_etykiet


st.set_page_config(page_title="FIELDEYE - symulator lotu", layout="wide")

ROZSZERZENIA = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# gotowe zestawy zaklocen - zeby nie ustawiac suwakow recznie
PRESETY = {
    "Czyste ciecie": dict(obrot=0.0, skala=0.0, ekspozycja=0.0, rozmycie=0.0,
                          szum=0.0, winieta=0.0, jitter=0.0),
    "Realistyczny dron": dict(obrot=3.0, skala=0.03, ekspozycja=0.05, rozmycie=0.1,
                              szum=2.0, winieta=0.3, jitter=0.1),
    #raczej dla zabawy
    "Trudne warunki": dict(obrot=6.0, skala=0.05, ekspozycja=0.07, rozmycie=0.25,
                           szum=4.0, winieta=0.35, jitter=0.12),
}


# ---------------- stan sesji ----------------

# kazda sesja przegladarki dostaje swoj wlasny katalog na pliki
def katalog_roboczy():
    """Zwraca osobny folder tymczasowy dla tej sesji przegladarki.

    Po co: kazda karta przegladarki (i kazda osoba wchodzaca przez Network URL)
    dostaje wlasny folder, wiec dwie osoby korzystajace z aplikacji naraz nie
    nadpisza sobie kadrow. Folder powstaje raz na sesje (tempfile.mkdtemp,
    nazwa fieldeye_ + losowa koncowka), sciezka jest trzymana w session_state.
    Aplikacja go nie usuwa; F5 = nowa sesja = nowy folder.

    Argumenty:
        brak

    Zwraca:
        Path folderu, np. /var/folders/.../T/fieldeye_a8k2j3x1 (Mac), /tmp/fieldeye_... (Linux)

    Uzywana w:
        zakladka 1 - folder "symulacja" (kadry, prawda.json, podglad trasy)
        zakladka 3 - folder "wgrane" (wlasne zdjecia / ZIP) i "ocena" (podglady oceny)
        zakladka 4 - folder "lot" (kadry lotu z detekcja)
    """
    if "workdir" not in st.session_state:
        st.session_state.workdir = Path(tempfile.mkdtemp(prefix="fieldeye_"))#koncowka random
    return st.session_state.workdir



# przesuwanie wszystkich suwakow gdy wybrierzemy zestaw zaklocen presety
def ustaw_preset():
    """Przestawia wszystkie suwaki zaklocen na wartosci wybranego zestawu.

    Po co: zeby nie ustawiac 7 suwakow recznie. Wywolywana przez selectbox
    "Zestaw" (on_change) PRZED ponownym wykonaniem skryptu - wtedy zaden suwak
    jeszcze nie istnieje i wolno zmienic jego wartosc w session_state.
    Suwaki maja te same klucze (key="obrot" itd.), wiec od razu pokazuja nowe liczby.

    Argumenty:
        brak (nazwa zestawu z st.session_state.preset)

    Zwraca:
        nic; ustawia st.session_state: obrot, skala, ekspozycja, rozmycie,
        szum, winieta, jitter (np. "Trudne warunki" -> obrot 6.0, szum 4.0 ...)

    Uzywana w:
        zakladka 1 - selectbox "Zestaw" (on_change=ustaw_preset)
    """
    for k, v in PRESETY[st.session_state.preset].items():
        st.session_state[k] = v


# setdefault - ustawia wartosc tylko przy pierwszym uruchomieniu, nie nadpisuje zmian
def inicjuj_suwaki():
    """Ustawia wartosci startowe suwakow zaklocen (zestaw "Realistyczny dron").

    Po co: suwaki zaklocen nie maja podanej wartosci domyslnej, tylko klucz -
    bez tego startowalyby od minimum (0). setdefault wpisuje wartosc tylko,
    gdy klucza jeszcze nie ma, czyli przy pierwszym uruchomieniu; pozniejsze
    przebiegi (rerun) nie cofaja tego, co ustawil uzytkownik.

    Argumenty:
        brak

    Zwraca:
        nic; ustawia (tylko raz) st.session_state: preset + 7 wartosci zaklocen

    Uzywana w:
        poczatek skryptu - wolana przy kazdym przebiegu, przed narysowaniem zakladek
    """
    for k, v in PRESETY["Realistyczny dron"].items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("preset", "Realistyczny dron")


# ---------------- pomocnicze ----------------

# OpenCV trzyma kolory jako BGR, przegladarka oczekuje RGB
def bgr_na_rgb(img):
    """Zamienia kolejnosc kanalow kolorow z BGR (OpenCV) na RGB (przegladarka).

    Po co: OpenCV trzyma piksel jako B, G, R, a Streamlit, PIL i przegladarka
    oczekuja R, G, B. Bez zamiany czerwony i niebieski zamieniaja sie miejscami
    (brazowa gleba wyglada na szaroniebieska). cv2.imwrite nie potrzebuje zamiany.

    Argumenty:
        img - obraz OpenCV (BGR, uint8), ksztalt (h, w, 3)

    Zwraca:
        nowy obraz RGB, ten sam ksztalt; oryginal bez zmian

    Uzywana w:
        kazde st.image w zakladkach 1-4, gif_z_klatek(), gif_lotu()
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # zamiana kolejnosci kanalow BGR -> RGB


# plik z uploadu to bajty w pamieci - imdecode dekoduje je bez zapisu na dysk
def wczytaj_z_uploadu(plik):
    """Zamienia wgrany plik na obraz OpenCV - bez zapisu na dysk.

    Po co: plik z st.file_uploader jest tylko w pamieci, a cv2.imread potrzebuje
    sciezki na dysku. Bajty pliku (skompresowany JPG/PNG) -> tablica uint8
    (np.frombuffer) -> dekodowanie (cv2.imdecode) jak przy imread.
    IMREAD_COLOR - zawsze 3 kanaly (szary rozszerzony, przezroczystosc PNG odrzucona).

    Argumenty:
        plik - UploadedFile z st.file_uploader

    Zwraca:
        obraz BGR (h, w, 3), uint8; None - gdy plik nie jest obrazem
        np. JPG 2 MB -> 3000 x 4000 x 3 = 36 MB pikseli

    Uzywana w:
        zakladka 1 - "Rozloz zdjecie"
        zakladka 4 - pole "Wgraj zdjecie"
    """
    dane = np.frombuffer(plik.getvalue(), np.uint8)
    return cv2.imdecode(dane, cv2.IMREAD_COLOR)

#obraz mam mieć najwyżej max_wymiar pikeseli dluzszy bok
def zmniejsz(img, max_wymiar):
    """Zmniejsza obraz tak, zeby dluzszy bok mial najwyzej max_wymiar px.

    Po co: zdjecia z drona maja ponad 4000 px - zmniejszenie przyspiesza ciecie,
    skladanie i ocene kilkukrotnie. Proporcje zostaja (ta sama skala k w poziomie
    i w pionie), obraz nigdy nie jest powiekszany. INTER_AREA - przy zmniejszaniu
    usrednia piksele, bez mory na rzedach uprawy.

    Argumenty:
        img        - obraz BGR
        max_wymiar - najwiekszy dozwolony dluzszy bok w px; 0 / None = bez zmian

    Zwraca:
        obraz zmniejszony albo ten sam, gdy juz jest mniejszy
        np. 4000 x 3000 przy 2400 -> k = 0.6 -> 2400 x 1800

    Uzywana w:
        zakladka 1 - zrodlo po wczytaniu (suwak "Zmniejsz zrodlo do", dom. 2400)
        zakladka 4 - para [kadr | wykryte chwasty] do animacji GIF (do 1600 px)
    """
    if max_wymiar and max(img.shape[:2]) > max_wymiar: #jezeli obraz a raczej jego najdluzszy bok ma wiecej niz max wymiar
        k = max_wymiar / max(img.shape[:2]) #o ile przeskalowac
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return img


# ZIP w pamieci (bez zapisu na dysk)
def spakuj(pliki, dodatkowe=None):
    """Pakuje pliki do ZIP-a w pamieci (bez zapisu na dysk) - do przycisku pobierania.

    Po co: st.download_button potrzebuje bajtow pliku; ZIP budowany w io.BytesIO
    nie zostawia smieci na dysku. Pliki z dysku trafiaja do folderu images/
    w ZIP-ie, dodatkowe dane (JSON, JPG z pamieci) - pod podana nazwa.

    Argumenty:
        pliki     - lista sciezek do plikow z dysku (moze byc pusta)
        dodatkowe - {nazwa_w_zipie: bajty albo tekst}, np. {"prawda.json": "..."};
                    None = brak

    Zwraca:
        bytes - gotowy plik ZIP (kompresja ZIP_DEFLATED)

    Uzywana w:
        zakladka 2 - kadry.zip (same kadry) i symulacja_pelna.zip (+ prawda, podglad, oryginal)
        zakladka 3 - skladanie.zip (kroki skladania + mozaika + raport)
    """
    bufor = io.BytesIO()
    with zipfile.ZipFile(bufor, "w", zipfile.ZIP_DEFLATED) as z:
        for p in pliki:
            z.write(p, arcname=f"images/{Path(p).name}")
        for nazwa, dane in (dodatkowe or {}).items():
            z.writestr(nazwa, dane)
    return bufor.getvalue()

#zamienia na jpg w bajtach
def jpg_bajty(img, jakosc=95):
    """Koduje obraz do JPG w pamieci (bajty pliku) - odwrotnosc wczytaj_z_uploadu.

    Po co: przyciski pobierania i ZIP-y potrzebuja bajtow pliku, a nie tablicy
    pikseli. cv2.imencode kompresuje, tobytes() zamienia na zwykle bytes.
    Jakosc 0-100: powyzej ok. 90 roznicy prawie nie widac, a plik rosnie
    (pole 2400 x 1600: jakosc 95 -> 1.9 MB, 92 -> 1.5 MB, 85 -> 1.1 MB).

    Argumenty:
        img    - obraz BGR
        jakosc - jakosc JPEG 0-100 (dom. 95)

    Zwraca:
        bytes - plik JPG

    Uzywana w:
        zakladka 3 - "Mozaika (JPG)", kroki skladania w ZIP-ie (jakosc 85), mozaika.jpg w ZIP-ie
        zakladka 4 - "Pobierz mozaike (JPG, aktualny widok)"
    """
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jakosc])
    return buf.tobytes()

# zakladka 3: animacja rosnacej mozaiki; klatki - lista obrazow OpenCV,
# szer - szerokosc GIF w px, ms - czas jednej klatki w ms
def gif_z_klatek(klatki, szer=700, ms=500):
    """Sklada kolejne stany mozaiki w animowany GIF - widac, jak mozaika rosnie.

    Po co: podglad i plik do pobrania z przebiegu skladania. Wszystkie klatki
    sa wklejane w lewy gorny rog wspolnego czarnego plotna (rozmiar najwiekszej
    klatki), zeby animacja nie skakala, gdy mozaika rosnie. Potem plotno
    zmniejszane do szerokosci szer, BGR -> RGB, PIL zapisuje GIF w pamieci.
    GIF ma najwyzej 256 kolorow na klatke - do ogladania wystarczy.

    Argumenty:
        klatki - lista obrazow BGR (kolejne stany mozaiki), rozne rozmiary
        szer   - szerokosc GIF-a w px (wysokosc z proporcji), dom. 700
        ms     - czas jednej klatki w ms, dom. 500 (2 klatki na sekunde)

    Zwraca:
        bytes - plik GIF (petla w nieskonczonosc); None - gdy brak klatek
        np. 55 kadrow, odswiezanie co 3 -> ok. 19 klatek, 700 x 502 px, 9.5 s

    Uzywana w:
        zakladka 3 - "Animacja skladania (GIF)" i podglad animacji
    """
    if not klatki:
        return None
    H = max(k.shape[0] for k in klatki) #najwieksza wysokosc
    W = max(k.shape[1] for k in klatki) #najwieksza szerokosc z klatek do tworzenia czarnego plotna
    s = szer / W #skala
    obrazy = []
    for k in klatki:
        plotno = np.zeros((H, W, 3), np.uint8) #czarne polotno
        plotno[:k.shape[0], :k.shape[1]] = k #dodaje do szerokosci wysokosci poloczone zdj
        plotno = cv2.resize(plotno, (szer, max(1, int(H * s))), interpolation=cv2.INTER_AREA) #
        obrazy.append(Image.fromarray(bgr_na_rgb(plotno)))
    bufor = io.BytesIO()
    obrazy[0].save(bufor, format="GIF", save_all=True, append_images=obrazy[1:],
                   duration=ms, loop=0) # loop=0 - petla w nieskonczonosc
    return bufor.getvalue()


def rozpakuj_kadry(pliki_uploadu, cel):
    """Zapisuje kadry wgrane w zakladce 3 do jednego folderu i zwraca je po kolei.

    Po co: uzytkownik moze wgrac pojedyncze zdjecia albo ZIP (np. kadry.zip
    z zakladki 2). Skladacz potrzebuje listy plikow na dysku w kolejnosci lotu.
    Z ZIP-a brane sa tylko obrazy; foldery wewnatrz sa splaszczane (images/IMG_0001.jpg
    -> IMG_0001.jpg), smieci z __MACOSX pomijane. Folder docelowy jest najpierw
    czyszczony, zeby nie mieszac starych i nowych zdjec.

    Argumenty:
        pliki_uploadu - lista UploadedFile z st.file_uploader: obrazy
                        (.jpg .jpeg .png .tif .tiff) albo ZIP-y
        cel           - Path folderu docelowego, np. katalog_roboczy() / "wgrane"

    Zwraca:
        list[Path] - zapisane obrazy posortowane po nazwie (= kolejnosc lotu, gdy
        nazwy maja zera wiodace: IMG_0001...; 1.jpg, 10.jpg, 2.jpg ulozylyby sie zle);
        pusta lista, gdy nie wgrano zadnego obrazu

    Uzywana w:
        zakladka 3 - zrodlo "Wgraj wlasne zdjecia / ZIP" po kliknieciu "Zloz mozaike"
    """
    if cel.exists():
        shutil.rmtree(cel)
    cel.mkdir(parents=True)
    for f in pliki_uploadu:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z:
                for n in z.namelist():
                    if Path(n).suffix.lower() in ROZSZERZENIA and not n.startswith("__MACOSX"):
                        (cel / Path(n).name).write_bytes(z.read(n))
        elif Path(f.name).suffix.lower() in ROZSZERZENIA:
            (cel / f.name).write_bytes(f.getvalue())
    return sorted(p for p in cel.iterdir() if p.suffix.lower() in ROZSZERZENIA)


# =====================================================================
inicjuj_suwaki()
st.title("FIELDEYE — symulator lotu i składanie zdjęć")
st.caption("Tniesz jedno zdjęcie na kadry tak, jakby robił je dron, pobierasz je, "
           "a potem sprawdzasz, czy program złoży je z powrotem.")

# cztery zakladki - kod kazdej jest w bloku 'with tabN:' ponizej
tab1, tab2, tab3, tab4 = st.tabs(["1. Rozkład zdjęcia", "2. Kadry do pobrania", "3. Składanie",
                                  "4. Lot z detekcją"])

# =====================================================================
# 1. ROZKLAD
# =====================================================================
# =====================================================================
# ZAKLADKA 1: parametry po lewej, wynik po prawej
# =====================================================================
with tab1:
    lewa, prawa = st.columns([1, 2])

    with lewa:
        st.subheader("Źródło")

        st.text("Wgraj zdjęcie")

        plik = st.file_uploader("Duże zdjęcie (zdjęcie z drona, pola ...)",
                                type=["jpg", "jpeg", "png", "tif", "tiff"])

        max_wym = st.slider("Zmniejsz źródło do (dłuższy bok, px)", 800, 6000, 2400, 100,
                            help="Duże zdjęcia z drona mają 4000+ px - zmniejszenie "
                                 "przyspiesza wszystko kilkukrotnie")

        st.subheader("FlightConfig")
        tryb = st.radio("Tryb cięcia", ["trasa", "losowy"], horizontal=True,
                        help="trasa = przelot tam i z powrotem, losowy = losowe miejsca "
                             "z zachowanym pokryciem między kolejnymi kadrami")
        kadr = st.select_slider("Rozmiar kadru (px)", [320, 400, 480, 560, 640, 800], 480)
        overlap = st.slider("Pokrycie kolejnych kadrów", 0.3, 0.9, 0.65, 0.05)
        if tryb == "trasa":
            sidelap = st.slider("Pokrycie między liniami", 0.2, 0.9, 0.5, 0.05)
            obrot_linii = st.checkbox("Co druga linia obrócona o 180° (dron zawraca)")
            liczba = 0
        else:
            sidelap, obrot_linii = 0.5, False
            liczba = st.slider("Liczba kadrów", 5, 150, 30)

        st.subheader("Wygląd jak z drona")
        st.selectbox("Zestaw", list(PRESETY), key="preset", on_change=ustaw_preset)
        with st.expander("Szczegóły zakłóceń", expanded=False):
            st.slider("Odchylenie kursu (°)", 0.0, 20.0, key="obrot", step=0.5)
            st.slider("Zmiana wysokości lotu", 0.0, 0.15, key="skala", step=0.01)
            st.slider("Zmienność jasności", 0.0, 0.2, key="ekspozycja", step=0.01)
            st.slider("Szansa rozmycia ruchowego", 0.0, 1.0, key="rozmycie", step=0.05)
            st.slider("Szum matrycy", 0.0, 15.0, key="szum", step=0.5)
            st.slider("Winietowanie (ciemne rogi)", 0.0, 0.8, key="winieta", step=0.05)
            st.slider("Losowe odchylenie pozycji", 0.0, 0.4, key="jitter", step=0.01)
        seed = st.number_input("Ziarno losowania", 0, 99999, 7,
                               help="To samo ziarno = te same kadry")

        uruchom = st.button("Rozłóż zdjęcie", type="primary", width="stretch")

    with prawa:
        # Ten blok wykonuje sie tylko w przebiegu, w ktorym kliknieto przycisk.
        # Wynik trafia do session_state, wiec zostaje widoczny po kolejnych kliknieciach.
        if uruchom:
            if plik is None:
                st.warning("Najpierw wgraj zdjęcie.")
                st.stop()
            img = wczytaj_z_uploadu(plik)
            if img is None:
                st.error("Nie udało się odczytać obrazu.")
                st.stop()

            img = zmniejsz(img, max_wym)

            u = SimSettings(tryb=tryb, kadr=int(kadr), overlap=overlap, sidelap=sidelap,
                           liczba=int(liczba), jitter=st.session_state.jitter,
                           obrot=st.session_state.obrot, obrot_linii=obrot_linii,
                           skala=st.session_state.skala,
                           ekspozycja=st.session_state.ekspozycja,
                           rozmycie=st.session_state.rozmycie, szum=st.session_state.szum,
                           winieta=st.session_state.winieta, seed=int(seed))
            out = katalog_roboczy() / "symulacja"
            try:
                with st.spinner("Tnę zdjęcie na kadry..."):
                    prawda = rozloz(img, out, u)
            except ValueError as e:
                st.error(f"{e}. Zmniejsz rozmiar kadru albo zwiększ źródło.")
                st.stop()
            # Zapamietujemy: gdzie sa kadry, prawde (do oceny) i oryginal (do porownania).
            st.session_state.symulacja = {"katalog": out, "prawda": prawda, "oryginal": img}
            st.session_state.pop("wynik", None)     # stary wynik składania nieaktualny

        sym = st.session_state.get("symulacja")
        if sym:
            k = sym["prawda"]["kadry"]
            st.success(f"Gotowe: {len(k)} kadrów {sym['prawda']['ustawienia']['kadr']} px. "
                       "Przejdź do zakładki 2, żeby je pobrać, albo 3, żeby je złożyć.")
            st.image(bgr_na_rgb(cv2.imread(str(sym["katalog"] / "podglad_trasy.jpg"))),
                     caption="Czerwone ramki = kadry, żółta linia = kolejność lotu",
                     width="stretch")
            przyklad = [sym["katalog"] / "images" / x["plik"] for x in k[:4]]
            st.caption("Pierwsze kadry:")
            kol = st.columns(len(przyklad))
            for c, p in zip(kol, przyklad):
                c.image(bgr_na_rgb(cv2.imread(str(p))), caption=p.name,
                        width="stretch")
        else:
            st.info("Ustaw parametry po lewej i kliknij **Rozłóż zdjęcie**.")

# =====================================================================
# 2. KADRY
# =====================================================================
# =====================================================================
# ZAKLADKA 2: przegladanie i pobieranie kadrow z zakladki 1
# =====================================================================
with tab2:
    sym = st.session_state.get("symulacja")
    if not sym:
        st.info("Najpierw rozłóż zdjęcie w zakładce 1.")
    else:
        kat = sym["katalog"]
        pliki = sorted((kat / "images").glob("*.jpg"))
        rozmiar_mb = sum(p.stat().st_size for p in pliki) / 1e6

        c1, c2, c3 = st.columns(3)
        c1.metric("Kadrów", len(pliki))
        c2.metric("Rozmiar kadru", f"{sym['prawda']['ustawienia']['kadr']} px")
        c3.metric("Łącznie", f"{rozmiar_mb:.1f} MB")

        dodatki = {
            "prawda.json": json.dumps(sym["prawda"], indent=1),
            "podglad_trasy.jpg": (kat / "podglad_trasy.jpg").read_bytes(),
            "oryginal.jpg": (kat / "oryginal.jpg").read_bytes(),
        }
        d1, d2 = st.columns(2)
        d1.download_button("Pobierz wszystkie kadry (ZIP)", spakuj(pliki),
                           file_name="kadry.zip", mime="application/zip",
                           type="primary", width="stretch")
        d2.download_button("Pobierz kadry + prawda.json + podgląd trasy (ZIP)",
                           spakuj(pliki, dodatki), file_name="symulacja_pelna.zip",
                           mime="application/zip", width="stretch")
        st.caption("`prawda.json` zawiera prawdziwe położenie, kąt i zasięg każdego kadru "
                   "w oryginale - przydaje się do sprawdzenia dokładności składania.")

        st.divider()
        # galeria stronicowana - setka miniatur naraz spowolnilaby przegladarke
        na_strone = 24
        stron = max(1, (len(pliki) + na_strone - 1) // na_strone)
        strona = st.number_input("Strona galerii", 1, stron, 1) if stron > 1 else 1
        widoczne = pliki[(strona - 1) * na_strone: strona * na_strone]
        kolumny = st.columns(6)
        for i, p in enumerate(widoczne):
            kolumny[i % 6].image(bgr_na_rgb(cv2.imread(str(p))), caption=p.name,
                                 width="stretch")

        st.divider()
        wybrany = st.selectbox("Podgląd pojedynczego kadru", [p.name for p in pliki])
        p = kat / "images" / wybrany
        info = next(x for x in sym["prawda"]["kadry"] if x["plik"] == wybrany)
        a, b = st.columns([2, 1])
        a.image(bgr_na_rgb(cv2.imread(str(p))), width="stretch")
        b.json({"srodek_w_oryginale_px": info["srodek"], "kat_st": info["kat_st"],
                "zasieg_px": info["zasieg_px"]})
        b.download_button(f"Pobierz {wybrany}", p.read_bytes(), file_name=wybrany,
                          mime="image/jpeg", width="stretch")

# =====================================================================
# 3. SKLADANIE
# =====================================================================
# =====================================================================
# ZAKLADKA 3: skladanie kadrow w mozaike z podgladem na zywo
# =====================================================================
with tab3:
    lewa, prawa = st.columns([1, 2])

    with lewa:
        st.subheader("Kadry do złożenia")
        opcje = ["Z zakładki 1 (symulacja)", "Wgraj własne zdjęcia / ZIP"]
        zrodlo = st.radio("Źródło", opcje,
                          index=0 if st.session_state.get("symulacja") else 1)
        wgrane = None
        if zrodlo == opcje[1]:
            wgrane = st.file_uploader("Zdjęcia w kolejności lotu albo jeden ZIP",
                                      type=["jpg", "jpeg", "png", "tif", "tiff", "zip"],
                                      accept_multiple_files=True)

        st.subheader("Metoda")
        metoda = st.radio("Algorytm", ["global", "detail", "basic"], horizontal=True,
                          help="global - odporna na obrót kadrów (domyślna); "
                               "detail - pipeline cv2.detail; basic - cv2.Stitcher")
        megapix = st.slider("Rozdzielczość robocza (Mpx)", 0.1, 2.0, 0.6, 0.1)
        renderuj_co = st.slider("Odświeżaj mozaikę co N kadrów", 1, 20, 3,
                                help="Mniejsze N = płynniejszy podgląd, ale wolniej")
        zakres = st.number_input("Dopasowuj z N poprzednimi kadrami (0 = z każdym)", 0, 200, 0)
        szwy = st.checkbox("Wyznaczaj szwy (ładniej, wolniej)")
        opoznienie = st.slider("Odstęp między kadrami (s) - symulacja lotu", 0.0, 2.0, 0.0, 0.1)
        zloz = st.button("Złóż mozaikę", type="primary", width="stretch")

    with prawa:
        if zloz:
            if zrodlo == opcje[0]:
                sym = st.session_state.get("symulacja")
                if not sym:
                    st.warning("Brak symulacji - rozłóż zdjęcie w zakładce 1.")
                    st.stop()
                sciezki = sorted((sym["katalog"] / "images").glob("*.jpg"))
            else:
                if not wgrane:
                    st.warning("Wgraj zdjęcia albo ZIP.")
                    st.stop()
                sciezki = rozpakuj_kadry(wgrane, katalog_roboczy() / "wgrane")
                sym = None
            if len(sciezki) < 2:
                st.warning("Potrzebne są co najmniej dwa zdjęcia.")
                st.stop()

            if metoda == "global":
                # wszystkie trzy skladacze maja te sama metode dodaj() - reszta kodu jest wspolna
                s = GlobalStitcher(work_megapix=megapix, zakres_par=int(zakres),
                                     szwy=szwy, renderuj_co=renderuj_co)
            elif metoda == "detail":
                s = DetailStitcher(work_megapix=megapix, zakres_par=int(zakres),
                                   szwy=szwy, renderuj_co=renderuj_co)
            else:
                s = BasicStitcher(downscale=min(1.0, megapix / 0.6 * 0.5))

            pasek = st.progress(0.0, text="Start...")
            podglad = st.empty()
            # Petla kadr po kadrze. st.empty() to 'miejsce' na stronie, ktore nadpisujemy
            # nowym obrazem - dlatego mozaika rosnie w tym samym miejscu zamiast dopisywac
            # kolejne obrazki pod spodem. klatki - kolejne stany mozaiki do animacji GIF.
            klatki, t0 = [], time.time()
            for i, p in enumerate(sciezki, start=1):
                ok = s.dodaj(p)
                if ok and s.mozaika is not None:
                    klatki.append(s.mozaika.copy())
                    podglad.image(bgr_na_rgb(s.mozaika), width="stretch",
                                  caption=f"Mozaika po {i} z {len(sciezki)} kadrów")
                pasek.progress(i / len(sciezki), text=f"Kadr {i}/{len(sciezki)}: {p.name}")
                if opoznienie:
                    time.sleep(opoznienie)
            if hasattr(s, "zakoncz") and s.zakoncz() and s.mozaika is not None:
                if not klatki or klatki[-1].shape != s.mozaika.shape or \
                        not np.array_equal(klatki[-1], s.mozaika):
                    klatki.append(s.mozaika.copy())
            czas = time.time() - t0
            pasek.empty()
            podglad.empty()

            if s.mozaika is None:
                st.error("Nie udało się złożyć mozaiki. Zwiększ pokrycie kadrów albo "
                         "spróbuj metody global.")
                st.stop()

            wynik = {"mozaika": s.mozaika, "klatki": klatki, "czas": czas,
                     "n": len(sciezki), "metoda": metoda,
                     "raport": s.raport() if hasattr(s, "raport") else None}
            if sym:
                wynik["ocena"] = ocen(s.mozaika, sym["oryginal"], sym["prawda"],
                                      outdir=katalog_roboczy() / "ocena")
            st.session_state.wynik = wynik

        w = st.session_state.get("wynik")
        if w:
            h, wd = w["mozaika"].shape[:2]
            c = st.columns(4)
            c[0].metric("Kadrów", w["n"])
            c[1].metric("Czas", f"{w['czas']:.1f} s")
            c[2].metric("Mozaika", f"{wd}×{h}")
            if w.get("raport"):
                c[3].metric("W mozaice", f"{w['raport']['kadrow_w_mozaice']}/{w['n']}")

            st.image(bgr_na_rgb(w["mozaika"]), caption=f"Mozaika ({w['metoda']})",
                     width="stretch")

            o = w.get("ocena")
            if o and o.get("ok"):
                st.subheader("Porównanie z oryginałem")
                m = st.columns(4)
                m[0].metric("PSNR", f"{o['psnr_db']} dB", help="powyżej ~25 dB = bardzo dobrze")
                m[1].metric("Pokrycie", f"{o['pokrycie']:.1%}")
                m[2].metric("Skala", o["skala"], help="1.0 = zgodna z oryginałem")
                m[3].metric("Błąd średni", o["blad_sredni"])
                st.write(f"**Werdykt:** {o['werdykt']}")
                nal = katalog_roboczy() / "ocena" / "ocena_nalozenie.jpg"
                if nal.exists():
                    st.image(bgr_na_rgb(cv2.imread(str(nal))),
                             caption="Mozaika nałożona na oryginał", width="stretch")
            elif o:
                st.warning(f"Ocena nieudana: {o.get('powod')}")

            st.subheader("Pobierz")
            gif = gif_z_klatek(w["klatki"])
            raport = {"metoda": w["metoda"], "kadrow": w["n"], "czas_s": round(w["czas"], 2),
                      "rozmiar_px": [wd, h], "wyrownanie": w.get("raport"),
                      "ocena": {k: v for k, v in (o or {}).items() if k != "macierz"}}
            d = st.columns(3)
            d[0].download_button("Mozaika (JPG)", jpg_bajty(w["mozaika"]),
                                 file_name="mozaika.jpg", mime="image/jpeg",
                                 type="primary", width="stretch")
            if gif:
                d[1].download_button("Animacja składania (GIF)", gif,
                                     file_name="skladanie.gif", mime="image/gif",
                                     width="stretch")
            # ZIP: kazdy kolejny stan mozaiki jako osobny JPG + koncowa mozaika + raport
            kroki = {f"kroki/krok_{i:03d}.jpg": jpg_bajty(k, 85)
                     for i, k in enumerate(w["klatki"], start=1)}
            kroki["mozaika.jpg"] = jpg_bajty(w["mozaika"])
            kroki["raport.json"] = json.dumps(raport, indent=2, ensure_ascii=False,
                                              default=str)
            d[2].download_button("Wszystko: mozaika + kroki + raport (ZIP)",
                                 spakuj([], kroki), file_name="skladanie.zip",
                                 mime="application/zip", width="stretch")
            if gif:
                with st.expander("Podgląd animacji składania"):
                    st.image(gif)
        else:
            st.info("Wybierz kadry i kliknij **Złóż mozaikę**. Podgląd będzie rosnął "
                    "kadr po kadrze, jak w trakcie lotu.")


# =====================================================================
# ZAKLADKA 4: symulacja lotu z detekcja chwastow
# Kadry pojawiaja sie po kolei: oryginal | wykryte chwasty. Na koncu mozaika
# z chwastami przeliczonymi z kadrow (GlobalStitcher.na_mozaike) i scalonymi,
# bo ten sam chwast widac na kilku nakladajacych sie kadrach.
# =====================================================================

@st.cache_resource(show_spinner="Wczytuję model...")
def detektor_lotu(model, prog, wejscie):
    """Wczytuje model do wykrywania chwastow - raz na sesje (cache).

    Po co: wczytanie modelu trwa kilka sekund; @st.cache_resource sprawia, ze dla
    tych samych argumentow model jest tworzony raz, a kolejne loty biora gotowy.
    Wybor po rozszerzeniu pliku:
        .hef        -> HailoDetector (tylko na urzadzeniu z Hailo-8, np. Raspberry Pi)
        inny plik   -> YoloDetector (.pt / .onnx przez Ultralytics)
        brak pliku  -> DummyDetector - atrapa szukajaca plam zieleni (NIE jest detektorem
                       chwastow) + ostrzezenie

    Argumenty:
        model   - sciezka do modelu, np. "models/weights/fieldeye_final.pt"
        prog    - prog pewnosci (0.05-0.9, dom. 0.15)
        wejscie - rozmiar wejscia sieci w px (640 / 960 / 1280)

    Zwraca:
        (detektor, uwaga) - uwaga to tekst ostrzezenia albo None, gdy model wczytany

    Uzywana w:
        zakladka 4 - "Start lotu"
    """
    if model and Path(model).exists() and str(model).lower().endswith(".hef"):
        try:
            return HailoDetector(model, prog), None
        except ImportError as e:
            return DummyDetector(), f"{e} Plik .hef działa tylko na urządzeniu z Hailo. Używam atrapy."
    if model and Path(model).exists():
        try:
            return YoloDetector(model, prog, wejscie), None
        except ImportError as e:
            return DummyDetector(), f"{e} Używam atrapy."
    return DummyDetector(), (f"Nie znaleziono modelu {model}. Używam atrapy, która szuka "
                              "plam zieleni i NIE jest detektorem chwastów.")


def kafelki_wedlug_dni(katalog):
    """Grupuje kafelki WeedsGalore wedlug dnia lotu (data w nazwie pliku).

    Po co: WeedsGalore to jedno pole sfotografowane kilka razy - kafelki z roznych
    dni pokazuja te same miejsca. Lot musi byc nad jednym dniem, inaczej to samo
    miejsce wystapi w siatce dwa razy i mozaika sie nie zlozy.
    Data szukana wyrazeniem regularnym RRRR-MM-DD, np.
    weedsgalore__2023-05-25__2023-05-25_0780.jpg -> "2023-05-25".

    Argumenty:
        katalog - folder z kafelkami, np. "zbior_lotnicze/dataset/images/test"

    Zwraca:
        {dzien: [pliki]} - pliki posortowane po nazwie; bez daty w nazwie -> "bez daty"
        np. {"2023-05-25": [14 plikow], "2023-05-30": [...], ...}

    Uzywana w:
        zakladka 4 - lista dni w selectboxie "Dzien lotu"
    """
    import re
    dni = {}
    for p in sorted(Path(katalog).glob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            m = re.search(r"\d{4}-\d{2}-\d{2}", p.name)
            dni.setdefault(m.group(0) if m else "bez daty", []).append(p)
    return dni


def pole_z_kafelkow(pliki, ile=25, z_etykietami=False):
    """Skleja kafelki w prostokatna siatke - zrodlo do symulacji lotu nad prawdziwymi chwastami.

    Po co: zbiory treningowe to osobne kafelki, a symulator potrzebuje jednego
    duzego zdjecia pola. Kafelki musza pochodzic z JEDNEGO dnia lotu (patrz
    kafelki_wedlug_dni). To nie jest ciagle pole - na stykach kafelkow sa
    skoki - ale chwasty sa prawdziwe i maja prawdziwe etykiety.
    Jak sklada: pierwsze `ile` kafelkow, kolumn = ceil(sqrt(n)), tylko pelne wiersze,
    wszystkie przeskalowane do rozmiaru pierwszego; np. 25 kafelkow -> siatka 5 x 5.
    Z etykietami: ramki z plikow YOLO (cx, cy, w, h - ulamki) przeliczone na piksele
    i przesuniete o polozenie kafelka w siatce.

    Argumenty:
        pliki       - lista sciezek kafelkow (z jednego dnia)
        ile         - najwiecej kafelkow do uzycia (dom. 25)
        z_etykietami - True: zwroc tez prawdziwe ramki chwastow

    Zwraca:
        obraz BGR siatki; z z_etykietami=True: (obraz, [(x0, y0, x1, y1), ...]) w px siatki;
        None - gdy kafelkow mniej niz 4

    Uzywana w:
        zakladka 4 - pole "Kafelki WeedsGalore (test)" po kliknieciu "Start lotu"
    """
    pliki = list(pliki)[:ile]
    if len(pliki) < 4:
        return None
    kol = int(np.ceil(np.sqrt(len(pliki))))
    wier = len(pliki) // kol                      # tylko pelne wiersze
    obrazy = [cv2.imread(str(p)) for p in pliki[:wier * kol]]
    h, w = obrazy[0].shape[:2]
    obrazy = [cv2.resize(o, (w, h)) for o in obrazy]
    pole = np.vstack([np.hstack(obrazy[r * kol:(r + 1) * kol]) for r in range(wier)])
    if not z_etykietami:
        return pole
    # prawdziwe ramki z plikow YOLO obok kafelkow, przeniesione na sklejke
    prawda = []
    for i, p in enumerate(pliki[:wier * kol]):
        et = katalog_etykiet(Path(p).parent) / (Path(p).stem + ".txt")
        if not et.exists():
            continue
        ox, oy = (i % kol) * w, (i // kol) * h
        for linia in et.read_text().split("\n"):
            cz = linia.split()
            if len(cz) >= 5:
                cx, cy, bw, bh = (float(v) for v in cz[1:5])
                prawda.append((ox + (cx - bw / 2) * w, oy + (cy - bh / 2) * h,
                               ox + (cx + bw / 2) * w, oy + (cy + bh / 2) * h))
    return pole, prawda


def prawda_w_kadrze(prawda, kadr_info, kadr, min_widocznosc=0.4, kadr_wys=None):
    """Przelicza prawdziwe ramki chwastow z pola na wspolrzedne jednego kadru.

    Po co: zeby ocenic detekcje na kadrze, trzeba wiedziec, gdzie na NIM sa
    prawdziwe chwasty. Macierz z macierz_kadru() przenosi piksel kadru na pole,
    wiec w druga strone idziemy jej odwrotnoscia (cv2.invertAffineTransform).
    Ramka obrocona razem z kadrem -> jej obwiednia -> przyciecie do krawedzi kadru.
    Ramka zostaje, gdy w kadrze widac co najmniej min_widocznosc jej powierzchni
    (tak samo jak przy kafelkowaniu zbioru).

    Argumenty:
        prawda         - prawdziwe ramki na polu: [(x0, y0, x1, y1), ...] w px pola
        kadr_info      - wpis kadru z prawda.json (srodek, kat_st, zasieg_px, kadr_px)
        kadr           - szerokosc kadru (px)
        min_widocznosc - jaka czesc ramki musi byc w kadrze (dom. 0.4 = 40%)
        kadr_wys       - wysokosc kadru (px); None = z kadr_info albo kwadrat

    Zwraca:
        lista ramek (x0, y0, x1, y1) w px kadru

    Uzywana w:
        zakladka 4 - dla kazdego kadru: zielone ramki i porownanie z wykrytymi
    """
    kh = kadr_wys or (kadr_info.get("kadr_px") or [kadr, kadr])[1]
    M = macierz_kadru(*kadr_info["srodek"], kadr_info["kat_st"], kadr_info["zasieg_px"], kadr, kh)
    Mi = cv2.invertAffineTransform(M)
    out = []
    for x0, y0, x1, y1 in prawda:
        q = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], float) @ Mi.T
        (a0, b0), (a1, b1) = q.min(axis=0), q.max(axis=0)
        pole = (a1 - a0) * (b1 - b0)
        c0, d0, c1, d1 = max(a0, 0), max(b0, 0), min(a1, kadr), min(b1, kh)
        if c1 > c0 and d1 > d0 and pole > 0 and (c1 - c0) * (d1 - d0) >= min_widocznosc * pole:
            out.append((c0, d0, c1, d1))
    return out


def porownaj(wykryte, prawdziwe, iou=0.5):
    """Porownuje wykryte ramki z prawdziwymi na jednym obrazie.

    Po co: zeby policzyc, ile chwastow model znalazl, ile przegapil i ile
    falszywych alarmow dal. Para wykryta-prawdziwa liczy sie jako trafienie, gdy
    ramki nachodza na siebie w co najmniej iou (IoU = czesc wspolna / suma).
    Liczy to pr_przy_progach z models/evaluation.py (ten sam kod co ocena modelu).

    Argumenty:
        wykryte   - [(x0, y0, x1, y1, pewnosc), ...]
        prawdziwe - [(x0, y0, x1, y1), ...]
        iou       - prog nakladania (dom. 0.5)

    Zwraca:
        (trafione, pominiete, falszywe) - trzy liczby calkowite

    Uzywana w:
        zakladka 4 - dla kazdego kadru (opis pod obrazem) i na koncu dla mozaiki
    """
    w = pr_przy_progach({"x": [(0, r[4], *r[:4]) for r in wykryte]},
                        {"x": [(0, *r[:4]) for r in prawdziwe]}, [0.0], iou)[0]
    return w["trafione"], w["pominiete"], w["falszywe"]


def gif_lotu(klatki, mozaika, szer=640, ms=400):
    """Sklada animacje lotu: kadry po kolei, na koncu mozaika z chwastami.

    Po co: plik do pobrania pokazujacy caly lot. Kazda klatka to para
    [kadr | wykryte chwasty] przeskalowana do szerokosci szer; ostatnia to
    mozaika wpisana (wysrodkowana) w ten sam rozmiar, wyswietlana 4 s.

    Argumenty:
        klatki  - lista obrazow BGR (pary kadr | oznaczony), ten sam rozmiar
        mozaika - mozaika z narysowanymi ramkami (BGR) albo None
        szer    - szerokosc GIF-a w px (dom. 640)
        ms      - czas jednej klatki w ms (dom. 400); ostatnia klatka 4000 ms

    Zwraca:
        bytes - plik GIF (petla w nieskonczonosc); None - gdy brak klatek

    Uzywana w:
        zakladka 4 - "Pobierz animacje lotu (GIF)"
    """
    if not klatki:
        return None
    h, w = klatki[0].shape[:2]
    s = szer / w
    kl = [cv2.resize(k, (szer, int(h * s)), interpolation=cv2.INTER_AREA) for k in klatki]
    H, W = kl[0].shape[:2]
    if mozaika is not None:
        f = min(W / mozaika.shape[1], H / mozaika.shape[0])
        m = cv2.resize(mozaika, (int(mozaika.shape[1] * f), int(mozaika.shape[0] * f)), interpolation=cv2.INTER_AREA)
        plotno = np.zeros((H, W, 3), np.uint8)
        y, x = (H - m.shape[0]) // 2, (W - m.shape[1]) // 2
        plotno[y:y + m.shape[0], x:x + m.shape[1]] = m
        kl.append(plotno)
    obrazy = [Image.fromarray(bgr_na_rgb(k)) for k in kl]
    czasy = [ms] * (len(obrazy) - 1) + [4000]
    bufor = io.BytesIO()
    obrazy[0].save(bufor, format="GIF", save_all=True, append_images=obrazy[1:], duration=czasy, loop=0)
    return bufor.getvalue()


ZIELONY, ROZOWY = (60, 200, 60), (255, 0, 255)


def rysuj_ramki(img, ramki, grubosc=2, kolor=ROZOWY, kopia=True):
    """Rysuje prostokaty ramek na obrazie.

    Po co: wspolne rysowanie ramek w zakladce 4 - wykryte chwasty na rozowo,
    prawdziwe na zielono (kolory w BGR: ROZOWY (255, 0, 255), ZIELONY (60, 200, 60)).

    Argumenty:
        img     - obraz BGR
        ramki   - [(x0, y0, x1, y1, ...), ...]; dalsze pola (np. pewnosc) sa pomijane
        grubosc - grubosc linii w px (dom. 2)
        kolor   - kolor w BGR (dom. ROZOWY)
        kopia   - True: rysuj na kopii (oryginal bez zmian); False: rysuj na img
                  (szybciej, gdy kilka warstw ramek na jednym obrazie)

    Zwraca:
        obraz z ramkami (kopia albo ten sam img)

    Uzywana w:
        zakladka 4 - kadry podczas lotu, mozaika (widok "Wykryte / Prawdziwe / Obie"), GIF
    """
    out = img.copy() if kopia else img
    for x0, y0, x1, y1, *_ in ramki:
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), kolor, grubosc)
    return out


def scal_ramki(ramki, prog_iou=0.3):
    """Scala ramki tego samego chwastu z kilku kadrow w jedna.

    Po co: kadry na siebie zachodza, wiec ten sam chwast jest wykrywany na 2-6
    kadrach; po przeniesieniu na mozaike jego ramki leza prawie w tym samym miejscu.
    Zachlanne tlumienie (NMS): ramki od najpewniejszej, ramka zostaje, gdy z kazda
    juz zostawiona nachodzi mniej niz prog_iou. Przy gesto rosnacych chwastach
    moze polaczyc dwa sasiednie w jeden.

    Argumenty:
        ramki    - [(x0, y0, x1, y1, pewnosc), ...] na mozaice
        prog_iou - powyzej tego nakladania ramki uznane za ten sam chwast (dom. 0.3)

    Zwraca:
        lista ramek, ktore zostaly (ten sam format), od najpewniejszej

    Uzywana w:
        zakladka 4 - na_mozaike_wszystkie(): wykryte i prawdziwe ramki na mozaice
    """
    def iou(a, b):
        x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        wsp = (x1 - x0) * (y1 - y0)
        return wsp / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - wsp)
    zostaja = []
    for r in sorted(ramki, key=lambda r: -r[4]):
        if all(iou(r, z) < prog_iou for z in zostaja):
            zostaja.append(r)
    return zostaja


with tab4:
    st.subheader("Lot z detekcją chwastów")
    st.caption("Kadry pojawiają się po kolei, tak jak w trakcie lotu: po lewej zdjęcie, "
               "po prawej wykryte chwasty. Po ostatnim kadrze składana jest mozaika "
               "z chwastami naniesionymi na właściwe miejsca.")
    ust, wyn = st.columns([1, 2.4])
    with ust:
        zr4 = st.radio("Pole", ["Kafelki WeedsGalore (test)", "Wgraj zdjęcie", "Sztuczne pole"], key="lot_zrodlo")
        kat4 = plik4 = None
        if zr4 == "Kafelki WeedsGalore (test)":
            kat4 = st.text_input("Katalog z kafelkami", "zbior_lotnicze/dataset/images/test", key="lot_katalog")
            dni4 = kafelki_wedlug_dni(kat4) if Path(kat4).is_dir() else {}
            dni4 = {d: p for d, p in dni4.items() if len(p) >= 4}
            dzien4 = None
            if dni4:
                dzien4 = st.selectbox("Dzień lotu", list(dni4),
                                      format_func=lambda d: f"{d} ({len(dni4[d])} kafelków)",
                                      index=max(range(len(dni4)), key=lambda i: len(list(dni4.values())[i])),
                                      key="lot_dzien")
            st.caption("Kafelki jednego dnia są sklejane w siatkę — to nie ciągłe pole, ale prawdziwe "
                       "chwasty. Kafelki z różnych dni pokazują te same miejsca pola, dlatego "
                       "lot jest zawsze nad jednym dniem.")
        elif zr4 == "Wgraj zdjęcie":
            plik4 = st.file_uploader("Zdjęcie pola (widok z góry)", type=["jpg", "jpeg", "png"], key="lot_plik")
        else:
            st.caption("Sztuczne pole nie ma prawdziwych chwastów — przydaje się tylko do "
                       "sprawdzenia samego przebiegu.")
        model4 = st.text_input("Model", "models/weights/fieldeye_final.pt", key="lot_model")
        prog4 = st.slider("Próg ufności", 0.05, 0.90, 0.15, 0.05, key="lot_prog")
        wej4 = st.select_slider("Wejście sieci", [640, 960, 1280], 1280, key="lot_wejscie")
        kam4 = st.checkbox("Kamera drona (wysokość lotu i optyka)", False, key="lot_kamera",
                           help="Kadr = tyle pola, ile widzi kamera z danej wysokości; przed "
                                "modelem dzielony na kafelki 1,5 m, jak na dronie.")
        info4 = None
        if kam4:
            kamera4 = KAMERY[st.selectbox("Kamera", list(KAMERY), key="lot_kamera_typ")]
            wys4 = st.number_input("Wysokość lotu [m]", 0.5, 50.0, 5.0, 0.5, key="lot_wysokosc")
            gsd4 = st.number_input("GSD źródła [mm/px]", 0.5, 20.0, GSD_WEEDSGALORE_MM, 0.1, key="lot_gsd",
                                   help="Ile mm pola na piksel obrazu, nad którym lecimy. "
                                        "WeedsGalore: ok. 2,5.")
            v4 = st.number_input("Prędkość [m/s]", 0.5, 15.0, 3.0, 0.5, key="lot_predkosc")
            czas4 = st.select_slider("Czas naświetlania [s]", ["1/250", "1/500", "1/1000", "1/2000", "1/4000"],
                                     "1/1000", key="lot_naswietlanie")
            kadr4 = 600
        else:
            kadr4 = st.slider("Rozmiar kadru [px]", 400, 1000, 600, 50, key="lot_kadr",
                              help="600 px = rozmiar kafelków, na których uczył się model. "
                                   "Inny rozmiar zmienia skalę chwastów na wejściu sieci.")
        pokr4 = st.slider("Pokrycie kadrów", 0.50, 0.85, 0.70, 0.05, key="lot_pokrycie")
        bok4 = st.slider("Pokrycie między liniami", 0.20, 0.80, 0.50, 0.05, key="lot_pokrycie_bok")
        if kam4:
            info4 = opis_lotu(kamera4, wys4, gsd4, v4, 1 / float(czas4.split("/")[1]), pokr4)
            st.caption(f"Zdjęcie: {info4['zasieg_m'][0]:.2f} × {info4['zasieg_m'][1]:.2f} m, "
                       f"GSD kamery {info4['gsd_kamery_mm']} mm/px · kadr w źródle "
                       f"{info4['kadr_px'][0]} × {info4['kadr_px'][1]} px · ok. "
                       f"{info4['kafelkow_na_kadr']} kafelków po {info4['kafelek_px']} px (1,5 m) · "
                       f"rozmycie ruchem {info4['rozmycie_px']} px · zdjęcie co {info4['odstep_zdjec_s']} s. "
                       f"Na dronie: kafelek {info4['kafelek_kamery_px']} px zdjęcia.")
        odst4 = st.slider("Przerwa między kadrami [s]", 0.0, 2.0, 0.3, 0.1, key="lot_odstep")
        zakl4 = st.checkbox("Zakłócenia kamery (obrót, jasność, drgania)", True, key="lot_zaklocenia",
                            help="Wyłączone: kadry to dokładne wycinki bez obrotu i bez zmian "
                                 "jasności — pozwala sprawdzić, ile kosztują same zakłócenia.")
        start4 = st.button("Start lotu", type="primary", key="lot_start")

    with wyn:
        if start4:
            if zr4 == "Kafelki WeedsGalore (test)":
                wynik_pola = pole_z_kafelkow(dni4[dzien4], z_etykietami=True) if dzien4 else None
                zrodlo4, prawda_zr = wynik_pola if wynik_pola else (None, [])
                if zrodlo4 is None:
                    st.error(f"W katalogu {kat4} nie ma co najmniej 4 kafelków z jednego dnia. "
                             "Najpierw zbuduj zbiór lotniczy.")
                    st.stop()
            elif zr4 == "Wgraj zdjęcie":
                if plik4 is None:
                    st.warning("Wgraj zdjęcie.")
                    st.stop()
                zrodlo4, prawda_zr = wczytaj_z_uploadu(plik4), []
            else:
                if info4:
                    kw_, kh_ = info4["kadr_px"]
                    zrodlo4 = zbuduj_pole(int(kw_ * (1 + 5 * (1 - pokr4))) + 40,
                                          int(kh_ * (1 + 2 * (1 - bok4))) + 40, seed=7)
                else:
                    zrodlo4 = zbuduj_pole(2000, 1400, seed=7)
                prawda_zr = []
            jest_prawda = bool(prawda_zr)

            det4, uwaga4 = detektor_lotu(model4, prog4, int(wej4))
            if uwaga4:
                st.warning(uwaga4)

            kat_lotu = katalog_roboczy() / "lot"
            kw4, kh4 = info4["kadr_px"] if info4 else (int(kadr4), 0)
            u4 = SimSettings(tryb="trasa", kadr=int(kw4), kadr_wys=int(kh4), overlap=pokr4, sidelap=bok4,
                            obrot=2.0 if zakl4 else 0.0, ekspozycja=0.03 if zakl4 else 0.0,
                            jitter=0.08 if zakl4 else 0.0,
                            rozmycie_ruch_px=info4["rozmycie_px"] if info4 else 0.0, seed=7)
            try:
                prawda4 = rozloz(zrodlo4, kat_lotu, u4)
            except ValueError:
                H_, W_ = zrodlo4.shape[:2]
                if info4:
                    st.error(f"Pole {W_} × {H_} px (ok. {W_ * gsd4 / 1000:.1f} × {H_ * gsd4 / 1000:.1f} m) "
                             f"jest mniejsze niż jedno zdjęcie z {wys4:g} m "
                             f"({info4['zasieg_m'][0]:.1f} × {info4['zasieg_m'][1]:.1f} m). Użyj większego "
                             "pola (ortomozaika WeedsGalore przez „Wgraj zdjęcie” albo sztuczne pole) "
                             "albo mniejszej wysokości.")
                else:
                    st.error(f"Pole {W_} × {H_} px jest za małe na kadr {kw4} px.")
                st.stop()
            pliki4 = [kat_lotu / "images" / k["plik"] for k in prawda4["kadry"]]
            # kafelki 1,5 m przed modelem, jak na dronie
            det_lot4 = TiledDetector(det4, kafelek=info4["kafelek_px"], zakladka=0.15) if info4 else det4
            # przy wielu kadrach porownujemy tylko sasiadow: ta sama i sasiednia linia trasy
            na_linii = max(sum(1 for k in prawda4["kadry"] if k["linia"] == l)
                           for l in {k["linia"] for k in prawda4["kadry"]})
            zakres4 = 2 * na_linii + 2 if len(pliki4) > 40 else 0

            skl4 = GlobalStitcher(renderuj_co=10 ** 6, zakres_par=zakres4)
            pasek4 = st.progress(0.0, text="Start lotu...")
            podglad4 = st.empty()
            if jest_prawda:
                st.caption("Zielone ramki — prawdziwe chwasty z etykiet zbioru, różowe — wykryte przez model.")
            wykrycia4, prawdy4, rozmiary4, klatki4 = [], [], [], []
            laczenie, suma = 0, [0, 0, 0]
            for i, (p, info_k) in enumerate(zip(pliki4, prawda4["kadry"]), 1):
                img = cv2.imread(str(p))
                ramki = [(d.x0, d.y0, d.x1, d.y1, d.pewnosc) for d in det_lot4.wykryj(img)
                         if d.pewnosc >= prog4]
                gt = prawda_w_kadrze(prawda_zr, info_k, img.shape[1], kadr_wys=img.shape[0]) if jest_prawda else []
                wykrycia4.append(ramki)
                prawdy4.append(gt)
                rozmiary4.append(img.shape[:2])
                laczenie += len(ramki)
                oznaczony = rysuj_ramki(img, gt, 2, ZIELONY) if jest_prawda else img.copy()
                rysuj_ramki(oznaczony, ramki, 2, ROZOWY, kopia=False)
                if jest_prawda:
                    t_, p_, f_ = porownaj(ramki, gt)
                    suma = [suma[0] + t_, suma[1] + p_, suma[2] + f_]
                    opis = f"trafione {t_} · pominięte {p_} · fałszywe {f_}"
                else:
                    opis = f"Wykryte chwasty: {len(ramki)}"
                klatki4.append(zmniejsz(np.hstack([img, oznaczony]), 1600))    # duze kadry: oszczednie z pamiecia
                skl4.dodaj(p)
                with podglad4.container():
                    a, b = st.columns(2)
                    a.image(bgr_na_rgb(img), caption=f"Kadr {i} z {len(pliki4)}", width="stretch")
                    b.image(bgr_na_rgb(oznaczony), caption=opis, width="stretch")
                tekst = f"Kadr {i}/{len(pliki4)} · wykryć łącznie: {laczenie}"
                if jest_prawda:
                    tekst += f" · w kadrach: trafione {suma[0]}, pominięte {suma[1]}, fałszywe {suma[2]}"
                pasek4.progress(i / len(pliki4), text=tekst)
                if odst4:
                    time.sleep(odst4)

            pasek4.progress(1.0, text="Składanie mozaiki...")
            skl4.zakoncz()
            if skl4.mozaika is None:
                st.error("Nie udało się złożyć mozaiki.")
                st.stop()
            def na_mozaike_wszystkie(lista):
                """Przenosi ramki ze wszystkich kadrow na mozaike i scala powtorzone.

                Po co: detekcja dziala na kadrach, a wynik dla rolnika jest na mozaice. Dla kazdej
                ramki 4 rogi przeliczane przez skl4.na_mozaike() (skala robocza, polozenie
                kadru, poczatek mozaiki) -> obwiednia na mozaice; potem scal_ramki(), bo ten sam
                chwast byl na kilku kadrach. Ramki z kadrow, ktore nie weszly do mozaiki, sa pomijane.

                Argumenty:
                    lista - dla kazdego kadru lista ramek [(x0, y0, x1, y1[, pewnosc]), ...]
                            (ta sama kolejnosc co pliki4); bez pewnosci -> 1.0

                Zwraca:
                    lista unikalnych ramek (x0, y0, x1, y1, pewnosc) w px mozaiki

                Uzywana w:
                    zakladka 4 - dla wykrytych (wykrycia4) i prawdziwych (prawdy4) chwastow
                """
                wynik = []
                for p, ramki, rozm in zip(pliki4, lista, rozmiary4):
                    for r in ramki:
                        x0, y0, x1, y1 = r[:4]
                        q = skl4.na_mozaike(p, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], rozm)
                        if q is not None:
                            (mx0, my0), (mx1, my1) = q.min(axis=0), q.max(axis=0)
                            wynik.append((mx0, my0, mx1, my1, r[4] if len(r) > 4 else 1.0))
                return scal_ramki(wynik)
            unikalne = na_mozaike_wszystkie(wykrycia4)
            prawdziwe = na_mozaike_wszystkie(prawdy4) if jest_prawda else None
            st.session_state["lot_wynik"] = {
                "mozaika_czysta": skl4.mozaika, "wykryte": unikalne, "prawdziwe": prawdziwe,
                "kadrow": len(pliki4), "w_mozaice": len(skl4.S or {}),
                "wykryc": laczenie, "unikalnych": len(unikalne),
                "porownanie": porownaj(unikalne, prawdziwe) if jest_prawda else None,
                "detektor": det4.nazwa,
                "kamera": dict(info4, wysokosc=wys4, kamera=kamera4.nazwa) if info4 else None,
                "gif": gif_lotu(klatki4, rysuj_ramki(skl4.mozaika, unikalne, 3)),
            }
            pasek4.progress(1.0, text="Gotowe")

        w4 = st.session_state.get("lot_wynik")
        if w4:
            st.markdown("#### Mozaika z wykrytymi chwastami")
            if w4.get("kamera"):
                k4 = w4["kamera"]
                st.caption(f"{k4['kamera']}, lot na {k4['wysokosc']:g} m: zdjęcie {k4['zasieg_m'][0]:.2f} × "
                           f"{k4['zasieg_m'][1]:.2f} m, ok. {k4['kafelkow_na_kadr']} kafelków 1,5 m na kadr, "
                           f"rozmycie ruchem {k4['rozmycie_px']} px.")
            if w4["w_mozaice"] < 0.8 * w4["kadrow"]:
                st.warning(f"Mozaika złożyła się tylko z {w4['w_mozaice']} z {w4['kadrow']} kadrów. "
                           "Zwykle oznacza to, że to samo miejsce występuje w źródle kilka razy "
                           "albo kadry za mało na siebie zachodzą.")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Kadrów", w4["kadrow"])
            k2.metric("W mozaice", w4["w_mozaice"])
            k3.metric("Wykryć w kadrach", w4["wykryc"])
            k4.metric("Chwastów po scaleniu", w4["unikalnych"])
            if w4["porownanie"]:
                t_, p_, f_ = w4["porownanie"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Prawdziwych chwastów", t_ + p_)
                m2.metric("Trafione", t_, f"{100 * t_ / max(1, t_ + p_):.0f}% znalezionych", delta_color="off")
                m3.metric("Pominięte", p_)
                m4.metric("Fałszywe alarmy", f_)
                st.caption("Liczby na mozaice są przybliżone: ramki z sąsiednich kadrów są scalane, "
                           "co przy gęsto rosnących chwastach może połączyć dwa sąsiednie w jeden.")

            opcje = ["Wykryte", "Prawdziwe", "Obie", "Bez ramek"] if w4["prawdziwe"] is not None \
                else ["Wykryte", "Bez ramek"]
            widok = st.radio("Ramki na mozaice", opcje, horizontal=True, key="lot_widok")
            moz = w4["mozaika_czysta"].copy()
            if widok in ("Prawdziwe", "Obie"):
                rysuj_ramki(moz, w4["prawdziwe"], 2, ZIELONY, kopia=False)
            if widok in ("Wykryte", "Obie"):
                rysuj_ramki(moz, w4["wykryte"], 2, ROZOWY, kopia=False)
            st.image(bgr_na_rgb(moz), width="stretch",
                     caption=f"Detektor: {w4['detektor']}. Ten sam chwast widziany na kilku "
                             "kadrach jest liczony raz.")
            d1, d2 = st.columns(2)
            d1.download_button("Pobierz mozaikę (JPG, aktualny widok)", jpg_bajty(moz),
                               "mozaika_chwasty.jpg", "image/jpeg", key="lot_pobierz")
            if w4["gif"]:
                d2.download_button("Pobierz animację lotu (GIF)", w4["gif"], "lot.gif",
                                   "image/gif", key="lot_gif")
        elif not start4:
            st.info("Ustaw parametry i kliknij **Start lotu**.")