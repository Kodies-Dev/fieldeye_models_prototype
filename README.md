# stitch-live — FIELDEYE

Oprogramowanie warstwy wizyjnej projektu **FIELDEYE**: wykrywanie chwastów ze
zdjęć z drona (Raspberry Pi 5 + Hailo-8, kamera Arducam 64MP, lot na 5 m)
i nanoszenie ich na mapę pola.

```
   zbiory publiczne ─► normalization/  formaty zbiorów → jeden zbiór treningowy YOLO
                              │
                              ▼
                        models/        trening, ocena według dni, eksport ONNX
                              │
                              ▼  scripts/HAILO.md: ONNX → .hef (Linux x86 / WSL)
   zdjęcia z drona ─► mission/         fieldeye-detect: detekcja (YOLO / ONNX / Hailo, kafelki),
                              │        współrzędne chwastów, mapa oprysku GeoJSON
                              ▼
                        stitchlive/    składanie zdjęć w mozaikę, symulator lotu,
                                       model kamery drona
   app.py: interfejs graficzny (Streamlit) — rozkład zdjęcia, składanie, lot z detekcją
```

## Stan

| Część | Stan |
|---|---|
| detektor chwastów | model końcowy `models/weights/fieldeye_final.pt` (YOLOv8n, wejście 1280, WeedsGalore + Sudars + EWIS1): na zwykłej kamerze RGB (EWIS1, dzień niewidziany w treningu) AP50 0,817, czułość 78%, precyzja 84%; na WeedsGalore AP50 0,714 (z TTA) — `models/README.md` |
| słaby punkt | najmłodsze siewki (WeedsGalore, 25 maja): ok. 32% — ograniczeniem jest ich wielkość w pikselach |
| symulator | lot z kamerą drona (5 m, Arducam), kafelki 1,5 m, rozmycie ruchem, porównanie z prawdziwymi ramkami |
| składanie | metoda `global` (własna), PSNR 22–25 dB na symulowanych lotach |
| Hailo-8 | `fieldeye_final_1280_h8.hef` skompilowany (`scripts/compile_hef.py`, jeden kontekst, 5,4 MB; w `model_fin.zip`); sprawdzenie na Pi: `bash scripts/check_hef.sh` |
| do zrobienia | pomiar HEF na Pi względem `.pt`; zdjęcia z naszego drona do dostrojenia modelu |
| testy | ok. 210, wszystkie przechodzą (`tests/README.md`) |

Nazwy pakietów, plików i klas są angielskie, opisy po polsku.

## Szybki start

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
caffeinate -i bash scripts/final_model.sh
python -m pytest tests -q
streamlit run app.py
```

| Co | Gdzie |
|---|---|
| dane i normalizacja | `normalization/README.md` |
| trening, ocena, wyniki, eksport | `models/README.md` |
| detekcja w locie (binarka), detektory, Hailo | `mission/README.md`, `docs/fieldeye-detect.md`, `scripts/HAILO.md` |
| składanie, symulator, kamera | sekcje niżej, `stitchlive/README.md` |
| skrypty | `scripts/README.md` |

---

## Moduł stitchlive — składanie zdjęć

Łączenie zdjęć z przelotu drona w mozaikę — w czasie rzeczywistym, na bazie
OpenCV. Zawiera też **symulator lotu**: bierze jedno duże zdjęcie, tnie je na
kadry tak, jakby robił je dron (w różnych, także losowych miejscach), składa
z powrotem i porównuje wynik z oryginałem.

To jest **Program 1** z planu projektu — weryfikacja, czy materiał daje się
złożyć. Georeferencja i optymalizacja pamięci to Program 2.

---

## Składanie — szybki start

Polecenia `python -m stitchlive.cli ...` uruchamiaj **z katalogu projektu**.
Po `pip install -e .` możesz używać krótko `stitchlive ...` z dowolnego miejsca.

```bash
pip install -r requirements.txt

# rozłóż -> złóż -> oceń, na sztucznym polu (nic nie trzeba pobierać)
python -m stitchlive.cli demo

# to samo na własnym zdjęciu, cięcie losowe z zakłóceniami
python -m stitchlive.cli demo zdjecie.jpg --max-wymiar 2400 \
    --tryb-ciecia losowy --liczba 30 --obrot 6 --rozmycie 0.2 --szum 3
```

Wynik w katalogu `demo/`:

| Plik | Co to |
|---|---|
| `symulacja/podglad_trasy.jpg` | oryginał z zaznaczonymi kadrami i kolejnością lotu |
| `symulacja/images/` | wycięte kadry — tak, jakby zrobił je dron |
| `symulacja/prawda.json` | prawdziwe położenie, kąt i zasięg każdego kadru |
| `mozaika/mozaika_biezaca.jpg` | złożona mozaika |
| `ocena/ocena_nalozenie.jpg` | mozaika nałożona na oryginał |

---

## GUI (Streamlit)

```bash
streamlit run app.py
```

Otwiera się w przeglądarce, cztery zakładki:

| Zakładka | Co robi |
|---|---|
| **1. Rozkład zdjęcia** | wgrywasz zdjęcie (albo generujesz sztuczne pole), ustawiasz tryb lotu i zakłócenia, klikasz *Rozłóż* — widzisz trasę na oryginale i pierwsze kadry |
| **2. Kadry do pobrania** | galeria kadrów, podgląd pojedynczego z jego prawdziwym położeniem; pobieranie: same kadry (ZIP), kadry + `prawda.json` + podgląd trasy (ZIP), pojedynczy kadr |
| **3. Składanie** | kadry z zakładki 1 albo własne (zdjęcia lub ZIP); mozaika rośnie na żywo kadr po kadrze; ocena względem oryginału; pobieranie: mozaika (JPG), animacja składania (GIF), mozaika + kroki + raport (ZIP) |
| **4. Lot z detekcją** | symulowany lot z modelem (domyślnie `models/weights/fieldeye_final.pt`, wejście 1280): wykrycia na każdym kadrze, prawdziwe ramki z kafelków zbioru (`zbior_lotnicze/...`: kafelki jednego dnia sklejane w siatkę), mozaika z chwastami, tryb kamery drona, GIF z lotu |

Zestawy zakłóceń w zakładce 1:

| Zestaw | Obrót | Wysokość | Jasność | Rozmycie | Szum | Winieta |
|---|---|---|---|---|---|---|
| Czyste cięcie | 0 | 0 | 0 | 0 | 0 | 0 |
| Realistyczny dron | 3° | 3% | 5% | 10% kadrów | 2 | 0,30 |
| Trudne warunki | 6° | 5% | 7% | 25% kadrów | 4 | 0,35 |

„Trudne warunki" są na granicy możliwości metody — przy mocniejszych
zakłóceniach kadry zaczynają odpadać, co widać w metryce *W mozaice*.

**Winietowanie** (ciemniejsze rogi) jest typowe dla kamer dronowych i
przyciemnia akurat strefy nakładania kadrów. Metoda `global` wyrównuje
lokalnie kontrast (CLAHE) przed szukaniem punktów — bez tego przy winiecie
0,3 liczba dopasowanych par spadała ze 147 do 35.

---

## Symulator lotu — rozkład jednego zdjęcia

```bash
python -m stitchlive.cli rozloz [OBRAZ] [opcje] -o symulacja
```

Bez podanego obrazu generowane jest sztuczne pole (rzędy uprawy, chwasty,
kamienie).

### Dwa tryby cięcia

| `--tryb-ciecia` | Jak tnie |
|---|---|
| `trasa` (domyślnie) | przelot „kosiarką" — linie tam i z powrotem, z małym losowym odchyleniem pozycji |
| `losowy` | losowe miejsca cięcia; każdy kolejny kadr leży w losowym kierunku od poprzedniego, ale zawsze z zadanym minimalnym pokryciem, żeby dało się go dopasować |

### Zakłócenia

| Opcja | Symuluje | Przykład |
|---|---|---|
| `--obrot` | odchylenie kursu drona (odchylenie std. w stopniach) | `--obrot 5` |
| `--obrot-linii` | co druga linia obrócona o 180° (dron zawraca) | |
| `--skala` | zmiana wysokości lotu (ułamek) | `--skala 0.05` |
| `--ekspozycja` | zmiana jasności między kadrami | `--ekspozycja 0.06` |
| `--rozmycie` | prawdopodobieństwo rozmycia ruchowego kadru | `--rozmycie 0.2` |
| `--szum` | szum matrycy (poziomy jasności) | `--szum 3` |
| `--winieta` | ciemniejsze rogi kadru, 0–1 | `--winieta 0.3` |
| `--jitter` | losowe odchylenie pozycji (tryb trasa) | `--jitter 0.1` |

Pozostałe: `--kadr` (rozmiar kadru w px), `--zasieg` (ile px oryginału obejmuje
kadr), `--overlap`, `--sidelap`, `--liczba` (tryb losowy), `--seed`.

### Wypuszczanie kadrów jak w locie

```bash
# terminal 1 - składanie na bieżąco
python -m stitchlive.cli watch naplyw -m global --renderuj-co 4 -o wyniki

# terminal 2 - kadry trafiają do katalogu co 0,5 s
python -m stitchlive.cli rozloz zdjecie.jpg --tryb-ciecia losowy \
    --na-zywo naplyw --odstep 0.5
```

---

## Model końcowy — co w nim jest

Jeden skrypt od surowych danych do modelu i paczki dla Hailo: `scripts/final_model.sh`.

| Element | Co robi | Gdzie |
|---|---|---|
| dane | WeedsGalore + Sudars + EWIS1 (zdjęcia ×0,47 do skali drona, 2,5 mm/px) | `normalization/` |
| sezonowość | kafelki 25 maja ×3, 30 maja ×2 w treningu — najmłodsze siewki są najtrudniejsze | `models/season.py` |
| trening | jeden etap, YOLOv8n, wejście 1280, 100 epok, SGD lr0 0,01 | `models/plan.py`, `models/training.py` |
| ocena | AP50 i czułość/precyzja przy progu 0,15, osobno dla każdego dnia; z TTA (odbicie poziome, WBF) | `models/evaluation.py`, `mission/detector.py` |
| test | dzień EWIS1 2022-05-16 (niewidziany w treningu) i test WeedsGalore | `wyniki/model_koncowy/` |

Wyniki i przebieg eksperymentów, które doprowadziły do tego przepisu: `models/README.md`.

## Kamera drona w symulatorze

Symulator może ciąć kadry tak, jak robi to kamera FIELDEYE: **Arducam 64MP
Hawkeye** (matryca 7,4 × 5,55 mm, obiektyw 5,1 mm), skierowana pionowo w dół.
Z wysokości lotu i optyki liczy, jaki kawałek pola obejmuje zdjęcie, i wycina
z obrazu źródłowego dokładnie tyle. Kod: `stitchlive/camera.py`.

| Lot z 5 m, tryb 16 MP | |
|---|---|
| zdjęcie obejmuje | 7,25 × 5,44 m pola |
| GSD kamery | 1,57 mm/px |
| kadr w źródle WeedsGalore (2,5 mm/px) | 2902 × 2176 px |
| kafelek przed modelem | 600 px źródła = 1,5 m pola = kafelek treningowy (na dronie: 956 px zdjęcia) |
| rozmycie ruchem przy 3 m/s i 1/1000 s | 1,2 px |
| zdjęcie co | 0,73 s przy pokryciu 70% |

Kadr ma rozdzielczość źródła, a nie kamery: powiększenie źródła nie dodałoby
szczegółów. Przed modelem kadr jest dzielony na kafelki obejmujące 1,5 m pola,
tak jak na dronie, więc chwasty mają na wejściu sieci tę samą wielkość co w treningu.

```bash
python -m stitchlive.cli rozloz --wysokosc 5 --pole-w 8000 --pole-h 4400 -o symulacja_5m
python -m stitchlive.cli rozloz zdjecie.jpg --wysokosc 5 --gsd-zrodla 2.5 --predkosc 3 --naswietlanie 0.001
```

W interfejsie: zakładka 4, pole **„Kamera drona (wysokość lotu i optyka)”**.
Pole musi być większe niż jedno zdjęcie (z 5 m: ponad 7,3 × 5,4 m); sklejka
kilku kafelków testowych WeedsGalore jest za mała. Do lotu z prawdziwą kamerą
nadaje się ortomozaika WeedsGalore (wgrana przez „Wgraj zdjęcie”) albo sztuczne pole.

## Metody składania

### `global` — domyślna

Pary kadrów liczy OpenCV (ORB, test proporcji, RANSAC). Położenie **wszystkich**
kadrów naraz wyznacza jeden układ równań liniowych: przekształcenie
podobieństwa (obrót + skala + przesunięcie) jest liniowe w parametrach, więc
każda para punktów daje dwa równania. Kadr pierwszy jest układem odniesienia.
Błąd nie kumuluje się wzdłuż trasy.

**Dlaczego osobna metoda:** pipeline `cv2.detail` — i wzorcowy `cv2.Stitcher`
w trybie SCANS — przy kadrach obróconych o kilka stopni dają rozmytą mozaikę,
choć pojedyncze pary dopasowują się z błędem 0,1–0,8 px. Bez obrotu problemu nie
ma, bo same przesunięcia sumują się poprawnie w dowolnej kolejności.

Zabezpieczenia wbudowane w metodę:

- pary z za małym poparciem RANSAC albo absurdalną skalą są odrzucane,
- pary, które po rozwiązaniu nie pasują do reszty (błąd > 3 px), są usuwane
  i układ jest rozwiązywany ponownie,
- kadr niepołączony z resztą nie trafia do mozaiki, zamiast jej psuć;
  składana jest największa połączona grupa, nawet jeśli pierwszy kadr odpadł,
- rozmyty kadr, na którym domyślny detektor nic nie widzi, dostaje drugą próbę
  z niższym progiem.

### `detail` — pipeline `cv2.detail`

Model afiniczny (to samo co `cv2.Stitcher` SCANS, rozłożone na bloki).
Działa dobrze przy kadrach bez obrotu; ma okno przesuwne (`--okno N`)
dające stały czas dodania kadru.

### `basic` — `cv2.Stitcher`

Po każdym zdjęciu przelicza wszystko od nowa. Punkt odniesienia.

---

## Zmierzone wyniki

PSNR mozaiki względem oryginału (powyżej ~25 dB — bardzo dobra zgodność).

| Przypadek | `detail` / `cv2.Stitcher` | `global` |
|---|---|---|
| sztuczne pole, bez obrotu | 27,8 dB | 27,8 dB |
| sztuczne pole, obrót 8° | 19,7 dB (rozmyte) | 26,7 dB |
| prawdziwe zdjęcie z drona, trasa, 48 kadrów | — | 25,2 dB, 48/48 kadrów |
| prawdziwe zdjęcie, cięcie losowe + obrót 6° + zmiana wysokości + rozmycie + szum, 30 kadrów | — | 23,3 dB, 30/30 kadrów |

Prawdziwe zdjęcie: `DJI_0018.JPG` ze zbioru ODMdata *brighton_beach*
(las, droga, trawnik — drzewa to trudna, powtarzalna tekstura).

### Czas

48 kadrów, prawdziwe zdjęcie:

| Ustawienie | Czas |
|---|---|
| blokowe wyrównanie ekspozycji, 3000 cech | 117 s |
| wyrównanie per kadr, 1500 cech (domyślnie) | 10,5 s |

Jakość w obu przypadkach praktycznie ta sama (24,6 dB). Blokowe wyrównanie
skaluje się kwadratowo z liczbą nakładających się kadrów — zostało jako
`--bloki-ekspozycji`.

Tryb na żywo (16 kadrów co 0,5 s, render co 4 kadry): każde odświeżenie
mozaiki 0,1–0,45 s.

---

## Użycie

### `batch` — cały katalog naraz

```bash
python -m stitchlive.cli batch KATALOG -o wyniki
```

### `watch` — obserwacja katalogu w trakcie lotu

```bash
python -m stitchlive.cli watch KATALOG --renderuj-co 4 -o wyniki
```

Wątek w tle obserwuje katalog, wątek główny składa. Plik jest brany dopiero,
gdy przestanie rosnąć — inaczej program złapałby zdjęcie w trakcie zapisu.
Po zakończeniu strumienia mozaika jest renderowana jeszcze raz, żeby objąć
ostatnie kadry.

### `ocen` — porównanie mozaiki z oryginałem

```bash
python -m stitchlive.cli ocen mozaika.jpg oryginal.jpg --prawda prawda.json
```

Mozaika ma własny układ współrzędnych, więc najpierw jest dopasowywana do
oryginału, potem liczone są: skala, pokrycie, średni błąd jasności i PSNR.

### Najważniejsze opcje składania

| Opcja | Opis |
|---|---|
| `-m, --metoda` | `global` (domyślnie), `detail`, `basic` |
| `--megapix` | rozdzielczość robocza (domyślnie 0,6 Mpx) |
| `--zakres-par N` | dopasowuj kadr tylko z N poprzednimi; 0 = z każdym |
| `--renderuj-co N` | odświeżaj mozaikę co N kadrów |
| `--cechy` | liczba punktów na kadr (domyślnie 1500) |
| `--szwy` | wyznaczanie szwów — usuwa widoczne granice, wolniej |
| `--bloki-ekspozycji` | blokowe wyrównanie jasności — wolno |
| `--bez-ekspozycji` | bez wyrównania jasności |
| `--detektor` | `ORB` (domyślnie) albo `SIFT` |

Dla przelotu „kosiarką" `--zakres-par` warto ustawić na co najmniej dwie
długości linii — kadr z sąsiedniej linii leży w kolejności daleko.

---

## Wygenerowane zdjęcia pola

```bash
python scripts/generate_field.py dane_testowe --liczba 6 --uprawa mix
```

Realistyczne (proceduralne) zdjęcia pola z drona — marchew, kukurydza,
burak — razem z etykietami chwastów w formacie YOLO. Można je pociąć
symulatorem i złożyć z powrotem albo użyć do sprawdzenia potoku YOLO.
Szczegóły: `scripts/README.md`.

---

## Podfoldery

| Folder | Co zawiera |
|---|---|
| `mission/` | **detekcja w locie**: `fieldeye-detect` (binarka na Pi) — nowe zdjęcia → chwasty → punkty oprysku z telemetrii GeoJSON; symulowany dron |
| `normalization/` | **trzy zbiory → zbiory YOLO**: adaptery, ujednolicenie klas, deduplikacja, podział bez przecieków, kafelkowanie |
| `models/` | **trening, ocena i eksport** do ONNX i kalibracji pod Hailo-8 |

Każdy ma własne README. `normalization/` ma dodatkowo `HOW_IT_WORKS.md` (opis kodu)
obok `README.md` (plan: co trzeba zrobić i dlaczego).

```bash
# detekcja na zdjęciach z lotu (telemetria GeoJSON obok zdjęć) -> mapa oprysku
python -m mission.detect_cli zdjecia/ wyniki/ models/weights/fieldeye_final.pt

# dane: zbiory publiczne -> zbiór treningowy
python -m normalization.build --weedsgalore ... --sudars ... -o zbior

# model: kontrola zbioru przed treningiem (bez ultralytics)
python -m models.cli sprawdz zbior/dataset/data.yaml
```

Kluczowa własność: **mapowanie chwastu na współrzędne nie wymaga mozaiki** —
wystarczy GPS, kurs i wysokość pojedynczego zdjęcia. Dlatego detekcja działa w locie,
zdjęcie po zdjęciu.

W module `models` sam trening wymaga pakietu `ultralytics` (licencja AGPL-3.0);
plan treningu i kontrola `data.yaml` działają bez niego.


---

## Hailo-8 (Raspberry Pi 5 + AI HAT+)

Na dronie model działa jako plik `.hef` na akceleratorze Hailo-8. Trening, ocena
i symulator na komputerze używają `.pt`; `.hef` powstaje na końcu z ONNX.

| Plik | Co to |
|---|---|
| `models/weights/fieldeye_final.pt` | model końcowy (PyTorch), wejście 1280 |
| `fieldeye_final_1280_h8.hef` (w `model_fin.zip`) | ten sam model skompilowany dla Hailo-8, HailoRT 4.23 |

```
Mac:           python -m models.cli onnx ... --wejscie 1280 --opset 11, models.cli kalibracja ...
WSL (Ubuntu):  python scripts/compile_hef.py model.onnx kalibracja_1280 -o fieldeye_final_1280_h8.hef --rozmiar 1280
Raspberry Pi:  bash scripts/check_hef.sh       → akcelerator, HEF, szybkość, dokładność na teście EWIS1 względem .pt
```

- `scripts/compile_hef.py` używa kompilatora Hailo (DFC 3.33.0, Model Zoo 2.17.0) bezpośrednio, z własną
  konfiguracją NMS (obraz 1280, jedna klasa, próg 0,05, do 300 ramek) — samo `hailomz compile` ma NMS dla 640 i próg 0,2,
- `mission/detector.py` → `HailoDetector`: binarka i `ocen-detektor` wybierają go same dla pliku `.hef`,
- `python -m models.cli ocen-detektor model.pt|model.onnx|model.hef data.yaml`: ta sama ocena dla każdego formatu,
- na Pi: `./fieldeye-detect ZDJECIA WYNIKI fieldeye_final_1280_h8.hef` (`docs/fieldeye-detect.md`, kafelek 950 px = 1,5 m pola z 5 m).

## Pełna dokumentacja

Nazwy pakietów, plików i klas są angielskie. Dokumenty dla zespołu (Word, na Dysku): opis projektu, model i konwersja do HEF, zbiory danych.

| Plik | Zawartość |
|---|---|
| `README.md` (ten plik) | przegląd projektu, moduł stitchlive, uruchomienie |
| `mission/README.md` | detekcja w locie, detektory, symulowany lot |
| `docs/fieldeye-detect.md` | binarka `fieldeye-detect`: uruchamianie, telemetria, mapa oprysku, Raspberry Pi |
| `normalization/README.md` | **lista zadań** do wykonania przy przygotowaniu danych |
| `normalization/HOW_IT_WORKS.md` | opis działania kodu normalizacji |
| `models/README.md` | trening, ocena, wyniki, eksport do Hailo |
| `stitchlive/README.md` | moduły pakietu składającego |
| `scripts/README.md` | model końcowy, Hailo, binarka, generator pola |
| `tests/README.md` | co sprawdza który plik testów |

## Co działa, a co nie — wprost

| Element | Stan |
|---|---|
| składanie mozaiki (metoda `global`) | działa, zmierzone na prawdziwych zdjęciach z drona |
| symulator lotu i ocena względem oryginału | działa |
| georeferencja detekcji (piksel → współrzędne) | działa (`fieldeye-detect`, symulowany dron: `test_spray_map.py`) |
| normalizacja zbiorów (WeedsGalore, Sudars, EWIS1) | działa |
| plan treningu, kontrola danych | działa |
| **detektor chwastów** | **model końcowy** — na zwykłej kamerze RGB AP50 0,817, czułość 0,78; na WeedsGalore AP50 0,714 (`models/README.md`) |
| trening, ocena modelu | **działa** — Windows (CUDA) i Mac (MPS) |
| eksport ONNX | działa, z kontrolą zgodności z `.pt` |
| Hailo-8: `HailoDetector`, `.hef` | **plik `.hef` skompilowany**; pomiar na Pi (`check_hef.sh`) do zrobienia |
| test na innej kamerze (RGB) | zrobiony (`scripts/final_model.sh`): bez EWIS1 w treningu model nic nie wykrywał, z nim 78% |
| georeferencja mozaiki (world file) | **niezrobione** |
| korekta przechylenia z IMU | odczytywane — tylko do odrzucania przechylonych zdjęć, **bez korekty** |


## Testy

```bash
python -m pytest tests -q
```

Ok. 210 testów; lista plików i tego, co sprawdzają: `tests/README.md`.

---

## Struktura

```
stitchlive/
├── global_stitcher.py   metoda global - pary OpenCV + globalne wyrównanie
├── detail.py     pipeline cv2.detail, model afiniczny, okno przesuwne
├── basic.py      opakowanie cv2.Stitcher
├── flight_sim.py    symulator lotu - cięcie jednego zdjęcia na kadry
├── camera.py     model kamery drona (Arducam 64MP) - kadr, kafelek, rozmycie
├── evaluation.py      porównanie mozaiki z oryginałem
├── field.py       generator sztucznego pola
├── source.py     źródła obrazów (katalog obserwowany, tryb wsadowy)
└── cli.py        wiersz poleceń
mission/           fieldeye-detect: detekcja, mapa oprysku, Hailo; symulowany dron (mission/README.md)
models/            trening, ocena, eksport                                  (models/README.md)
normalization/     zbiory publiczne → YOLO                                  (normalization/README.md)
scripts/           model końcowy, Hailo, binarka, narzędzia                 (scripts/README.md)
tests/             ok. 210 testów                                           (tests/README.md)
app.py             GUI w Streamlit
```

---

## Ograniczenia

- **Brak georeferencji** — mozaika jest w układzie pierwszego kadru, bez
  współrzędnych geograficznych. To zadanie Programu 2 (GNSS/IMU).
- **Płaski teren, zdjęcia z nadiru** — model podobieństwa nie uwzględnia
  perspektywy ani różnic wysokości terenu.
- **Pamięć** — wszystkie kadry robocze są trzymane w pamięci; przy bardzo
  długich misjach trzeba obniżyć `--megapix`.
- **Czas dopasowania rośnie z liczbą kadrów**, jeśli `--zakres-par` = 0.
- **Plamy jasności** na jednolitych powierzchniach (asfalt) przy wyrównaniu
  per kadr — pomaga `--szwy` albo `--bloki-ekspozycji`.