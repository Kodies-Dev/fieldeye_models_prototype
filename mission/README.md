# misja — pełny przebieg lotu

Realizuje scenariusz z założeń projektu:

1. **W locie** — każde zdjęcie sprawdzane osobno zaraz po zrobieniu: detekcja
   chwastów i przeliczenie ich na współrzędne geograficzne.
2. **Po wylądowaniu** — scalenie powtórzeń, mapa zachwaszczenia i mozaika.

---

## Kluczowa decyzja: mapowanie nie wymaga mozaiki

Żeby nanieść chwast na mapę, **nie trzeba składać zdjęć**. Wystarczy pozycja
GPS, kurs i wysokość tego jednego zdjęcia:

```
piksel w kadrze
   → przesunięcie od środka kadru w pikselach
   → × GSD (metry na piksel, z wysokości i optyki)
   → obrót o kurs drona
   → dodanie do pozycji GPS
   → współrzędne geograficzne
```

Dzięki temu:

- **etap w locie jest lekki** — jedno zdjęcie w pamięci naraz, stała zajętość
  niezależnie od długości misji,
- **składanie mozaiki może poczekać** — jest ciężkie obliczeniowo, a mapa
  zabiegowa i tak jest potrzebna dopiero po wylądowaniu,
- **awaria zasilania nie kasuje misji** — wyniki dopisywane są do pliku po
  każdym zdjęciu.

Mozaika służy człowiekowi do oglądania. Opryskiwacz potrzebuje współrzędnych.

---

## Szybki start

```bash
pip install -r requirements.txt     # pyproj i piexif sa w pliku

# oba etapy naraz, na gotowym katalogu zdjęć
python -m mission.cli pelna zdjecia/ -o wynik --zdjecia zdjecia/

# osobno
python -m mission.cli wlocie  zdjecia/ -o wynik          # etap 1
python -m mission.cli polocie wynik/wyniki.jsonl -o wynik  # etap 2
```

Bez podanego modelu YOLO używana jest **atrapa** — patrz niżej.

### Tryb rzeczywisty (w trakcie lotu)

```bash
# terminal 1 — program czeka na zdjęcia
python -m mission.cli lot naplyw/ -o wynik --telemetria telemetria.csv --timeout 20

# terminal 2 — kapsuła (albo symulacja) wrzuca zdjęcia do katalogu
```

Zdjęcie jest brane dopiero, gdy jego rozmiar przestanie rosnąć — inaczej program
złapałby plik w połowie zapisu.

---

## Skąd program wie, gdzie było zdjęcie

Dwa źródła, drugie nadpisuje pierwsze:

| Źródło | Co daje | Uwaga |
|---|---|---|
| EXIF zdjęcia | pozycja GPS, wysokość **npm**, kurs | zapisują drony komercyjne |
| plik CSV z kapsuły | pozycja, wysokość **nad terenem**, kurs, przechylenia | dokładniejszy — barometr mierzy wysokość nad polem |

Format CSV:

```csv
plik,lat,lon,agl,kurs,przechyl,pochylenie,czas
IMG_0001.jpg,50.26450,19.02310,30.0,90.0,0.4,-0.8,1723900000
```

Wymagane: `plik`, `lat`, `lon`. Reszta opcjonalna — brakujące wartości brane
są z konfiguracji.

---

## Konfiguracja

```yaml
kamera:                      # Arducam 64MP Hawkeye (FIELDEYE); domyślnie w kodzie: Raspberry Pi HQ
  szer_matrycy_mm: 7.4       # z karty katalogowej sensora
  ogniskowa_mm: 5.1          # RZECZYWISTA, nie ekwiwalent 35 mm

lot:
  agl_domyslne: 5.0          # wysokość nad terenem, gdy brak w metadanych (FIELDEYE: 5 m)
  wysokosc_terenu_npm: 0.0   # do przeliczenia wysokości GPS na AGL
  kurs_domyslny: 0.0
  korekta_kursu: 0.0         # montaż kamery względem osi drona

detekcja:
  model: modele_sezon_nadprob/model.pt   # .pt / .onnx / .hef; null = atrapa testowa
  prog_pewnosci: 0.15        # dobrany na walidacji
  wejscie_sieci: 1280        # jak w treningu; przy .hef wbudowane w plik
  kafelkowanie: true
  kafelek: 950               # ok. 1,5 m pola dla Arducam 16 MP z 5 m

mapa:
  promien_scalania_m: 0.35
  min_pewnosc_na_mape: 0.30
```

`--config plik.yaml`. Wszystkie sekcje opcjonalne.

**Ogniskowa** to najczęstsze źródło błędu: podanie ekwiwalentu 35 mm zamiast
rzeczywistej daje kilkukrotnie złą skalę, a więc złe współrzędne wszystkich
detekcji.

---

## Detektor

Wymienny — potok nie wie, jakiego modelu używa. Wszystkie klasy są w `detector.py`
i zwracają te same obiekty `Detection`.

| Klasa | Do czego |
|---|---|
| `YoloDetector` | model `.pt` albo `.onnx` przez Ultralytics (komputer) |
| `HailoDetector` | model `.hef` na akceleratorze Hailo-8 (Raspberry Pi 5 + AI HAT+); sam wpisuje obraz w kwadrat wejścia (*letterbox*) i przelicza ramki z NMS Hailo; wymaga tylko `hailo_platform`, `numpy`, `opencv` |
| `TiledDetector` | opakowanie: tnie zdjęcie na kafelki, scala wyniki, usuwa duplikaty z zakładki |
| `FlipTTADetector` | opakowanie: detekcja na oryginale i odbiciu poziomym, łączona przez WBF (TTA) — w modelu docelowym ok. 30% mniej fałszywych alarmów |
| `EnsembleDetector` | kilka modeli naraz, łączonych przez WBF |
| `DummyDetector` | **nie jest detektorem chwastów** (patrz niżej) |

`zbuduj_detektor(cfg)` wybiera klasę po rozszerzeniu pliku modelu (`.hef` → Hailo),
a `zbuduj_z_plikow(modele, tta=...)` składa ensemble i TTA.

### Gotowe konfiguracje

| Plik | Do czego |
|---|---|
| `configs/mission_wg_sudars.yaml` | komputer: model `.pt`, wejście 1280, próg 0,15 |
| `configs/mission_hailo.yaml` | Raspberry Pi z Hailo-8: model `.hef`, próg 0,15 |

Domyślne wartości w `config.py`: próg **0,15**, wejście **1280** (tak model był
uczony i oceniany). Kafelek dobieraj tak, żeby obejmował ok. **1,5 m pola** —
na takiej skali uczył się model. Dla kamery Arducam 64MP w trybie 16 MP z 5 m
to ok. 950 px (`stitchlive/camera.py`).

### Po co kafelkowanie

Zdjęcie z kamery ma kilka tysięcy pikseli, wejście sieci 640–1280 px. Zmniejszenie
całego zdjęcia sprawia, że kilkunastopikselowy chwast znika. Kafelkowanie zachowuje
rozdzielczość — kosztem kilkunastu–kilkudziesięciu uruchomień sieci na zdjęcie
(na Hailo-8 to ułamek sekundy).

### O atrapie — wprost

`DummyDetector` szuka zwartych plam zieleni odstających od otoczenia. Na
prawdziwym polu zaznaczy tak samo uprawę jak chwast — **nie odróżnia gatunków
i nie ma nic wspólnego z detekcją chwastów**. Istnieje tylko po to, żeby dało się
uruchomić cały potok bez wytrenowanego modelu. Jej detekcje mają klasę `atrapa`.

---

## Symulowany lot z detekcją

`sim_flight.py` — wspólny dla zakładki 4 interfejsu i komendy `symulacja`:
cięcie pola na kadry (symulator), detekcja na każdym kadrze, mozaika, ramki na
mozaice i porównanie z prawdziwymi ramkami.

```bash
python -m mission.cli symulacja --model modele_sezon_nadprob/model.pt --tta --prog 0.15
python -m mission.cli symulacja --model a.pt b.pt --zaklocenia -o wynik.json     # ensemble, z zakłóceniami kamery
```

Dla każdego dnia lotu (kafelki testowe WeedsGalore) wypisuje czułość i precyzję
czterech sposobów liczenia chwastów na mozaice: zwykłe scalanie, łączenie widoków
(`fusion.py`, noisy-OR: słabe wykrycia z kilku kadrów się sumują), łączenie
z głosowaniem i niski próg z głosowaniem.

**Wynik pomiaru:** łączenie widoków nie dało zysku netto na prawdziwych
chwastach (w maju trochę więcej, w czerwcu trochę mniej, precyzja niższa),
więc nie wchodzi do konfiguracji docelowej.

---

## Wyniki

### Etap w locie: `wyniki.jsonl`

Jeden wiersz = jedno zdjęcie, dopisywany od razu po przetworzeniu:

```json
{"kadr": {"plik": "IMG_0001.jpg", "szer": 4056, "wys": 3040,
          "lat": 50.26450, "lon": 19.02310, "agl": 30.0, "kurs": 90.0},
 "gsd_m_px": 0.00775,
 "detekcje": [{"bbox": [1204, 880, 1290, 966], "klasa": "chwast",
               "pewnosc": 0.91, "lat": 50.26452, "lon": 19.02314,
               "rozmiar_m": [0.67, 0.67]}],
 "czas_s": 0.34}
```

### Etap po wylądowaniu

| Plik | Zawartość |
|---|---|
| `chwasty.geojson` | punkty roślin w WGS84 — otwiera QGIS, Folium, geopandas |
| `slad_lotu.geojson` | trasa drona — sprawdzenie, czy misja objęła całe pole |
| `mapa.html` | podgląd w przeglądarce, bez instalowania czegokolwiek |
| `mozaika.jpg` | złożony obraz pola (gdy podano `--zdjecia`) |
| `raport.json` | liczby podsumowujące misję |

### Scalanie powtórzeń

Przy pokryciu 60–80% ten sam chwast jest widoczny na dwóch–trzech zdjęciach.
Scalanie łączy detekcje bliższe niż `promien_scalania_m` (tej samej klasy)
w jedną roślinę i liczy, na ilu zdjęciach była widoczna.

**Liczba obserwacji jest użyteczna sama w sobie** — roślina widziana trzy razy
jest pewniejsza niż widziana raz.

---

## Dokładność — co zmierzono

Test `test_znacznik_trafia_we_wlasciwe_miejsce` sprawdza cały łańcuch na
znacznikach umieszczonych w **znanych** miejscach obrazu.

Przy GSD 1 cm/px i symulowanym locie na 30 m: **średni błąd 3 cm, maksymalny
5,5 cm** — czyli kilka pikseli. Reszta to zaokrąglenie pozycji GPS w EXIF.

To mierzy poprawność geometrii, nie dokładność w terenie. W rzeczywistości
dochodzi błąd odbiornika GPS (bez korekcji różnicowej: 2–5 m), który jest
o dwa rzędy większy niż błąd przeliczeń.

---

## Ograniczenia

- **Zakłada nadir i płaski teren.** Odczyty przechylenia i pochylenia są
  wczytywane i zapisywane, ale jeszcze nieużywane do korekty. Przy wyraźnym
  przechyle drona kadr jest trapezem, nie prostokątem.
- **Dokładność ograniczona przez GPS**, nie przez obliczenia. Moduł LC29H(DA) bez
  korekcji daje ok. 1–2 m; RTK (poprawki NTRIP przez LTE) dałby centymetry.
- **Brak kompasu i IMU** na dronie FIELDEYE: kurs trzeba brać z kierunku ruchu
  w GPS (zdanie RMC), co działa tylko przy locie po prostej ze stałą prędkością.
- **Wysokość musi być nad terenem**, nie nad poziomem morza. Barometr trzeba
  skalibrować na poziom pola przed startem.
- **Brak korekty zniekształceń obiektywu.** Przy szerokim kącie błąd rośnie ku
  brzegom kadru.

---

## Struktura

```
mission/
├── geo.py            GSD, UTM, piksel → współrzędne  ← rdzeń
├── telemetry.py     odczyt EXIF i CSV
├── detector.py       YOLO, Hailo, kafelki, TTA, ensemble, atrapa
├── fusion.py         łączenie wykryć z wielu zdjęć (noisy-OR, głosowanie)
├── sim_flight.py     symulowany lot z detekcją (zakładka 4 i komenda symulacja)
├── in_flight.py         etap 1: kadr po kadrze, zapis JSONL
├── post_flight.py        etap 2: scalanie, GeoJSON, mapa HTML, mozaika
├── config.py   wszystkie parametry
└── cli.py            wiersz poleceń
```

Testy: `tests/test_mission.py`, `test_hailo.py`, `test_fusion.py`, `test_inference.py`.