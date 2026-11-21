# fieldeye-detect: wykrywanie chwastów ze zdjęć drona i mapa oprysku

`fieldeye-detect` obserwuje folder, do którego dron zapisuje zdjęcia, przepuszcza każde
nowe zdjęcie przez model chwastów i — gdy obok zdjęcia leży telemetria GPS — zamienia
wykrycia na współrzędne i buduje mapę oprysku. Działa ze źródeł (Python) i jako binarka,
na Macu, Ubuntu i Raspberry Pi 5 z akceleratorem Hailo-8.

```
kamera + autopilot                folder ZDJECIA                     fieldeye-detect                 folder WYNIKI
(dron, Raspberry Pi)   ──►   20261109T142859.998Z.jpg      ──►   kafelki → model → geolokalizacja  ──►   oprysk.geojson / oprysk.csv
                             20261109T142859.998Z.geojson        → łączenie w punkty oprysku           chwasty.geojson, trasa.geojson
                                                                                                       zdjęcia z ramkami
```

## Spis treści
1. [Uruchamianie](#1-uruchamianie)
2. [Telemetria drona (format)](#2-telemetria-drona-format)
3. [Wyniki](#3-wyniki)
4. [Jak liczone są współrzędne](#4-jak-liczone-są-współrzędne)
5. [Zabezpieczenia](#5-zabezpieczenia)
6. [Binarka](#6-binarka)
7. [Raspberry Pi](#7-raspberry-pi)
8. [Symulowany dron i ocena namiarów](#8-symulowany-dron-i-ocena-namiarów)
9. [Mapa kodu](#9-mapa-kodu)
10. [Najczęstsze problemy](#10-najczęstsze-problemy)

---

## 1. Uruchamianie

```bash
fieldeye-detect ZDJECIA WYNIKI MODEL [opcje]                         # binarka
python -m mission.detect_cli ZDJECIA WYNIKI MODEL [opcje]            # ze źródeł (z katalogu projektu)
bash scripts/fieldeye-detect.sh ZDJECIA WYNIKI MODEL [opcje]         # ze źródeł, sam włącza .venv
```

| Argument | Znaczenie |
|---|---|
| `ZDJECIA` | folder ze zdjęciami; w trybie obserwacji powstaje sam, jeśli go nie ma |
| `WYNIKI` | folder na wyniki (powstaje sam); bez niego tylko wynik w terminalu |
| `MODEL` | ścieżka z nazwą pliku: `.hef` (Hailo), `.onnx`, `.pt`; pominięty → model obok binarki albo model końcowy z projektu |

Ścieżki mogą zaczynać się od `~` (także w cudzysłowie).

| Opcja | Domyślnie | Znaczenie |
|---|---|---|
| `--raz` | nie | przetwórz to, co jest w folderze, i zakończ (bez tego: obserwacja do Ctrl+C) |
| `--timeout S` | brak | zakończ po S sekundach bez nowych zdjęć |
| `--prog P` | 0,15 | próg pewności wykrycia (niski celowo: pominięty chwast jest gorszy niż fałszywy alarm) |
| `--kafelek PX` | z telemetrii (1,5 m pola); bez niej 950 | bok kafelka; `0` = całe zdjęcie naraz |
| `--tta` | nie | dodatkowo odbicie lustrzane (ok. 2× wolniej, mniej fałszywych) |
| `--kurs-domyslny ST` | brak | kurs przyjmowany, gdy telemetria go nie podaje (testy w zawisie) |
| `--maks-przechyl ST` | 5 | zdjęcia przechylone bardziej nie trafiają na mapę |
| `--promien-scalania M` | 0,10 | wykrycia bliżej siebie to ten sam chwast |
| `--czekaj-telemetrie S` | 3 | ile czekać na `.geojson` gotowego zdjęcia, zanim pójdzie bez GPS |
| `--stabilnosc S` | 1 | ile sekund rozmiar zdjęcia musi być stały, zanim zostanie wczytane |
| `--jsonl` | nie | dodatkowo `WYNIKI/wyniki.jsonl` (ramki w pikselach, GPS, współrzędne) |
| `--json` | nie | na wyjściu tylko JSON, jedna linia na zdjęcie (dla innych programów) |
| `--szczegoly N` | 0 | wypisz N najpewniejszych wykryć ze zdjęcia |
| `--bez-telemetrii` | nie | nie szukaj plików `.geojson` (same wykrycia) |

Linijka w terminalu dla każdego zdjęcia:

```
[14:29:01] 20261109T142859.998Z.jpg   4624x3472   chwasty:   17   max 0.91   0.84 s   GPS 50.290049, 18.731013   oprysk: 15 pkt (+15)   (w kolejce: 1)
```

`max` — najwyższa pewność na zdjęciu; `(w kolejce: N)` — ile zdjęć czeka (rosnąca liczba: model nie nadąża za kamerą).

Kolejność przetwarzania to kolejność zapisu zdjęć (FIFO). Zdjęcie jest wczytywane dopiero po
pełnym zapisie (rozmiar stały przez `--stabilnosc` s i kompletny JPEG/PNG). Pliki ukryte (`.`, `~`)
i tymczasowe (`.part`) są pomijane, uszkodzone nie zatrzymują obserwacji.

## 2. Telemetria drona (format)

Obok każdego zdjęcia plik o **tej samej nazwie** z rozszerzeniem **`.geojson`**, w tym samym
folderze, zapisywany **atomowo** (plik tymczasowy, potem zmiana nazwy):

```
20261109T142859.998Z.jpg   →   20261109T142859.998Z.geojson
```

Wielkość liter w nazwie musi się zgadzać; rozszerzenie `.geojson` małymi literami. Telemetria może
przyjść przed zdjęciem albo najpóźniej `--czekaj-telemetrie` s po jego zapisie.

```json
{"type": "FeatureCollection",
 "features": [{"type": "Feature",
   "geometry": {"type": "Point", "coordinates": [18.73101322, 50.29004902, 239.73]},
   "properties": {
     "image": "20261109T142859.998Z.jpg", "seq": 1, "flight_id": "20261109T142856Z",
     "time_utc": "2026-11-09T14:28:59.998Z", "fix": true, "fix_quality": 2,
     "speed_mps": 0.02, "course_deg": 7.9,
     "heading_deg": 7.5, "roll_deg": 0.4, "pitch_deg": -1.2,
     "survey": {"agl_m": 2.0, "gsd_mm_per_px": 0.623},
     "camera": {"width": 4624, "height": 3472}}}]}
```

| Pole | Wymagane | Do czego |
|---|---|---|
| `coordinates` `[długość, szerokość]` (WGS84) | **tak** | pozycja drona = środek zdjęcia na polu (kamera patrzy pionowo w dół) |
| `survey.agl_m` | **tak** (albo `gsd_mm_per_px`) | wysokość **nad polem** [m] → skala zdjęcia z modelu kamery Arducam |
| `survey.gsd_mm_per_px` | nie | skala wprost [mm pola na piksel]; ma pierwszeństwo przed wysokością |
| `heading_deg` | **tak**\* | kurs drona: 0 = północ, zgodnie z zegarem; góra zdjęcia = przód drona |
| `roll_deg`, `pitch_deg` | zalecane | kontrola bujania (ponad `--maks-przechyl` zdjęcie poza mapą) |
| `fix`, `fix_quality` | zalecane | `false` / `0` → zdjęcie bez GPS |
| `time_utc`, `image` | nie | czas zapisywany na mapie; dopasowanie punktu do zdjęcia |

\* Gdy brak `heading_deg`, kurs = `course_deg` (kierunek ruchu z GPS), ale tylko przy `speed_mps ≥ 1`:
w zawisie kierunek ruchu to szum, a w locie dron może lecieć bokiem. Bez kursu zdjęcie idzie bez GPS
(do testów: `--kurs-domyslny 0`). Pozostałe pola są pomijane.

Wzorzec prawdziwego pliku: [`tests/dane/telemetria_dron.geojson`](../tests/dane/telemetria_dron.geojson).

## 3. Wyniki

W folderze `WYNIKI`, odświeżane po każdym zdjęciu, zapis atomowy:

| Plik | Kiedy | Zawartość |
|---|---|---|
| `<nazwa zdjęcia>.jpg` | zawsze | zdjęcie z różowymi ramkami wokół chwastów |
| `oprysk.geojson` | z telemetrią | **punkty do oprysku**: jeden punkt na chwast; `id`, `promien_m`, `pewnosc`, `liczba_wykryc`, `liczba_zdjec` |
| `oprysk.csv` | z telemetrią | to samo jako tabela: `id,szerokosc,dlugosc,promien_m,pewnosc,liczba_zdjec` |
| `chwasty.geojson` | z telemetrią | każde wykrycie z każdego zdjęcia; `pewnosc`, `rozmiar_m`, `zdjecie`, `czas`, `niepewnosc_przechylu_m` |
| `trasa.geojson` | z telemetrią | trasa drona (linia) i miejsca zdjęć |
| `wyniki.jsonl` | z `--jsonl` | linia na zdjęcie: ramki w pikselach, GPS, współrzędne wykryć |

Mapę można obejrzeć, przeciągając pliki `.geojson` na [geojson.io](https://geojson.io) albo w QGIS.
Bez telemetrii są tylko zdjęcia z ramkami (bez współrzędnych).

## 4. Jak liczone są współrzędne

Kod: [`mission/spray_map.py`](../mission/spray_map.py).

1. **Skala** (m pola na piksel, GSD): `survey.gsd_mm_per_px / 1000` albo
   `agl · szerokość matrycy / (ogniskowa · szerokość zdjęcia w px)` — Arducam 16 MP: matryca 7,4 mm,
   ogniskowa 5,1 mm (2 m → 0,623 mm/px; 5 m → 1,57 mm/px).
2. **Kafelek** = 1,5 m / skala — model widzi chwasty w takiej wielkości jak przy treningu.
3. **Przesunięcie chwastu od środka zdjęcia** (środek ramki `(px, py)`, zdjęcie `W × H`):
   `dx = (px − W/2) · GSD` (w prawo), `dy = (py − H/2) · GSD` (w dół).
4. **Obrót o kurs θ** (przód drona = góra zdjęcia):
   `wschód = dx·cos θ − dy·sin θ`, `północ = −dx·sin θ − dy·cos θ`.
5. **Metry → stopnie** w lokalnym układzie stycznym do elipsoidy WGS84 (punkt odniesienia: pierwsze
   zdjęcie; na 50° szer. ok. 111 235 m na 1° szerokości i 71 260 m na 1° długości). Błąd przybliżenia
   na polu rzędu setek metrów: ułamki milimetra.
6. **Łączenie**: wykrycie bliżej niż `--promien-scalania` od istniejącego punktu dołącza do niego
   (położenie = średnia ważona pewnością), inaczej nowy punkt. Promień oprysku = połowa największej
   ramki + 5 cm.

Przykład (telemetria z [`tests/dane`](../tests/dane/telemetria_dron.geojson), kurs 90°): chwast w pikselu
(3100, 900) zdjęcia 4624 × 3472 przy 0,623 mm/px → 0,521 m na wschód i 0,491 m na południe od drona →
50,29004461° N, 18,73102053° E.

**Założenia** (świadome uproszczenia): kamera patrzy pionowo w dół (bez poprawki na bujanie), teren płaski,
pozycja z chwili migawki, kamera pod anteną GPS. O dokładności mapy decyduje głównie GPS: bez RTK
(`fix_quality` 1–2) ok. 1 m, z RTK (`fix_quality` 4) kilka cm.

## 5. Zabezpieczenia

| Sytuacja | Reakcja |
|---|---|
| brak `.geojson` po `--czekaj-telemetrie` s | zdjęcie z ramkami, bez współrzędnych (`bez GPS`) |
| plik nieczytelny / nie `FeatureCollection` | jak wyżej, z komunikatem |
| współrzędne poza zakresem albo 0, 0 | bez GPS: „zamieniona kolejność?” / „brak pozycji GPS” |
| `fix = false` albo `fix_quality = 0` | bez GPS: „brak fixa GPS” |
| brak kursu | bez GPS: „dodaj heading_deg…” |
| `agl_m` poza 0,3–150 m | bez GPS: „wysokość npm zamiast nad polem?” |
| `gsd_mm_per_px` poza 0,1–100 | bez GPS: „złe jednostki?” |
| przechył > `--maks-przechyl` | zdjęcie z ramkami, poza mapą (`PRZECHYL … - poza mapa oprysku`) |
| przechył do limitu | na mapie, z `niepewnosc_przechylu_m = agl · tg(kąt)` |
| zdjęcie w trakcie zapisu | czekanie na stały rozmiar i kompletny plik |
| uszkodzone zdjęcie | pominięte (`POMINIETE`), obserwacja trwa |
| folder wyników = folder zdjęć | odmowa uruchomienia |

Nie da się wykryć automatycznie zamiany kolejności współrzędnych w Polsce (50 i 18 mieszczą się w obu
zakresach) — widać to od razu na `trasa.geojson`.

## 6. Binarka

Jeden plik z Pythonem i bibliotekami w środku (PyInstaller); model leży obok albo podaje się go
jako trzeci argument. **Budować na tym systemie, na którym ma działać** (Mac → Mac, Ubuntu → Ubuntu,
Pi → Pi).

```bash
bash scripts/build_binary.sh onnx            # Mac / Ubuntu: ONNX Runtime, bez PyTorcha (~95 MB)
bash scripts/build_binary.sh hailo           # Raspberry Pi z hailo-all: model .hef (~170 MB)
bash scripts/build_binary.sh pt --katalog    # PyTorch; duży, lepiej jako katalog
```

Wynik: `dist/fieldeye-detect`. Ta sama binarka ma też polecenia pomocnicze:

```bash
./fieldeye-detect ZDJECIA WYNIKI MODEL
./fieldeye-detect dron FOLDER --ewis1 KAFELKI        # symulowany dron (rozdział 8)
./fieldeye-detect porownaj OPRYSK PRAWDA             # ocena namiarów
```

| Wariant | `.onnx` | `.hef` | `.pt` |
|---|---|---|---|
| `onnx` | ✅ | ❌ | ❌ |
| `hailo` | ❌ | ✅ | ❌ |
| `pt` | ✅ | ❌ | ✅ |

Model ONNX z modelu końcowego:
`python -m models.cli onnx models/weights/fieldeye_final.pt -o fieldeye_final.onnx --wejscie 1280 --opset 11`.
Samowystarczalny folder (binarka, modele, przykłady): `bash scripts/przygotuj_folder.sh CEL`.

## 7. Raspberry Pi

Na Macu: `bash scripts/spakuj_na_pi.sh --hef ŚCIEŻKA/fieldeye_final_1280_h8.hef` →
`~/Downloads/fieldeye_pi.zip` (kod, HEF, 132 kafelki testowe EWIS1, `na_pi.sh`, `README_PI.txt`).

Na Pi (Raspberry Pi OS 64-bit, AI HAT+, HailoRT 4.23):

```bash
sudo apt install -y hailo-all python3-venv python3-pip python3-opencv python3-numpy unzip
unzip -oq fieldeye_pi.zip && cd fieldeye_pi
bash na_pi.sh sprawdz      # Hailo, Python, OpenCV, numpy, model, dane
bash na_pi.sh test         # 132 kafelki przez .hef (bez GPS)
bash na_pi.sh gps          # symulowany lot z telemetrią + mapa + ocena namiarów
bash na_pi.sh na-zywo      # zdjęcie co 0,7 s, detekcja na bieżąco: czy Hailo nadąża
bash na_pi.sh binarka      # dist/fieldeye-detect
bash na_pi.sh wszystko     # wszystko po kolei
```

Wyniki w `/tmp/fieldeye_*`, logi w `/tmp/fieldeye_*.log` (`/tmp` czyszczony przy restarcie).
`requirements.txt` na Pi **nie jest potrzebny** (PyTorch/Ultralytics służą tylko do treningu).

Pierwsze wyniki na Pi (HEF, 132 kafelki testowe): 4090 wykryć (z `.pt` na Macu: 4313), ok. 0,03 s na kafelek.

Kompilacja HEF (WSL, Hailo DFC 3.33 / Model Zoo 2.17 dla HailoRT 4.23): `scripts/HAILO.md`,
`scripts/compile_hef.py` (nazwa sieci `yolov8n`, własna konfiguracja NMS dla wejścia 1280).

## 8. Symulowany dron i ocena namiarów

```bash
python -m mission.drone_sim lot --ewis1 zbior_ewis1_podzial/dataset/images/test [--interwal 0.7]
python -m mission.detect_cli lot wyniki fieldeye_final.onnx --raz
python -m mission.spray_map wyniki/oprysk.geojson lot_prawda.geojson
```

Symulowany dron składa zdjęcie EWIS1 z kafelków (`--ewis1`; inne źródła: `--pole ZDJECIE`, `--sztuczne`),
kładzie je na mapie (lewy górny róg w `--lat`/`--lon`, północ u góry, 2,5 mm/px), przelatuje nad nim
symulatorem `stitchlive` i co `--interwal` s zapisuje zdjęcie i telemetrię **w formacie drona** (nazwy jak
z kamery, `heading_deg`, `survey.agl_m`, `survey.gsd_mm_per_px`, `roll_deg = pitch_deg = 0`). Prawdziwe
położenia chwastów trafiają do `lot_prawda.geojson`.

Opcje: `--zaklocenia` (obrót, drgania, jasność), `--szum-gps-m 1.5` (GPS bez RTK), `--szum-kursu-st`,
`--numer N` (inne zdjęcie EWIS1), `--lat`/`--lon`.

`spray_map` porównuje punkty oprysku z prawdą (trafienie: bliżej niż 10 cm). Wyniki prób (sztuczne pole):

| Telemetria | Trafione | Błąd namiaru |
|---|---|---|
| idealna (także z obrotem, drganiami, zmianą wysokości) | 100 % | ok. 0,1–0,2 cm |
| GPS z RTK (±3 cm) | 100 % | ok. 3 cm |
| GPS bez RTK (±1,5 m) | 29 % | ten sam chwast wiele razy na mapie |

## 9. Mapa kodu

| Plik | Zawartość |
|---|---|
| [`mission/detect_cli.py`](../mission/detect_cli.py) | program: `argumenty`, `przygotuj`, `kandydaci` (FIFO), `gotowy`/`kompletny`, `telemetria_zdjecia`, `skala`, `ustaw_kafelek`, `przetworz`, `dodaj_do_mapy`, `zapisz_zdjecie`, `main` |
| [`mission/spray_map.py`](../mission/spray_map.py) | `czytaj_telemetrie`, `blad_telemetrii`, `przechyl_st`, `gsd_m`, `piksel_na_metry`, `UkladLokalny`, `MapaOprysku` (łączenie, zapis), `porownaj` |
| [`mission/detector.py`](../mission/detector.py) | `OnnxDetector`, `HailoDetector`, `YoloDetector`, `FlipTTADetector`, `TiledDetector`, `letterbox` |
| [`mission/drone_sim.py`](../mission/drone_sim.py) | symulowany dron |
| [`stitchlive/camera.py`](../stitchlive/camera.py) | model kamery Arducam (zasięg i skala z wysokości) |
| [`scripts/fieldeye_detect_main.py`](../scripts/fieldeye_detect_main.py) | punkt wejścia binarki (detekcja, `dron`, `porownaj`) |
| [`scripts/build_binary.sh`](../scripts/build_binary.sh) | budowa binarki |
| [`scripts/spakuj_na_pi.sh`](../scripts/spakuj_na_pi.sh), [`scripts/na_pi.sh`](../scripts/na_pi.sh) | paczka i testy na Pi |
| [`scripts/przygotuj_folder.sh`](../scripts/przygotuj_folder.sh) | samowystarczalny folder z binarką |
| `tests/test_detect_cli.py`, `test_spray_map.py`, `test_binary.py` | testy (`python -m pytest tests -q`) |

## 10. Najczęstsze problemy

| Objaw | Przyczyna / rozwiązanie |
|---|---|
| każde zdjęcie `bez GPS`, komunikat „brak kursu” | brak `heading_deg` w telemetrii (do testów `--kurs-domyslny 0`) |
| `bez GPS` bez komunikatu | brak `.geojson` o tej samej nazwie (wielkość liter, rozszerzenie, ten sam folder) |
| program czeka i nic się nie dzieje | obserwuje inny folder niż ten, do którego zapisuje kamera — sprawdź ścieżkę w nagłówku |
| `(w kolejce: N)` rośnie | model nie nadąża za kamerą (mniej zdjęć, większy kafelek, bez `--tta`) |
| fałszywe wykrycia z pewnością ok. 0,15 w pomieszczeniu | obraz zupełnie inny niż w treningu; do testów w pokoju `--prog 0.4` |
| binarka: `Exec format error` | zbudowana na innym systemie — zbuduj na tym, na którym działa |
| binarka na Macu zablokowana | `xattr -dr com.apple.quarantine fieldeye-detect` |
| na Pi „Unsupported hef version” | HEF z innej wersji kompilatora niż HailoRT na Pi (4.23 ↔ DFC 3.33) |
| rozmyte zdjęcia w locie | za długi czas naświetlania — w locie ok. 1 ms (`exposure_us ≈ 1000`) |
