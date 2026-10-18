# normalizacja — jak działa kod

Opis działającego kodu. Plan „co trzeba zrobić" i uzasadnienia poszczególnych
decyzji są w `README.md` — ten plik opisuje, jak to zostało zrealizowane.

---

## Uruchomienie

```bash
python -m normalization.build \
    --weedsgalore dane/weedsgalore-dataset \
    --cofly       dane/CoFly-WeedDB \
    --sudars      dane/sudars/dataset \
    --weedmaize   dane/weedmaize \
    --wlasne      dane/moje_misje \
    -o zbior --tryb-klas binarnie
```

Można podać dowolny podzbiór źródeł — brakujące są pomijane z ostrzeżeniem.

Bez żadnych danych na dysku:

```python
from normalization.examples import utworz
from normalization.build import zbuduj
zbuduj(utworz("dane_przykladowe"), "zbior", kafelek=320)
```

Imitacje mają tę samą strukturę katalogów i te same formaty co prawdziwe zbiory,
więc pozwalają sprawdzić kod przed pobraniem kilku gigabajtów.

---

## Co powstaje

```
zbior/
├── dataset/
│   ├── images/train|val|test/
│   ├── labels/train|val|test/      format YOLO
│   └── data.yaml                   gotowe do ultralytics
├── manifest.jsonl                  po kafelkowaniu
├── manifest_przed_kafelkami.jsonl  przed — do podglądu i debugowania
├── podglad/                        ramki narysowane na zdjęciach
├── duplikaty.json                  co odrzucono i dlaczego
└── build_info.json                 parametry budowy, do odtworzenia zbioru
```

---

## Kolejność kroków i dlaczego taka

```
adaptery → deduplikacja → mapowanie klas → podział → kafelkowanie → eksport
```

Dwie kolejności są istotne:

**Deduplikacja przed podziałem.** Gdyby duplikat przeszedł do podziału, jego
kopie mogłyby trafić do treningu i testu jednocześnie.

**Kafelkowanie po podziale.** Kafelki dziedziczą split ze zdjęcia źródłowego.
Kafelkowanie wcześniej rozrzuciłoby kafelki tego samego zdjęcia po różnych
częściach — to ten sam przeciek, tylko w mniejszej skali.

---

## Moduły

| Plik | Odpowiada za |
|---|---|
| `manifest.py` | `Record` i `LabeledObject` — wspólny format pośredni; zapis/odczyt JSONL |
| `taxonomy.py` + `taxonomy.yaml` | słownik gatunków z aliasami; mapowanie nazw na klasy |
| `adapters.py` | po jednym adapterze na zbiór, wszystkie zwracają listę `Record` |
| `steps.py` | deduplikacja, podział, kafelkowanie, kontrole |
| `export.py` | manifest → format YOLO, `data.yaml`, podgląd ramek |
| `build.py` | spina całość, CLI |
| `examples.py` | miniaturowe imitacje czterech zbiorów do testów |

---

## Manifest — format pośredni

Zamiast czterech konwerterów „zbiór → YOLO" jest jeden format pośredni.
Adapter wie tylko, jak czytać swój zbiór; reszta potoku operuje na `Record`.

```python
@dataclass
class Record:
    id: str          # jednoznaczny w całym zbiorze łączonym
    obraz: str       # ścieżka (WeedsGalore: prefiks bez _R.png)
    szer: int
    wys: int
    zrodlo: str      # weedsgalore / cofly / sudars / weedmaize / wlasne
    grupa: str       # klucz podziału — dzień lotu, misja, katalog
    domena: str      # uav / ground — do treningu etapowego
    wyczerpujace: bool   # czy zaznaczono WSZYSTKIE chwasty w kadrze
    split: str
    split_oficjalny: bool
    kafelek: list    # [x, y] początku kafelka
    obiekty: list    # ramki w pikselach BEZWZGLĘDNYCH
```

Ramki trzymane są w pikselach, nie znormalizowane. Normalizacja do 0–1 następuje
dopiero w eksporcie — dzięki temu kafelkowanie jest zwykłym odejmowaniem
przesunięcia, bez przeliczania w tę i z powrotem.

Dodanie piątego zbioru to napisanie jednego adaptera. Reszta potoku nie wie,
ile jest źródeł.

---

## Taksonomia

Jeden gatunek w trzech zbiorach bywa zapisany na trzy sposoby. Słownik w
`taxonomy.yaml` sprowadza je do wspólnego klucza:

```yaml
galinsoga_parviflora:
  lacina: Galinsoga parviflora
  polska: żółtlica drobnokwiatowa
  rola: chwast
  w_polsce: pospolity
  aliasy: [GP, "Galinsoga parviflora", GASPA, weed_3]
```

Dopasowanie jest odporne na drobne różnice zapisu (wielkość liter, spacje,
podkreślenia) i radzi sobie z kodami EPPO z fazą wzrostu: `ZEAMX_V1` nie pasuje
wprost, więc próbowana jest część przed podkreśleniem — `ZEAMX`. Faza zapisywana
jest osobno w polu `faza`.

**Nazwy nierozpoznane nie są cicho pomijane** — trafiają do licznika i na koniec
do raportu. Pusta lista oznacza, że wszystko zmapowano.

### Trzy tryby klas

| `--tryb-klas` | Klasy wyjściowe |
|---|---|
| `binarnie` | `chwast`, `uprawa` — zalecane na pierwszą iterację |
| `chwasty` | sam `chwast`, uprawy pomijane |
| `gatunki` | pełne nazwy gatunków, rzadkie zwinięte przez `zwin_do` |

Gatunki rzadkie w Polsce (sorgo aleppo, cibora okrągła) mają w YAML-u wpis
`zwin_do: chwast_inny` — w trybie `gatunki` nie dostają własnej klasy, bo model
i tak nigdy ich nie zobaczy na polu.

---

## Adaptery

| Zbiór | Format wejściowy | Na co trzeba uważać |
|---|---|---|
| WeedsGalore | kanały `_R/_G/_B/_NIR/_RE` osobno, maski `semantics/` i `instances/` | ramki z maski **instancji**, nie semantycznej |
| CoFly | maski kolorowe `labels/` albo liczbowe `labels_1d/` | 165 z 366 zdjęć nie ma chwastów — to przykłady negatywne |
| Sudars | Pascal VOC | skróty (`GP`) albo pełne nazwy łacińskie |
| WeedMaize | Pascal VOC | kody EPPO z fazą (`ZEAMX_V1`) |
| własne | YOLO `.txt` albo Pascal VOC | nazwy klas z `classes.txt` lub `data.yaml` obok |

**Maska instancji zamiast semantycznej:** w masce semantycznej dwie stykające się
rośliny tego samego gatunku mają tę samą wartość i zlałyby się w jedną ramkę.
Maska instancji nadaje każdej roślinie własny numer.

**Dane własne w formacie YOLO** mają klasy jako indeksy liczbowe. Adapter szuka
`classes.txt` albo `data.yaml` w katalogu; bez tego każda klasa trafiałaby do
raportu jako nierozpoznana nazwa `0`, `1`, `2`.

---

## Deduplikacja

Zwykła suma kontrolna nie wystarczy: po przeskalowaniu albo rekompresji plik jest
inny, a obraz ten sam. Używany jest **skrót percepcyjny** (dHash, 64 bity):
obraz sprowadzany do skali szarości 9×8 px, potem porównywane są sąsiednie piksele
w wierszach — wynikiem jest ciąg bitów „jaśniejszy / ciemniejszy".

Dwa obrazy uznawane są za duplikaty, gdy różnią się na co najwyżej `--prog-dedup`
bitach (domyślnie 5). Przy konflikcie zostaje zbiór o wyższym priorytecie:

```
wlasne > weedsgalore > cofly > sudars > weedmaize
```

Odrzucone pary trafiają do `duplikaty.json` — można sprawdzić, czy to faktycznie
duplikaty, a nie dwa podobne zdjęcia tego samego pola.

---

## Podział train / val / test

Podział idzie **po grupach**, nie po pojedynczych zdjęciach. Grupa to dzień lotu
(WeedsGalore), misja (dane własne) albo katalog źródłowy.

Przydział jest deterministyczny — z funkcji skrótu nazwy grupy:

```python
h = hashlib.sha256(f"{sol}:{grupa}".encode()).digest()
x = int.from_bytes(h[:8], "big") / 2**64
split = "train" if x < 0.70 else ("val" if x < 0.85 else "test")
```

Zaleta: **dodanie nowej misji nie przetasowuje istniejącego podziału**. Przy
losowaniu każdy nowy zbiór oznaczałby inny przydział wszystkiego, co już było.

### Oficjalne podziały autorów

Domyślnie rekordy, które mają podział od autorów zbioru (WeedsGalore), zachowują
go — wyniki są wtedy porównywalne z publikacją. Takie rekordy są oznaczane
`split_oficjalny=True`.

Kontrola `sprawdz_podzial()` rozróżnia dwa przypadki, bo znaczą co innego:

| Kategoria | Znaczenie |
|---|---|
| `bledy` | grupa, którą **my** dzieliliśmy, trafiła do dwóch części — błąd w kodzie, nie powinien wystąpić nigdy |
| `oficjalne` | grupę rozdzielił podział autorów |

Drugi przypadek występuje realnie: **oficjalny podział WeedsGalore (104/26/26)
jest warstwowany po datach**, więc każda data ma kafelki w train, val i test.
To jedno pole sfotografowane cztery razy — trzeba sprawdzić, czy te same miejsca
pola nie trafiają do dwóch części.

Jeśli wolisz rozłączność kosztem porównywalności z publikacją:

```bash
python -m normalization.build ... --wlasny-podzial
```

Opcja `--tylko-wlasne-do-testu` przenosi do treningu wszystko, co nie pochodzi
z danych własnych — zbiór testowy zostaje wtedy wyłącznie z docelowego pola.

---

## Kafelkowanie

Zdjęcie z kapsuły ma ~4000 px, wejście YOLO 640 px. Przeskalowanie całości
zmniejszyłoby chwast z 40 px do 6 px. Dlatego kadr jest cięty na kafelki
w oryginalnej rozdzielczości, z zakładką (domyślnie 10%).

Dwie rzeczy, które kod załatwia za Ciebie:

**Ramki przycięte na krawędzi.** Obiekt na granicy kafelka zostaje tylko wtedy,
gdy w kafelku jest co najmniej `min_widocznosc` (domyślnie 40%) jego pierwotnej
powierzchni. Szczątkowe ramki uczą model błędnych wzorców.

**Udział pustych kafelków liczony globalnie.** Kafelki bez obiektów to przykłady
negatywne — sama uprawa, goła gleba, resztki pożniwne. Bez nich model reaguje na
każdą nieregularność.

Udział jest liczony **na całym zbiorze**, nie na pojedynczym zdjęciu:

```python
ile = round(len(z_obiektami) * udzial / (1 - udzial))
```

Wcześniejsza wersja dobierała puste osobno dla każdego zdjęcia z minimum jednego,
przez co zdjęcie z jednym obiektem wymuszało jeden pusty kafelek i realny udział
skakał do 50% zamiast zadanych 15%.

---

## Eksport

Ramki przeliczane są z pikseli bezwzględnych na format YOLO:

```python
cx = (x0 + x1) / 2 / szerokosc
cy = (y0 + y1) / 2 / wysokosc
w  = (x1 - x0) / szerokosc
h  = (y1 - y0) / wysokosc
```

Kafelki są wycinane z obrazu źródłowego dopiero na tym etapie — do tego momentu
w manifeście jest tylko ścieżka i przesunięcie, więc potok nie trzyma w pamięci
setek wycinków.

### Podgląd ramek — zrób to przed treningiem

`podglad/` zawiera zdjęcia z narysowanymi ramkami i nazwami klas.

**Obejrzyj kilkanaście sztuk.** Błąd w konwersji — przesunięcie o pół piksela,
pomylona kolejność osi, zła wartość maski — jest niewidoczny w metrykach, a psuje
cały trening. Ten kwadrans oszczędza dni.

---

## build_info.json

Zapisywane są: źródła, wszystkie parametry, statystyki, lista klas, liczba
odrzuconych duplikatów, nierozpoznane etykiety i wynik kontroli podziału.

Bez tego po kilku tygodniach nie odtworzysz, na czym trenowany był model, który
dał najlepszy wynik — a to informacja wymagana w dokumentacji projektowej.

---

## Co zgłasza potok

Na koniec wypisywane są trzy rzeczy wymagające reakcji:

| Komunikat | Co zrobić |
|---|---|
| nierozpoznane etykiety | dopisać do `taxonomy.yaml` i zbudować ponownie |
| `BLAD: te same grupy w kilku częściach` | błąd w kodzie — zgłosić, nie trenować |
| `UWAGA: oficjalny podział autorów rozdziela te grupy` | sprawdzić nakładanie miejsc albo użyć `--wlasny-podzial` |

---

## Testy

```bash
python -m pytest tests/test_normalization.py -q
```

29 testów. Najważniejsze dotyczą rzeczy niewidocznych w metrykach:

- aliasy trzech zapisów żółtlicy prowadzą do jednej klasy,
- kody EPPO z fazą wzrostu mapują się na gatunek,
- ramka po kafelkowaniu wraca na właściwe miejsce w oryginale,
- podział nie przetasowuje się po dodaniu nowej grupy,
- grupa nigdy nie trafia do dwóch części,
- przeliczenie na format YOLO daje poprawne wartości.
