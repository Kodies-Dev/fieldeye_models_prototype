# normalizacja — co trzeba zrobić

Cztery zbiory (**WeedsGalore, CoFly-WeedDB, Sudars, WeedMaize**) plus dane własne
mają dać jeden zbiór treningowy w formacie YOLO.

**Kod jest napisany i przetestowany** — opis działania w `HOW_IT_WORKS.md`.
Ten plik to lista zadań, które zostały **dla człowieka**: rzeczy wymagających
pobrania danych, decyzji albo sprawdzenia wzrokiem.

---

## Źródła i opcje (stan aktualny)

| Zbiór | Opcja | Co to | Rola w modelu |
|---|---|---|---|
| WeedsGalore | `--weedsgalore` | dron 5 m, kukurydza, 4 dni, maski | trening, **walidacja i test** (oficjalny podział) |
| Sudars | `--sudars` | z ziemi, warzywa | trening |
| EWIS1 | `--ewis1` | dron 5 m, kukurydza (domyślnie bez sorgo), ramki z XML | trening; odłożony dzień = drugi test |
| PhenoBench | `--phenobench` | dron, burak cukrowy, 3 fazy, maski instancji | trening |
| WeedMaize | `--weedmaize` | z 1,3 m, kukurydza V1–V3 | trening (opcjonalnie) |
| CoFly | `--cofly` | dron, bawełna, oznaczone tylko 3 gatunki | **nieużywany**: nieoznaczone chwasty uczyłyby model, że chwast to tło |
| własne | `--wlasne` | zdjęcia z naszego drona | na przyszły sezon |

| Opcja | Do czego |
|---|---|
| `--tryb-klas chwasty` | jedna klasa „chwast”; uprawy są tłem |
| `--skala ZRODLO=WSP` | zmniejszenie zdjęć do skali drona (2,5 mm/px), np. `ewis1=0.47`, `weedmaize=0.144` |
| `--maks-zdjec ZRODLO=N` | najwyżej N zdjęć ze źródła |
| `--proporcje 1 0 0` | wszystko do treningu (zbiory pomocnicze) |
| `--ewis1-z-sorgo` | EWIS1 także z sorgo i zdjęciami bez nazwy uprawy |
| `--ewis1-test-dni 20220509` | EWIS1: wskazane dni lotu do testu, reszta do treningu |

Po każdej budowie sprawdź w podsumowaniu linijkę `nierozpoznane` (ma być „brak”)
i obejrzyj kilka obrazów w `podglad/`.

Poniżej pierwotna lista zadań przy budowie zbiorów — nadal aktualna jako instrukcja.

## Zanim zaczniesz

```bash
cd stitch-live
pip install -r requirements.txt
python -m pytest tests/test_normalization.py -q     # 29 testów, ok. 10 s
```

Sprawdź, że potok działa na danych przykładowych — bez pobierania czegokolwiek:

```python
from normalization.examples import utworz
from normalization.build import zbuduj
zbuduj(utworz("dane_przykladowe"), "zbior_testowy", kafelek=320)
```

Powinno powstać `zbior_testowy/dataset/data.yaml` i katalog `podglad/`.
Jeśli tak — środowisko jest gotowe i możesz pobierać prawdziwe dane.

---

## Zadanie 1. Pobrać zbiory

| Zbiór | Skąd | Rozmiar |
|---|---|---|
| WeedsGalore | https://doidata.gfz.de/weedsgalore_e_celikkan_2024/ (plik `weedsgalore-dataset`) | 0,4 GB |
| — podział autorów | jest w archiwum (`weedsgalore-dataset\splits\`), adapter znajduje go sam | — |
| CoFly-WeedDB | https://zenodo.org/record/6697343 | ~436 MB |
| Sudars | https://data.mendeley.com/datasets/nj4vtk4tt6/1 | — |
| WeedMaize | https://doi.org/10.5281/zenodo.5106795 (train_set, validation_set, test_set) | ~5,6 GB |

Struktura, której oczekują adaptery:

```
dane/
├── weedsgalore-dataset/2023-05-25/images/<id>_R.png ... _RE.png
│                                 /semantics/<id>.png
│                                 /instances/<id>.png
├── CoFly-WeedDB/images/, labels/, labels_1d/
├── sudars/dataset/raw images/*.jpg + annotations/*.xml
└── weedmaize/train_set/, validation_set/, test_set/   (jpg + xml razem)
```

Jeśli po rozpakowaniu układ jest inny, nie przestawiaj plików na siłę — zgłoś,
poprawimy adapter.

---

## Zadanie 2. Potwierdzić gatunki w WeedsGalore

**To jest jedyne zadanie, którego nie da się zrobić kodem.**

Kod autorów zbioru nazywa chwasty tylko `weed_1`…`weed_4` (wartości 2–5 w masce),
bez nazw gatunków. Przypisanie w `taxonomy.yaml` pochodzi z publikacji, ale
**nie zostało zweryfikowane co do kolejności**.

Co zrobić: otworzyć publikację (arXiv:2502.13103), znaleźć tabelę klas i sprawdzić,
czy kolejność zgadza się z tym, co jest w `taxonomy.yaml`:

```
# obecne zalozenie - DO POTWIERDZENIA
2 -> amaranthus_retroflexus   (szarłat szorstki)
3 -> echinochloa_crus_galli   (chwastnica jednostronna)
4 -> galinsoga_parviflora     (żółtlica drobnokwiatowa)
5 -> chwast_inny              (pozostałe chwasty)
```

Jeśli kolejność jest inna — poprawić w `taxonomy.yaml` i zbudować zbiór ponownie.

**Dlaczego to ważne:** przy trybie binarnym (chwast/uprawa) pomyłka nic nie zmienia,
bo wszystkie cztery i tak są chwastami. Ale przy trybie `gatunki` model uczyłby się
złych nazw, a wyniki per klasa byłyby bez sensu.

---

## Zadanie 3. Zbudować zbiór i obejrzeć wynik

```bash
python -m normalization.build \
    --weedsgalore dane/weedsgalore-dataset \
    --cofly       dane/CoFly-WeedDB \
    --sudars      dane/sudars/dataset \
    --weedmaize   dane/weedmaize \
    -o zbior --tryb-klas binarnie
```

Program wypisze na koniec podsumowanie. **Trzy rzeczy wymagają reakcji:**

| Komunikat | Co zrobić |
|---|---|
| nierozpoznane etykiety | dopisać do `taxonomy.yaml`, zbudować ponownie — zadanie 4 |
| `BŁĄD: te same grupy w kilku częściach` | błąd w kodzie, zgłosić — **nie trenować** |
| `UWAGA: oficjalny podział autorów rozdziela te grupy` | decyzja — zadanie 5 |

---

## Zadanie 4. Uzupełnić nierozpoznane etykiety

Jeśli lista nie jest pusta, każda pozycja oznacza obiekty, które **po cichu
zniknęłyby ze zbioru**.

Dopisać alias do istniejącego gatunku w `taxonomy.yaml`:

```yaml
galinsoga_parviflora:
  aliasy: [GP, "Galinsoga parviflora", GASPA, weed_3, NOWY_ALIAS]
```

albo dodać nowy gatunek, jeśli to coś, czego nie ma w słowniku:

```yaml
nowy_gatunek:
  lacina: Nazwa lacinska
  polska: nazwa polska
  rola: chwast          # albo uprawa
  w_polsce: pospolity   # pospolity / regionalnie / nie wystepuje
  aliasy: [SKROT, "Pelna nazwa"]
```

Gatunki, których w Polsce nie ma, dostają dodatkowo `zwin_do: chwast_inny` —
nie ma sensu trenować osobnej klasy pod roślinę, której model nigdy nie zobaczy.

Powtarzać, aż lista będzie pusta.

---

## Zadanie 5. Zdecydować o podziale WeedsGalore

Oficjalny podział autorów (104/26/26) jest **warstwowany po datach lotu** — każda
data ma kafelki w train, val i test. To jedno pole sfotografowane cztery razy.

Dwie możliwości, obie obsługiwane przez kod:

| Wariant | Komenda | Zaleta | Wada |
|---|---|---|---|
| podział autorów (domyślny) | — | wyniki porównywalne z publikacją | te same miejsca pola mogą być w dwóch częściach |
| własny podział po grupach | `--wlasny-podzial` | rozłączność gwarantowana | brak porównywalności z publikacją |

**Do rozstrzygnięcia:** czy te same fragmenty pola z różnych dni trafiają do dwóch
części. Jeśli tak, pierwszy wariant zawyża mAP.

Jeśli nie ma czasu tego sprawdzać — użyć `--wlasny-podzial`. Bezpieczniejsze.

---

## Zadanie 6. Obejrzeć podgląd ramek

**Nie pomijać tego kroku.**

```
zbior/podglad/
```

Obejrzeć kilkanaście obrazów z **każdego** zbioru. Ramki mają otaczać rośliny,
nie przesuwać się względem nich i nie być poprzesuwane o stałą wartość.

Błąd w konwersji — przesunięcie o pół piksela, pomylona kolejność osi, zła wartość
maski — **jest zupełnie niewidoczny w metrykach**, a psuje cały trening. Ten
kwadrans oszczędza dni zastanawiania się, czemu model się nie uczy.

Jeśli coś wygląda źle: zanotować, z którego zbioru i jak dokładnie (ramki
przesunięte? za duże? w złych miejscach?) i zgłosić.

---

## Zadanie 7. Decyzje do podjęcia

Nie ma jednej dobrej odpowiedzi — zależą od ustaleń z opiekunem.

| Decyzja | Opcje | Domyślne |
|---|---|---|
| liczba klas | `binarnie` / `chwasty` / `gatunki` | `binarnie` na pierwszą iterację |
| rozmiar kafelka | zależy od rozdzielczości docelowej kamery | 896 |
| udział pustych kafelków | 10–20% | 15% |
| zbiór testowy | wszystko / tylko dane własne | `--tylko-wlasne-do-testu`, gdy będą dane własne |

**Dlaczego binarnie na start:** znacznie łatwiej uzyskać użyteczne mAP, a do
zabiegu wystarczy wiedzieć *gdzie*, niekoniecznie jaki gatunek. Podział binarny
przenosi się też między uprawami — klasa „kukurydza" jest bezużyteczna na burakach.

---

## Zadanie 8. Dane własne (gdy będą)

Format: YOLO (`.txt` + `classes.txt`) albo Pascal VOC. Adapter rozpozna sam.

```bash
python -m normalization.build ... --wlasne dane/moje_misje --tylko-wlasne-do-testu
```

**Ważne:** katalog powinien mieć podkatalogi na misje — nazwa podkatalogu staje się
grupą podziału. Bez tego kadry z jednego przelotu rozjadą się po train i test,
a mAP będzie zawyżone.

```
moje_misje/
├── 2026-05-12_pole_A/   images/ + labels/
└── 2026-05-20_pole_B/   images/ + labels/
```

---

## Checklista przed przekazaniem zbioru do treningu

- [ ] Lista nierozpoznanych etykiet jest pusta
- [ ] Ramki w `podglad/` obejrzane dla każdego zbioru i wyglądają poprawnie
- [ ] Brak komunikatu `BŁĄD: te same grupy w kilku częściach`
- [ ] Decyzja o podziale WeedsGalore podjęta świadomie (zadanie 5)
- [ ] `build_info.json` zachowany razem ze zbiorem
- [ ] Kontrola przeszła:

```bash
python -m models.cli sprawdz  zbior/dataset/data.yaml
python -m models.cli rozmiary zbior/dataset --split train
```

Pierwsza komenda musi zwrócić „Dane wyglądają poprawnie". Druga pokaże rozkład
rozmiarów obiektów — przy udziale małych powyżej 50% kafelkowanie jest konieczne.

---

## Czego **nie** trzeba robić

Napisane i przetestowane — nie pisać od nowa:

- adaptery do czterech formatów (maski, Pascal VOC, YOLO),
- słownik gatunków z aliasami i kodami EPPO,
- deduplikacja skrótem percepcyjnym,
- podział po grupach, deterministyczny i odporny na dodanie nowych misji,
- kafelkowanie z odrzucaniem szczątkowych ramek,
- dobór pustych kafelków do zadanego udziału,
- eksport do YOLO, `data.yaml`, podgląd ramek, `build_info.json`.

Szczegóły działania i uzasadnienia poszczególnych decyzji: `HOW_IT_WORKS.md`.