# Makefile - skroty do najczestszych komend projektu FIELDEYE.
#
# Po co: zamiast pamietac dlugie komendy wystarczy `make <cel>`, np. `make model`.
#
# Cele:
#   install    - biblioteki z requirements.txt + pytest (w aktywnym .venv)
#   test       - wszystkie testy (tests/)
#   model      - normalizacja -> trening -> ocena -> paczka ONNX (scripts/final_model.sh)
#   gui        - aplikacja Streamlit (app.py)
#   dane       - male sztuczne zdjecia z drona do demo skladania -> dane/test/
#   demo       - sklada mozaike z dane/test (metoda detail, ze szwami) -> wyniki/
#   demo-live  - skladanie na zywo: obserwuje dane/naplyw, do ktorego zdjecia podrzuca
#                scripts/simulate_flight.sh w drugim terminalu
#   clean      - usuwa dane z `make dane` i cache; surowe zbiory w dane/ zostaja
#
#   make model        # ok. 4 h z treningiem
#   make gui
.PHONY: install test model gui dane demo demo-live clean

# biblioteki projektu + pytest
install:
	pip install -r requirements.txt pytest

# wszystkie testy, z nazwa kazdego testu (-v)
test:
	python -m pytest tests/ -v

# normalizacja -> trening -> ocena -> paczka ONNX (scripts/final_model.sh); caffeinate: Mac nie usypia
model:
	caffeinate -i bash scripts/final_model.sh

# aplikacja w przegladarce
gui:
	streamlit run app.py

# 18 sztucznych zdjec (3 rzedy x 6) z lotu wezykiem -> dane/test/images
dane:
	python scripts/make_test_data.py dane/test --rows 3 --cols 6 --boustrophedon

# mozaika ze wszystkich zdjec naraz
demo: dane
	python -m stitchlive.cli batch dane/test/images -m detail --szwy -o wyniki

# mozaika na zywo, jak w locie: okno 6 zdjec, koniec po 20 s bez nowych zdjec
demo-live: dane
	@echo "Uruchom w drugim terminalu:"
	@echo "  ./scripts/simulate_flight.sh dane/test/images dane/naplyw 2"
	python -m stitchlive.cli watch dane/naplyw --okno 6 --timeout 20 -o wyniki

# tylko dane testowe z `make dane` i cache - surowe zbiory w dane/ zostaja
clean:
	rm -rf dane/test dane/naplyw .pytest_cache
	find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} +
