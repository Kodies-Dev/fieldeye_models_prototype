"""mission - detekcja chwastow w locie i mapa oprysku.

    detect_cli.py  - program fieldeye-detect (binarka na Raspberry Pi): nowe zdjecia -> detekcja -> mapa
    spray_map.py   - telemetria GeoJSON obok kazdego zdjecia -> wspolrzedne chwastow -> mapa oprysku
    drone_sim.py   - symulowany dron: zdjecia + telemetria w tym samym formacie co prawdziwy dron
    detector.py    - detektory (.pt / .onnx / .hef, kafelki, TTA) - wspolne z app.py i models/
    sim_flight.py  - pomocnicze funkcje symulowanego lotu w zakladce 4 app.py
"""
