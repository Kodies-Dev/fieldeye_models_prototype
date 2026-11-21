"""Punkt wejscia binarki fieldeye-detect (PyInstaller).

  fieldeye-detect ZDJECIA WYNIKI MODEL [opcje]     detekcja (mission/detect_cli.py)
  fieldeye-detect dron FOLDER [opcje]              symulowany dron z telemetria GPS (mission/drone_sim.py)
  fieldeye-detect porownaj OPRYSK PRAWDA           ocena namiarow oprysku (mission/spray_map.py)
"""
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("dron", "porownaj"):
        polecenie = sys.argv.pop(1)
        if polecenie == "dron":
            from mission.drone_sim import main as uruchom
        else:
            from mission.spray_map import main as uruchom
        return uruchom(sys.argv[1:])
    from mission.detect_cli import main as uruchom
    return uruchom()


if __name__ == "__main__":
    raise SystemExit(main())
