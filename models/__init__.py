"""models - trening, ocena i eksport modelu detekcji chwastow.

Po co: wszystko, co dotyczy modelu YOLO - od kontroli danych po plik ONNX dla Hailo.

Pliki:
    cli.py        - wiersz polecen: python -m models.cli <komenda>
                    (nadprobkuj, trenuj, ocen-detektor, onnx, kalibracja, sprawdz, srodowisko)
    plan.py       - plan treningu (Stage, TrainingPlan) i kontrola danych (sprawdz_dane)
    training.py   - sam trening przez ultralytics
    device.py     - wybor urzadzenia: CUDA -> MPS (Mac) -> CPU
    season.py     - nadprobkowanie wczesnych dni lotu
    evaluation.py - ocena: AP50, czulosc, precyzja, wynik wedlug dni (tez dla .onnx i .hef)
    export.py     - eksport do ONNX i zdjecia kalibracyjne dla kompilatora Hailo

Na zewnatrz wystawia: Stage, TrainingPlan, sprawdz_dane (from models import TrainingPlan).
"""

from .plan import Stage, TrainingPlan, sprawdz_dane

__all__ = ["Stage", "TrainingPlan", "sprawdz_dane"]
