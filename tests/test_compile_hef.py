"""Skrypt kompilacji HEF: konfiguracja NMS i skrypt modelu (bez kompilatora Hailo)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compile_hef import KONCOWE, konfiguracja_nms, skrypt_modelu, zdjecia_kalibracyjne  # noqa: E402

ALLS = '''normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
change_output_activation(conv42, sigmoid)
change_output_activation(conv53, sigmoid)
change_output_activation(conv63, sigmoid)
nms_postprocess("../../postprocess_config/yolov8n_nms_config.json", meta_arch=yolov8, engine=cpu)
'''
NMS = {"nms_scores_th": 0.2, "nms_iou_th": 0.7, "image_dims": [640, 640], "max_proposals_per_class": 100,
       "classes": 80, "regression_length": 16, "background_removal": False, "bbox_decoders": []}


def test_konfiguracja_nms_1280(tmp_path):
    (tmp_path / "n.json").write_text(json.dumps(NMS))
    c = konfiguracja_nms(tmp_path / "n.json", 1280)
    assert c["image_dims"] == [1280, 1280] and c["classes"] == 1
    assert c["nms_scores_th"] == 0.05 and c["max_proposals_per_class"] == 300
    assert c["regression_length"] == 16                     # reszta bez zmian


def test_skrypt_modelu_podmienia_tylko_sciezke_nms(tmp_path):
    (tmp_path / "y.alls").write_text(ALLS)
    s = skrypt_modelu(tmp_path / "y.alls", "/home/x/fieldeye_nms.json")
    assert 'nms_postprocess("/home/x/fieldeye_nms.json", meta_arch=yolov8, engine=cpu)' in s
    assert "change_output_activation(conv63, sigmoid)" in s and "normalization1" in s
    (tmp_path / "z.alls").write_text("normalization1 = normalization([0], [255])\n")
    with pytest.raises(ValueError):
        skrypt_modelu(tmp_path / "z.alls", "x.json")


def test_zdjecia_kalibracyjne_rgb_i_rozmiar(tmp_path):
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, :, 0] = 200                                        # niebieski w BGR
    cv2.imwrite(str(tmp_path / "a.png"), img)
    d = zdjecia_kalibracyjne(tmp_path, 64, 10)
    assert d.shape == (1, 64, 64, 3) and d.dtype == np.float32
    assert d[0, 0, 0, 2] == 200 and d[0, 0, 0, 0] == 0         # po zamianie na RGB niebieski jest ostatni
    with pytest.raises(ValueError):
        zdjecia_kalibracyjne(tmp_path, 128, 10)               # zly rozmiar - wyraźny błąd


def test_koncowe_warstwy():
    assert len(KONCOWE) == 6 and all(k.startswith("/model.22/cv") for k in KONCOWE)