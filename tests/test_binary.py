"""Binarka fieldeye-detect: lekki detektor ONNX, model obok binarki, importy na zadanie."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _model_onnx(sciezka, ramki, wejscie=64):
    """Maly model (1,3,W,W) -> (1,5,N) zwracajacy stale ramki [cx, cy, w, h, pewnosc]."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    stala = np.array(ramki, np.float32).T[None]
    g = helper.make_graph(
        [helper.make_node("ReduceMean", ["images"], ["m"], keepdims=0),
         helper.make_node("Mul", ["m", "zero"], ["z"]),
         helper.make_node("Add", ["stala", "z"], ["output0"])],
        "proba", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, wejscie, wejscie])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(stala.shape))],
        [numpy_helper.from_array(stala, "stala"), numpy_helper.from_array(np.array(0, np.float32), "zero")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 7
    onnx.save(m, str(sciezka))


def test_onnx_detektor_prog_nms_i_przeliczenie(tmp_path):
    pytest.importorskip("onnxruntime")
    from mission.detector import OnnxDetector, zbuduj_z_plikow
    _model_onnx(tmp_path / "m.onnx", [[20, 30, 10, 10, 0.9], [21, 30, 10, 10, 0.6],
                                      [50, 50, 6, 6, 0.05], [45, 20, 8, 4, 0.7]])
    d = zbuduj_z_plikow(str(tmp_path / "m.onnx"), prog=0.15)
    assert isinstance(d, OnnxDetector)
    wyn = d.wykryj(np.zeros((96, 128, 3), np.uint8))       # skala 0,5, pas 8 px u gory i dolu
    ramki = sorted((round(r.x0), round(r.y0), round(r.x1), round(r.y1), round(r.pewnosc, 2)) for r in wyn)
    assert ramki == [(30, 34, 50, 54, 0.9), (82, 20, 98, 28, 0.7)]    # slaba odpada, NMS zostawia 0,9


def test_model_obok_binarki(tmp_path, monkeypatch):
    from mission import detect_cli as D
    exe = tmp_path / "fieldeye-detect"
    exe.write_text("")
    (tmp_path / "fieldeye_final.onnx").write_text("x")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert D.domyslny_model() == tmp_path / "fieldeye_final.onnx"
    (tmp_path / "fieldeye_final_1280_h8.hef").write_text("x")              # .hef ma pierwszenstwo
    assert D.domyslny_model() == tmp_path / "fieldeye_final_1280_h8.hef"


def test_detect_cli_nie_wczytuje_reszty_misji():
    kod = ("import sys; import mission.detect_cli; "
           "print(','.join(m for m in ('pyproj', 'mission.geo', 'mission.post_flight', 'torch') if m in sys.modules))")
    r = subprocess.run([sys.executable, "-c", kod], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "", r.stdout + r.stderr


def test_mission_dalej_daje_dostep_do_podmodulow():
    import mission
    assert mission.geo.__name__ == "mission.geo" and mission.MissionConfig.__name__ == "MissionConfig"


def test_skrypt_budujacy_bez_bledow_skladni():
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / "build_binary.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_polecenia_dron_i_porownaj_w_punkcie_wejscia(tmp_path):
    """fieldeye-detect dron ... / porownaj ... - te same polecenia, ktore ma binarka."""
    main = str(ROOT / "scripts" / "fieldeye_detect_main.py")
    env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}
    lot = tmp_path / "lot"
    r = subprocess.run([sys.executable, main, "dron", str(lot), "--sztuczne", "--interwal", "0"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert any(lot.glob("*.geojson")) and Path(str(lot) + "_prawda.geojson").exists()
    r = subprocess.run([sys.executable, main, str(lot), str(tmp_path / "w"), "--raz", "--atrapa"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "Punkty oprysku" in r.stdout, r.stderr
    r = subprocess.run([sys.executable, main, "porownaj", str(tmp_path / "w" / "oprysk.geojson"),
                        str(lot) + "_prawda.geojson"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "trafione" in r.stdout, r.stderr


def test_skrypty_bez_bledow_skladni():
    for s in ("przygotuj_folder.sh", "spakuj_na_pi.sh", "na_pi.sh"):
        r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / s)], capture_output=True, text=True)
        assert r.returncode == 0, s + ": " + r.stderr
