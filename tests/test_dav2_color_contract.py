"""Depth Anything v2's public image API consumes OpenCV BGR."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np


def test_runner_passes_bgr_once_to_official_api(tmp_path, monkeypatch):
    root = Path(os.environ.get('EGODEPTH_RUNNER_TEST_ROOT', Path(__file__).resolve().parents[1]))
    path = root / 'method_runners/run_depth_anything_v2.py'
    spec = importlib.util.spec_from_file_location('dav2_color_contract_runner', path)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    bgr = np.zeros((5, 7, 3), dtype=np.uint8)
    bgr[:] = (13, 71, 229)
    image_path = tmp_path / 'asymmetric_color.png'
    assert cv2.imwrite(str(image_path), bgr)
    received = []
    class Model:
        def infer_image(self, image):
            received.append(image.copy())
            return np.ones(image.shape[:2], np.float32)
    class Parser:
        def parse_args(self):
            return SimpleNamespace(device='cpu')
    monkeypatch.setattr(runner, 'load_model', lambda device: Model())
    monkeypatch.setattr(runner, 'build_parser', lambda *a: Parser())
    monkeypatch.setattr(runner, 'rgb_path', lambda *a: str(image_path))
    monkeypatch.setattr(runner, 'run_loop', lambda name, args, predict, **kw: predict({'frame_id':'x'}, tmp_path))
    runner.main()
    np.testing.assert_array_equal(received[0], bgr)
    # The public API itself converts this BGR array into the original RGB color.
    actual_rgb = cv2.cvtColor(received[0], cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(actual_rgb[0, 0], [229, 71, 13])
