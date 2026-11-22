from dataclasses import replace

import numpy as np
import pytest

from egodepth_eval.data.contract import DSCalibration
from egodepth_eval.infer.mapanything_backend import rig_camera_poses


def calibration(camera, pose):
    return DSCalibration(camera, 100, 100, 256, 208, 0, .6, 512, 416,
                         T_body_cam=pose)


def test_pose_order_and_camera_to_body_direction_preserve_metric_baseline():
    # A non-identity world frame exposes a mistaken per-camera inverse.
    body = np.array([[0, -1, 0, .4], [1, 0, 0, -.3],
                     [0, 0, 1, .2], [0, 0, 0, 1.]])
    relative = np.eye(4)
    relative[0, 3] = .06
    cams = {2: calibration(2, body), 3: calibration(3, body @ relative)}
    poses = rig_camera_poses(cams, [3, 2])
    # Reversing input order must reverse the relative baseline, not the poses.
    np.testing.assert_allclose(np.linalg.inv(poses[0]) @ poses[1],
                               np.linalg.inv(relative), atol=1e-7)


def test_missing_or_mislabeled_pose_never_becomes_identity():
    c = calibration(2, None)
    with pytest.raises(ValueError, match='missing'):
        rig_camera_poses({2: c}, [2])
    with pytest.raises(ValueError, match='mismatched'):
        rig_camera_poses({2: replace(c, camera_id=3, T_body_cam=np.eye(4))}, [2])


def test_nonrigid_pose_is_rejected():
    pose = np.eye(4)
    pose[0, 0] = 1000  # e.g. a unit conversion wrongly applied to the whole matrix
    with pytest.raises(ValueError, match=r'SE\(3\)'):
        rig_camera_poses({2: calibration(2, pose)}, [2])
