import copy
import numpy as np
import pytest
torch = pytest.importorskip("torch", reason="install the train extra for training contract tests")

from egodepth_eval.train.contract import adaptation_inputs, verify_calibration_inputs
from egodepth_eval.train.dataset import collate_views
from egodepth_eval.train.trainer import views_to_device, set_deterministic_geometric_inputs
from egodepth_eval.train.trainer import checkpoint_dense_heads


def config():
    method = dict(checkpoint='same', view_ids=list(range(6)), pose_mode='rig')
    return {'methods': {n: copy.deepcopy(method) for n in
            ('mapanything_6view', 'mapanything_scale_only', 'mapanything_ft')}, 'training': {}}


def test_adaptation_rejects_scale_or_ft_without_matching_geometry():
    c = config(); assert adaptation_inputs(c)['pose_mode'] == 'rig'
    c['methods']['mapanything_ft']['pose_mode'] = 'none'
    with pytest.raises(ValueError, match='inputs'): adaptation_inputs(c)
    c = config(); c['training']['pose_mode'] = 'none'
    with pytest.raises(ValueError, match='pose_mode'): adaptation_inputs(c)


def test_rig_rejects_old_scale_record_and_changed_view_order():
    expected = adaptation_inputs(config())
    with pytest.raises(ValueError, match='explicit'): verify_calibration_inputs({}, expected)
    changed = {**expected, 'view_ids': [2, 0, 1, 3, 4, 5]}
    with pytest.raises(ValueError, match='differ'): verify_calibration_inputs({'input_contract': changed}, expected)
    verify_calibration_inputs({'input_contract': expected}, expected)


def test_pose_presence_must_agree_with_model_flags():
    class Model: geometric_input_config = {}
    model = Model(); set_deterministic_geometric_inputs(model, 'rig')
    assert model.geometric_input_config['cam_prob'] == 1
    assert model.geometric_input_config['depth_prob'] == 0
    with pytest.raises(ValueError, match='requires every'):
        views_to_device([{}], 'cpu', pose_mode='rig')
    with pytest.raises(ValueError, match='Unexpected'):
        views_to_device([{'camera_poses': torch.eye(4)}], 'cpu', pose_mode='none')


def test_collate_keeps_distinct_rig_poses_for_every_batch_item():
    def item(x):
        pose = np.eye(4, dtype=np.float32); pose[0, 3] = x
        return {'views': [{'img': np.zeros((3, 2, 2), np.float32),
                 'ray_directions': np.zeros((2, 2, 3), np.float32),
                 'camera_id': 2, 'data_norm_type': 'dinov2', 'camera_poses': pose}],
                'target_depth_z': np.ones((2, 2), np.float32), 'target_valid': np.ones((2, 2), bool),
                'target_view_position': 0, 'target_camera_id': 2, 'sample_token': str(x),
                'site_id': str(x), 'recording_id': str(x)}
    batch = collate_views([item(.06), item(.08)])
    np.testing.assert_allclose(batch['views'][0]['camera_poses'][:,0,3], [.06,.08])
    bad = item(.08); del bad['views'][0]['camera_poses']
    with pytest.raises(ValueError, match='Mixed'): collate_views([item(.06), bad])


def test_dense_checkpointing_preserves_state_keys_outputs_and_gradients():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dpt_feature_head=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.GELU())
            self.dpt_regressor_head=torch.nn.Linear(8,1)
        def forward(self,x):return self.dpt_regressor_head(self.dpt_feature_head(x))
    first=Model();second=copy.deepcopy(first);keys=list(second.state_dict())
    assert len(checkpoint_dense_heads(second))==2 and checkpoint_dense_heads(second)==[]
    assert list(second.state_dict())==keys
    x=torch.randn(3,4,requires_grad=True); y=first(x);z=second(x)
    torch.testing.assert_close(y,z,rtol=0,atol=0)
    y.sum().backward();z.sum().backward()
    for a,b in zip(first.parameters(),second.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
