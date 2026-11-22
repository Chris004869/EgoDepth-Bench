"""A legal representation zero must survive the adapter, not only the fitter."""
import numpy as np
from egodepth_eval.infer.legacy_adapter import convert_legacy_output, E1_METHODS
from egodepth_eval.score.alignment import fit_alignment

def test_zero_disparity_is_scored_after_affine_offset(tmp_path):
    raw=np.linspace(0,1,64,dtype=np.float32).reshape(8,8)
    path=tmp_path/'raw.npz';np.savez(path,depth=raw)
    p=convert_legacy_output(path,E1_METHODS['depth_anything_v2'],None,(8,8))
    assert p['valid'].all()
    gt=1/(raw+1)
    fit=fit_alignment(p['raw'],gt,np.ones((8,8),bool),'affine_inverse',p['valid'])
    assert fit.n_fit_pixels==64
    np.testing.assert_allclose(fit.apply(p['raw']),gt,rtol=1e-6)

def test_affine_depth_negative_and_zero_survive_resize(tmp_path):
    raw=np.linspace(-1,1,64,dtype=np.float32).reshape(8,8);raw[1,1]=0
    path=tmp_path/'raw.npz';np.savez(path,depth=raw)
    p=convert_legacy_output(path,E1_METHODS['marigold'],None,(16,16))
    assert p['valid'].all() and np.isfinite(p['depth_z']).all()
    assert p['depth_z'].min()<0 and p['depth_z'].max()>0

def test_metric_depth_still_rejects_zero_and_negative(tmp_path):
    raw=np.ones((8,8),np.float32);raw[0,0]=0;raw[0,1]=-1
    path=tmp_path/'raw.npz';np.savez(path,depth=raw)
    p=convert_legacy_output(path,E1_METHODS['unidepth_v2'],None,(8,8))
    assert not p['valid'][0,:2].any() and p['valid'].sum()==62
