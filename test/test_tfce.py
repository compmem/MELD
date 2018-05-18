from meld.tfce import tfce
import pickle
import numpy as np
from os import path


here = path.abspath(path.dirname(__file__))

def test_tfce():
    with open(path.join(here,"tfce_test_data.pkl"), "rb") as h:
        tfce_tests = pickle.load(h)
    for pad in [True, False]:
        for td in tfce_tests[:-3]:
            res = tfce(td['test'], pad=True)
            assert res.shape == td['test'].shape
            # Make sure the positive and negative values are in the right place
            assert (res[((res < 0) != (td['test'] < 0))]==0).all()
            assert (res[((res > 0) != (td['test'] > 0))]==0).all()
            # Make sure all the values are equal to the TFCE res expected for two elements with e=2/3 and h=2
            assert (np.unique(np.abs(res)) == np.array([0.,  0.52913368], dtype=np.float32)).all()
            # Redundent with above, but since I've got it, might as well
            assert (res == td['res']).all()


        # third to last test is a larger random array to do a more thorough smoke test
        td = tfce_tests[-3]
        res = tfce(td['test'], pad=pad)
        assert res.shape == td['test'].shape
        # Make sure the positive and negative values are in the right place
        assert (res[((res < 0) != (td['test'] < 0))]==0).all()
        assert (res[((res > 0) != (td['test'] > 0))]==0).all()
        # Redundent with above, but since I've got it, might as well
        assert (res == td['res']).all()
        
        # second to last test is a 100x100x100 array
        td = tfce_tests[-2]
        res = tfce(td['test'], pad=pad)
        assert res.shape == td['test'].shape
        # Make sure the positive and negative values are in the right place
        assert (res[((res < 0) != (td['test'] < 0))]==0).all()
        assert (res[((res > 0) != (td['test'] > 0))]==0).all()
        # Redundent with above, but since I've got it, might as well
        assert (res == td['res']).all()
        
        # last test is comparison to hcp results
        td = tfce_tests[-1]
        res = tfce(td['test'], pad=pad)
        assert res.shape == td['test'].shape
        assert np.allclose(res, td['res'], atol=1e-01)