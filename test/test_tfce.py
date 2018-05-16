from meld.tfce import tfce
import pickle
import numpy as np
from os import path


here = path.abspath(path.dirname(__file__))

def test_tfce():

    with open(path.join(here,"tfce_test_data.pkl"), "rb") as h:
        tfce_tests = pickle.load(h)

    for td in tfce_tests[:-1]:
        res = tfce(td['test'])
        assert res.shape == td['test'].shape
        # Make sure the positive and negative values are in the right place
        assert ((td['test'] > 0) == (res > 0)).all()
        assert ((td['test'] < 0) == (res < 0)).all()
        # Make sure all the values are equal to the TFCE res expected for two elements with e=2/3 and h=2
        assert (np.unique(np.abs(res)) == np.array([0.,  0.52913368], dtype=np.float32)).all()
        # Redundent with above, but since I've got it, might as well
        assert (res == td['res']).all()

    # Last test is a larger random array to do a more thorough smoke test
    td = tfce_tests[-1]
    res = tfce(td['test'])
    assert res.shape == td['test'].shape
    # Make sure the positive and negative values are in the right place
    assert ((td['test'] > 0) == (res > 0)).all()
    assert ((td['test'] < 0) == (res < 0)).all()
    # Redundent with above, but since I've got it, might as well
    assert (res == td['res']).all()
