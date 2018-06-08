import numpy as np
from meld import stat_helper as msh

def test_boot_stat():
    stat = np.array([[[1.], [10.]], [[2.], [20.]], [[5.], [50.]]])
    sterr = np.array([[[0.1],[1]], [[0.4], [4]], [[2],[20]]])

    expected = np.array(      [[[ 10. ],
                                [ 10. ]],
                        
                               [[  2.5],
                                [  2.5]],

                               [[  2. ],
                                [  2. ]]], dtype=float)
    observed = msh.boot_stat(stat, sterr)
    assert (observed == expected).all()
    assert observed.dtype == expected.dtype