import numpy as np
from meld.meld import MELD

np.random.RandomState(seed = 42)

# test some MELD
n_jobs = 1
verbose = 10

# generate some fake data
nobs = 100
nsubj = 10
nfeat = (10, 30)
nperms = 100
nboots = 10
fvar_nboots = 10
use_ranks = False
smoothed = False
memmap = False

s = np.concatenate([np.array(['subj%02d' % i] * nobs)
                    for i in range(nsubj)])
beh = np.concatenate([np.array([1] * (nobs // 2) + [0]*(nobs // 2))
                      for i in range(nsubj)])

# observations (data frame)
ind_data = np.rec.fromarrays((np.zeros(len(s)),
                              beh,
                              np.random.rand(len(s)), s),
                             names='val,beh,beh2,subj')

# data with observations in first dimension and features on the remaining
dep_data = np.random.randn(len(s), *nfeat)
print('Data shape:', dep_data.shape)

# make a mask
dep_mask = np.ones(nfeat, dtype=np.bool)
dep_mask[:2, :] = False
dep_mask[:, :2] = False

# now with signal
# add in some signal
dep_data_s = dep_data.copy()
for i in range(0, 20, 2):
    for j in range(2):
        dep_data_s[:, 4, i+j] += (ind_data['beh'] * (i+1)*50.)
        dep_data_s[:, 5, i+j] += (ind_data['beh'] * (i+1)*50.)

# smooth the data
if smoothed:
    import scipy.ndimage
    dep_data = scipy.ndimage.gaussian_filter(dep_data, [0, 1, 1])
    dep_data_s = scipy.ndimage.gaussian_filter(dep_data_s, [0, 1, 1])


def test_meld():

    sig_location = np.array((np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
                                   5, 5, 5, 5, 5, 5, 5, 5, 5, 5]),
                         np.array([2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                                   12, 13, 14, 15, 16, 17, 18, 19,  2,  3,
                                   4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
                                   14, 15, 16, 17, 18, 19])))

    print('Data shape:', dep_data.shape)
    print("Starting MELD perm test")
    print("beh has signal, beh2 does not")
    me_s = MELD('val ~ beh+beh2', '(1|subj)', 'subj',
                dep_data_s, ind_data, factors={'subj': None},
                use_ranks=use_ranks,
                dep_mask=dep_mask,
                feat_nboot=1000, feat_thresh=0.05,
                do_tfce=True,
                connectivity=None, shape=None,
                dt=.01, E=2/3., H=2.0,
                n_jobs=n_jobs, verbose=verbose,
                memmap=memmap,
                # lmer_opts={'control':lme4.lmerControl(optimizer="nloptwrap",
                #                                       #optimizer="Nelder_Mead",
                #                                       optCtrl=r['list'](maxfun=100000))
                #        }
               )
    me_s.run_perms(nperms)
    pfts = me_s.p_features
    print("Number of signifcant features:", [(n, (pfts[n] <= .05).sum())
                                             for n in pfts.dtype.names])

    # This is just the most basic of idiot tests
    assert (np.array(np.where(me_s.t_features['beh'] > 90000)) == sig_location).all()
    assert ((me_s.t_features['beh2'] >90000)==0).all()

    print('Data shape:', dep_data.shape)
    print("Starting MELD bootstrap test")
    print("beh has signal, beh2 does not")
    me_s = MELD('val ~ beh+beh2', '(1|subj)', 'subj',
                dep_data_s, ind_data, factors={'subj': None},
                use_ranks=use_ranks,
                dep_mask=dep_mask,
                feat_nboot=1000, feat_thresh=0.05,
                do_tfce=True,
                connectivity=None, shape=None,
                dt=.01, E=2/3., H=2.0,
                n_jobs=n_jobs, verbose=verbose,
                memmap=memmap,
                # lmer_opts={'control':lme4.lmerControl(optimizer="nloptwrap",
                #                                       #optimizer="Nelder_Mead",
                #                                       optCtrl=r['list'](maxfun=100000))
                #        }
               )
    me_s.run_boots(nperms, fvar_nboots)
    pfts = me_s.p_features
    print("Number of signifcant features:", [(n, (pfts[n] <= .05).sum())
                                             for n in pfts.dtype.names])

    # This is just the most basic of idiot tests
    assert (np.array(np.where(me_s.t_features['beh'] > 6)) == sig_location).all()
    assert ((me_s.t_features['beh2'] >6)==0).all()
