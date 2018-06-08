#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from __future__ import print_function
from builtins import range

import os
import sys
import time
import tempfile
import numpy as np
import pandas as pd
from numpy.lib.recfunctions import append_fields
from scipy.linalg import diagsvd
from scipy.stats import rankdata, gaussian_kde
import scipy.stats.distributions as dists

from joblib import Parallel, delayed

# Connect to an R session
import rpy2.robjects
r = rpy2.robjects.r

# For a Pythonic interface to R
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula, FactorVector, pandas2ri
from rpy2.robjects.environments import Environment
from rpy2.robjects.vectors import DataFrame, Vector, FloatVector, StrVector
from rpy2.rinterface import MissingArg, SexpVector, RRuntimeError

# Make it so we can send numpy arrays to R
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
#import rpy2.robjects as ro
#import  rpy2.robjects.numpy2ri as numpy2ri
#ro.conversion.py2ri = numpy2ri
#numpy2ri.activate()

def get_rpackage(packname):
    try:
        result = importr(packname)
    except RRuntimeError:
        utils = importr('utils')
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(packname)
        result = importr(packname)
    return result

# load some required packages
# PBS: Eventually we should try/except these to get people
# to install missing packages
lme4 = get_rpackage('lme4')
rstats = get_rpackage('stats')
if hasattr(lme4,'coef'):
    r_coef = lme4.coef
else:
    r_coef = rstats.coef
if hasattr(lme4,'model_matrix'):
    r_model_matrix = lme4.model_matrix
else:
    r_model_matrix = rstats.model_matrix

# load ptsa clustering
from . import cluster
from . import stat_helper

# load cython tfce
from . import tfce

fdr_correction = stat_helper.fdr_correction

# deal with warnings for bootstrap
import warnings

class InstabilityWarning(UserWarning):
    """Issued when results may be unstable."""
    pass

# On import, make sure that InstabilityWarnings are not filtered out.
warnings.simplefilter('always', InstabilityWarning)

def get_re(model):
    """Given an lmer model, return the random effect stats.

    Parameters
    ----------
    model : R object with classes: ('lmerMod',)
        The fitted model object that you're extracting RE stats from.

    Returns
    -------
    ran_vars : DataFrame
        Dataframe of the random effects
    ran_corrs : DataFrame or None
        Dataframe of the correlations between random effects terms.
    """
    #pymer4
    base = importr('base')
    summary = base.summary(model)
    unsum = base.unclass(summary)

    df = pandas2ri.ri2py(base.data_frame(unsum.rx2('varcor')))
    ran_vars = df.query("(var2 == 'NA') | (var2 == 'N')").drop('var2', axis=1)
    ran_vars.index = ran_vars['grp']
    ran_vars.drop('grp', axis=1, inplace=True)
    ran_vars.columns = ['Name', 'Var', 'Std']
    ran_vars.index.name = None
    ran_vars.replace('NA', '', inplace=True)

    ran_corrs = df.query("(var2 != 'NA') & (var2 != 'N')").drop('vcov', axis=1)
    if ran_corrs.shape[0] != 0:
        ran_corrs.index = ran_corrs['grp']
        ran_corrs.drop('grp', axis=1, inplace=True)
        ran_corrs.columns = ['IV1', 'IV2', 'Corr']
        ran_corrs.index.name = None
    else:
        ran_corrs = None
    #\pymer4
    return ran_vars, ran_corrs

class LMER():
    """
    Wrapper for the lmer method provided by lme4 in R.

    This object facilitates fitting the same model multiple times
    and extracting the associated t-stat.

    Parameters
    ----------
    formula_str : str
        lme4 style mixed effects model specification
    df : DataFrame
        Dataframe with column labels corresponding to formula
    factors : list of str
        Optional, list of the names of the factors
    resid_formula_str : str
        lme4 style mixed effects model specification for residuals
    **lmer_opts
        additional keyword arguments for lme4
    """
    def __init__(self, formula_str, df, factors=None,
                 resid_formula_str=None, **lmer_opts):
        """
        """
        # get the pred_var
        pred_var = formula_str.split('~')[0].strip()

        # convert df to a recarray if it's a dataframe
        if isinstance(df, pd.DataFrame):
            df = df.to_records()

        # add column if necessary
        if pred_var not in df.dtype.names:
            # must add it
            df = append_fields(df, pred_var, [0.0]*len(df), usemask=False)

        # make factor list if necessary
        if factors is None:
            factors = {}
        # add in missingarg for any potential factor not provided
        for k in df.dtype.names:
            if isinstance(df[k][0], str) and k not in factors:
                factors[k] = MissingArg

        for f in factors:
            if factors[f] is None:
                factors[f] = MissingArg
            # checking for both types of R Vectors for rpy2 variations
            elif (not isinstance(factors[f], Vector) and
                  not factors[f] == MissingArg):
                factors[f] = Vector(factors[f])

        # convert the recarray to a DataFrame (releveling if desired)
        self._rdf = DataFrame({k: (FactorVector(df[k], levels=factors[k])
                                    if (k in factors) or isinstance(df[k][0], str)
                                   else df[k])
                               for k in df.dtype.names})

        # get the column index
        self._col_ind = list(self._rdf.colnames).index(pred_var)

        # make a formula obj
        self._rformula = Formula(formula_str)

        # make one for resid if necessary
        if resid_formula_str:
            self._rformula_resid = Formula(resid_formula_str)
        else:
            self._rformula_resid = None

        # save the args
        self._lmer_opts = lmer_opts

        # model is null to start
        self._ms = None

    def _get_re(self):
        """Get the random effects values and correlations for the LMER._ms model
        
        Returns
        -------
        ran_vars : DataFrame
            Dataframe of the random effects
        ran_corrs : DataFrame or None
            Dataframe of the correlations between random effects terms.
        """
        if self._ms is None:
            raise RuntimeError("Model hasn't been fit yet. Execute LMER.run() first")
        else:
            return get_re(self._ms)

    def run(self, vals=None, perms=None):
        """Fit the mixed effects models, potentially on different data or with permutations.

        """
        # set the col with the val
        if vals is not None:
            self._rdf[self._col_ind] = vals

        # just apply to actual data if no perms
        if perms is None:
            perms = [None]
            #perms = [np.arange(len(self._rdf[self._col_ind]))]

        # run on each permutation
        resds = None
        betas = None
        tvals = None
        log_likes = None
        for i, perm in enumerate(perms):
            if perm is not None:
                # set the perm
                self._rdf[self._col_ind] = self._rdf[self._col_ind].rx(perm+1)

            # inside try block to catch convergence errors
            try:
                if self._rformula_resid:
                    # get resid first
                    msr = lme4.lmer(self._rformula_resid, data=self._rdf,
                                    **self._lmer_opts)
                    self._rdf[self._col_ind] = lme4.resid(msr)
                # run the model (possibly on the residuals from above)
                ms = lme4.lmer(self._rformula, data=self._rdf,
                               **self._lmer_opts)
            except:
                continue
                #tvals.append(np.array([np.nan]))

            # save the model
            if self._ms is None:
                self._ms = ms
                if self._rformula_resid:
                    self._msr = msr

            # extract the result
            df = r['data.frame'](r_coef(r['summary'](ms)))
            if tvals is None:
                # init the data
                # get the row names
                rows = list(r['row.names'](df))
                betas = np.rec.fromarrays([np.ones(len(perms))*np.nan
                                           for ro in range(len(rows))],
                                          names=','.join(rows))
                tvals = np.rec.fromarrays([np.ones(len(perms))*np.nan
                                           for ro in range(len(rows))],
                                          names=','.join(rows))
                log_likes = np.zeros(len(perms))

            # set the values
            betas[i] = tuple(df.rx2('Estimate'))
            tvals[i] = tuple(df.rx2('t.value'))
            log_likes[i] = float(r['logLik'](ms)[0])
        resds = r['resid'](ms)
        #resds = np.zeros(len(self._rdf[self._col_ind]))
        return resds, betas, tvals, log_likes



class OnlineVariance(object):
    """
    Welford's algorithm computes the sample variance incrementally.

    http://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream
    """

    def __init__(self, iterable=None, ddof=1):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum):
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)
        self.variance = self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)


def R_to_tfce(R, mask, connectivity=None, shape=None,
              dt=.01, E=2/3., H=2.0):
    """Apply TFCE to the R values."""
    # allocate for tfce
    Rt = np.zeros_like(R)
    # Z = np.arctanh(R)
    Z = R
    # loop
    for i in range(Rt.shape[0]):
        for j in range(Rt.shape[1]):
            if connectivity is None:
                # If there's no connectivity defined
                # Use the fast neighbors tfce
                Zin = np.zeros(mask.shape)
                Zin[mask] = Z[i, j]
                # reshape back
                Zin = Zin.reshape(*shape)
                Rt[i, j] += tfce.tfce(Zin, volume=1,
                                      param_e=E, param_h=H,
                                      pad=True)[mask]
            else:
                # apply tfce in pos and neg direction
                Zin = Z[i, j]
                Rt[i, j] += cluster.tfce(Zin,
                                         dt=dt, tail=1,
                                         connectivity=connectivity,
                                         E=E, H=H).flatten()
                Rt[i, j] += cluster.tfce(Zin,
                                         dt=dt, tail=-1,
                                         connectivity=connectivity,
                                         E=E, H=H).flatten()
    return Rt


def pick_stable_features(Z, nboot=500):
    """Use a bootstrap to pick stable features.
    """
    # generate the boots
    boots = [np.random.random_integers(0, len(Z)-1, len(Z))
             for i in range(nboot)]

    # calc bootstrap ratio
    # calc the bootstrap std in efficient way
    # old way
    # Zb = np.array([Z[boots[b]].mean(0) for b in range(len(boots))])
    # Zbr = Z.mean(0)/Zb.std(0)
    ov = OnlineVariance(ddof=0)
    for b in range(len(boots)):
        ov.include(Z[boots[b]].mean(0))
    Zbr = Z.mean(0)/ov.std

    # ignore any nans
    Zbr[np.isnan(Zbr)] = 0.

    # bootstrap ratios are supposedly t-distributed, so test sig
    Zbr = dists.t(len(Z)-1).cdf(-1*np.abs(Zbr))*2.
    Zbr[Zbr > 1] = 1
    return Zbr


# modified from:
# http://stackoverflow.com/questions/20983882/efficient-dot-products-of-large-memory-mapped-arrays
def _block_slices(dim_size, block_size):
    """Generator that yields slice objects for indexing into
    sequential blocks of an array along a particular axis
    """
    count = 0
    while True:
        yield slice(count, count + block_size, 1)
        count += block_size
        if count > dim_size:
            raise StopIteration


def blockwise_dot(A, B, max_elements=int(2**26), out=None):
    """
    Computes the dot product of two matrices in a block-wise fashion.
    Only blocks of `A` with a maximum size of `max_elements` will be
    processed simultaneously.
    """

    if len(A.shape) == 1:
        A = A[np.newaxis, :]
    if len(B.shape) == 1:
        B = B[:, np.newaxis]
    m,  n = A.shape
    n1, o = B.shape

    if n1 != n:
        raise ValueError('matrices are not aligned')

    if A.flags.f_contiguous:
        # prioritize processing as many columns of A as possible
        max_cols = max(1, max_elements // m)
        max_rows = max_elements // max_cols

    else:
        # prioritize processing as many rows of A as possible
        max_rows = max(1, max_elements // n)
        max_cols = max_elements // max_rows

    if out is None:
        out = np.empty((m, o), dtype=np.result_type(A, B))
    elif out.shape != (m, o):
        raise ValueError('output array has incorrect dimensions')

    for mm in _block_slices(m, max_rows):
        out[mm, :] = 0
        for nn in _block_slices(n, max_cols):
            A_block = A[mm, nn].copy()  # copy to force a read
            out[mm, :] += np.dot(A_block, B[nn, :])
            del A_block
            # out[mm, :] += np.dot(A[mm, nn], B[nn, :])

    return out


# global container so that we can use joblib with a smaller memory
# footprint (i.e., we don't need to duplicate everything)
_global_meld = {}


def _eval_model(model_id, perm=None, boot=None):
    # set vars from model
    mm = _global_meld[model_id]
    ct = mm._component_thresh
    _R = mm._R

    # Calculate R
    R = []
    if boot is None:
        ind_b = np.arange(len(mm._groups))
    else:
        ind_b = boot


    # loop over group vars
    ind = {}
    for i,k in enumerate(mm._groups[ind_b]):
        # grab the A and M
        A = mm._A[k]
        M = mm._M[k]

        # gen a perm for that subj
        if perm is None:
            ind[k] = np.arange(len(A))
        else:
            ind[k] = perm[k]

        if perm is None and mm._R is not None:
            # reuse R we already calculated
            R.append(mm._R[ind_b[i]])
        else:
            # calc the correlation
            #R.append(np.dot(A.T,M[ind[k]].copy()))
            #R.append(blockwise_dot(M[:][ind[k]].T, A).T)
            R.append(blockwise_dot(M[ind[k]].T, A).T)

    # turn R into array
    R_nocat = np.array(R)

    # zero invariant features
    feat_mask = np.isnan(R)
    R_nocat[feat_mask] = 0.0

    # turn to Z
    # make sure we are not 1/-1
    epsilon = 1e-15
    R_nocat[R_nocat > 1 - epsilon] = 1 - epsilon
    R_nocat[R_nocat < -1 + epsilon] = -1 + epsilon
    if mm._z_transform:
        R_nocat = np.arctanh(R_nocat)

    if mm._do_tfce:
        # run TFCE
        Ztemp = R_to_tfce(R_nocat, mm._dep_mask, connectivity=mm._connectivity,
                          shape=mm._feat_shape,
                          dt=mm._dt, E=mm._E, H=mm._H)
    else:
        # just use the current Z without TFCE
        Ztemp = R_nocat

    # pick only stable features
    # NOTE: Ztemp is no longer R, it's either TFCE or Z
    Rtbr = pick_stable_features(Ztemp, nboot=mm._feat_nboot)

    # actually use the TFCE for SVD
    if mm._tfce_svd:
        R_nocat = Ztemp

    # apply the thresh
    stable_ind = Rtbr < mm._feat_thresh
    stable_ind = stable_ind.reshape((stable_ind.shape[0], -1))
    # zero out non-stable
    feat_mask[:, ~stable_ind] = True
    R_nocat[:, ~stable_ind] = 0.0

    # save R before concatenating if not a permutation
    if perm is None and boot is None and _R is None:
        _R = R_nocat.copy()

    # concatenate R for SVD
    # NOTE: It's really Z or TFCE now
    R = np.concatenate([R_nocat[i] for i in range(len(R_nocat))])

    # perform svd
    U, s, Vh = np.linalg.svd(R, full_matrices=False)
    #if boot is not None:
    #    import pdb; pdb.set_trace()
    # fix near zero vals from SVD
    # epsilon = 0
    # Vh[np.abs(Vh) < (epsilon * mm._dt)] = 0.0
    # s[np.abs(s) < epsilon] = 0.0

    # filtering by percent variance explained 
    # drops the number of components too low, especially on bootstraps
    #s[pve < 0.0000001] = 0.0
    # calc prop of variance accounted for
    if mm._ss is None:
        # _ss = np.sqrt((s*s).sum())
        _ss = s.sum()
    else:
        _ss = mm._ss
    ss = s

    # If it's a permutation, we'll normalize by the orig ss
    # otherwise, normalize by sum of ss
    if perm is not None:
        ss_ratio = ss/_ss
    else:
        ss_ratio = ss/ss.sum()
    
    # calc percent variance explained
    if (s**2).sum() == 0:
        pve = np.zeros_like(s)
        # must make ss and dependent terms, too
        ss = np.array([0.0])
        ss_norm = np.array([0.0])
        ss_filt = np.array([0.0])
        ss_diag = np.zeros((1,1))
        ssVh = np.zeros((1,*Vh.shape[1:]))
        beta_project = np.zeros((1,*Vh.shape[1:]))
    else:
        pve = s**2/(s**2).sum()
        ss_norm = ss_ratio[pve > ct]
        ss_filt = ss[pve>ct]
        ss_diag = diagsvd(ss_filt, len(ss_filt), len(ss_filt))
        ssVh = blockwise_dot(ss_diag, Vh[pve > ct, ...])
        beta_project = Vh[pve > ct, ...]
    # set up lmer
    O = None
    lmer = None
    if mm._mer is None or boot is not None:
        O = []
        for i, ib in enumerate(ind_b):
            tempO = mm._O[ib].copy()
            tempO[mm._re_group] = mm._groups[i]
            O.append(tempO)
        lmer = LMER(mm._formula_str, np.concatenate(O),
                    factors=mm._factors,
                    resid_formula_str=mm._resid_formula_str, **mm._lmer_opts)
        mer = None
    else:
        mer = mm._mer

    # loop over LVs performing LMER
    res = []
    for i in range(len(Vh)):
        if pve[i] <= ct:
            # print 'skipped ',str(i)
            continue

        # flatten then weigh features via dot product
        Dw = np.concatenate([#np.dot(mm._D[k][ind[k]].copy(),Vh[i])
                             #blockwise_dot(mm._D[k][:][ind[k]], Vh[i])
                             blockwise_dot(mm._D[k][ind[k]], Vh[i])
                             for g, k in enumerate(mm._groups[ind_b])])

        # run the main model
        if mer is None:
            # run the model for the first time and save it
            res.append(lmer.run(vals=Dw))
            mer = lmer._ms
        else:
            # use the saved model and just refit it for speed
            mer = r['refit'](mer, FloatVector(Dw))
            df = r['data.frame'](r_coef(r['summary'](mer)))
            rows = list(r['row.names'](df))
            new_betas = np.rec.fromarrays([[tv]
                                           for tv in tuple(df.rx2('Estimate'))],
                                          names=','.join(rows))
            new_tvals = np.rec.fromarrays([[tv]
                                           for tv in tuple(df.rx2('t.value'))],
                                          names=','.join(rows))

            new_ll = float(r['logLik'](mer)[0])
            #new_resds = r['resid'](mer)
            new_resds = np.zeros(Dw.shape[0])
            res.append((new_resds, new_betas, new_tvals, np.array([new_ll])))

    if len(res) == 0:
        # must make dummy data
        if lmer is None:
            O = [mm._O[i].copy() for i in ind_b]
            # if boot is not None:
            #     # replace the group
            #     for i, k in enumerate(mm._groups):
            #         O[i][mm._re_group] = k

            lmer = LMER(mm._formula_str, np.concatenate(O),
                        factors=mm._factors,
                        resid_formula_str=mm._resid_formula_str,
                        **mm._lmer_opts)

        Dw = np.random.randn(len(np.concatenate(O)))
        temp_resds, temp_b, temp_t, temp_ll = lmer.run(vals=Dw)

        for n in temp_t.dtype.names:
            temp_b[n] = 0.0
            temp_t[n] = 0.0
        temp_ll[0] = 0.0
        temp_resds = np.zeros(Dw.shape[0])
        res.append((temp_resds, temp_b, temp_t, temp_ll))


        # print "perm fail"

    # pull out data from all the components
    resds, betas, tvals, log_likes = zip(*res)
    resds = np.vstack(resds)
    betas = np.concatenate(betas)
    tvals = np.concatenate(tvals)
    log_likes = np.concatenate(log_likes)

    
    # recombine and scale the tvals across components
    # for k in tvals.dtype.names:
    #     if len(ss_norm) == 0:
    #         import pdb; pdb.set_trace()
    bs = np.rec.fromarrays([np.dot(betas[k], ss_norm)  # /(ss>0.).sum()
                            for k in tvals.dtype.names],
                           names=','.join(tvals.dtype.names))
    ts = np.rec.fromarrays([np.dot(tvals[k], ss_norm)  # /(ss>0.).sum()
                            for k in tvals.dtype.names],
                           names=','.join(tvals.dtype.names))

    # scale tvals across features
    # rfs = np.zeros()
    bfs = []
    tfs = []
    

    # in developing boot straps it appears that scaling the betas
    # hurts performance, so I'm just using Vh for the betas
    # only permutations use tfs, so I'm leaving that alone for now
    # TODO: evaluate permutations w/jackknife

    for k in tvals.dtype.names:
        # tfs.append(np.dot(tvals[k],
        #                   np.dot(diagsvd(ss[ss > 0],
        #                                  len(ss[ss > 0]),
        #                                  len(ss[ss > 0])),
        #                          Vh[ss > 0, ...])))  # /(ss>0).sum())
        bfs.append(blockwise_dot(betas[k], beta_project))
        tfs.append(blockwise_dot(tvals[k], ssVh))
    bfs = np.rec.fromarrays(bfs, names=','.join(tvals.dtype.names))
    tfs = np.rec.fromarrays(tfs, names=','.join(tvals.dtype.names))
    # Transform residuals back to feature space
    if boot is not None and mm._fvar_nboot == 0:
        rfs = resds.T @ ssVh
        rfs = np.array([rr.T @ rr for rr in rfs.T])
    else:
        rfs = np.zeros((Vh.shape[1]))

    # decide what to return
    if perm is None and boot is None:
        # return tvals, tfs, and R for actual non-permuted data
        out = (bs, ts, rfs, bfs, tfs, _R, feat_mask, _ss, mer)
    else:
        # return the tvals for the terms
        out = (bs, ts, rfs, bfs, tfs, ~feat_mask[0])
    return out


def _memmap_array(x, memmap_dir=None, use_h5py=False, unique_id=''):
    if memmap_dir is None:
        memmap_dir = tempfile.gettempdir()
    # generate the base filename
    filename = os.path.join(memmap_dir,
                            unique_id + '_' + str(id(x)))
    if use_h5py:
        import h5py
        filename += '.h5'
        h = h5py.File(filename)
        mmap_dat = h.create_dataset('mdat', data=x,
                                    compression='gzip')
        h.flush()
    else:
        # use normal memmap
        # filename = os.path.join(memmap_dir, str(id(x)) + '.npy')
        filename += '.npy'
        np.save(filename, x)
        mmap_dat = np.load(filename, 'r+')
    return mmap_dat

def gen_bal_boots(nsubj, nboots):
    """Generate indicies for balanced bootstraps as described 
       in https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470057339.vab028 
       page 4"""
    return np.random.choice(np.array([[ii for ii in range(nsubj)]]*(nboots)).flatten(), ((nboots), nsubj), replace=False)

def gen_jackknives(x, nj):
    """Given an array, return pseudo jackknifed samples of that array.
    Exact jackknife in the instance that nj == len(x), error if nj > len(x), 
    and leave more than one out if nj < len(x)"""
    a = len(x)
    # If nj is equal to a we'll do leave one out jackknife
    if nj == a:
        lno = 1
    elif nj > a:
        raise ValueError("Can't have more jacknife resamplings than subjects")
    else:
        lno = a//nj + 1

    a_inds = np.arange(a)
    left_out_inds = np.vstack((np.random.choice(a, (a//lno,lno), replace = False),
                               np.random.choice(a, (nj%(a//lno),lno), replace = False)))

    jks = []
    for loi in left_out_inds:
        jk = []
        for i in range(a):
            if i not in loi:
                jk.append(x[i])
        jks.append(np.array(jk))

    return jks

class MELD(object):
    """Mixed Effects for Large Datasets (MELD)

    me = MELD('val ~ beh + rt', '(beh|subj) + (rt|subj)', 'subj'
              dep_data, ind_data, factors = ['beh', 'subj'])
    me.run_perms(200)

    If you provide ind_data as a dict with a separate recarray for
    each group, you must ensure the columns match.


    """
    def __init__(self, fe_formula, re_formula,
                 re_group, dep_data, ind_data,
                 factors=None, row_mask=None,
                 dep_mask=None,
                 use_ranks=False, use_norm=True,
                 memmap=False, memmap_dir=None,
                 resid_formula=None,
                 svd_terms=None, 
                 feat_thresh=0.05, 
                 feat_nboot=1000, do_tfce=False,
                 tfce_svd=False, z_transform=True,
                 connectivity=None, shape=None,
                 dt=.01, E=2/3., H=2.0,
                 n_jobs=1, verbose=10,
                 lmer_opts=None):
        """

        dep_data can be an array or a dict of arrays (possibly
        memmapped), one for each group.

        ind_data can be a rec_array for each group or one large rec_array
        with a grouping variable.

        If connectivity is not passed, facing connectivity is used with
        fast cythonized TFCE. Otherwise PTSA's connectivity is used. 
        The fast TFCE's output is tested against HCP's output.

        """
        if verbose>0:
            sys.stdout.write('Initializing...')
            sys.stdout.flush()
            start_time = time.time()

        # save the formula
        self._formula_str = fe_formula + ' + ' + re_formula

        # see if there's a resid formula
        if resid_formula:
            # the random effects are the same
            self._resid_formula_str = resid_formula + ' + ' + re_formula
        else:
            self._resid_formula_str = None

        # save whether using ranks
        self._use_ranks = use_ranks

        # see the thresh for keeping a feature
        self._feat_thresh = feat_thresh
        # using anything else as a component threshold decreases
        # sensitivity to high signal
        self._component_thresh = 0.0
        self._feat_nboot = feat_nboot
        self._do_tfce = do_tfce
        self._z_transform = z_transform
        self._tfce_svd = tfce_svd
        self._connectivity = connectivity
        self._dt = dt
        self._E = E
        self._H = H
        # Eventually set the number of boots to use to
        # estimate feature level variance
        self._fvar_nboot = None

        # see if memmapping
        self._memmap = memmap

        # save job info
        self._n_jobs = n_jobs
        self._verbose = verbose

        # eventually fill the feature shape
        self._feat_shape = None

        # handle the dep_mask
        self._dep_mask = dep_mask

        # fill A,M,O,D
        self._A = {}
        self._M = {}
        self._O = {}
        self._D = {}
        O = []

        # loop over unique grouping var
        self._re_group = re_group
        if isinstance(ind_data, dict):
            # groups are the keys
            self._groups = np.array(ind_data.keys())
        else:
            # groups need to be extracted from the recarray
            self._groups = np.unique(ind_data[re_group])
        for g in self._groups:
            # get that subj inds
            if isinstance(ind_data, dict):
                # the index is just the group into that dict
                ind_ind = g
            else:
                # select the rows based on the group
                ind_ind = ind_data[re_group] == g

            # process the row mask
            if row_mask is None:
                # no mask, so all good
                row_ind = np.ones(len(ind_data[ind_ind]), dtype=np.bool)
            elif isinstance(row_mask, dict):
                # pull the row_mask from the dict
                row_ind = row_mask[g]
            else:
                # index into it with ind_ind
                row_ind = row_mask[ind_ind]

            # extract that group's A,M,O
            # first save the observations (rows of A)
            self._O[g] = ind_data[ind_ind][row_ind]
            if use_ranks:
                # loop over non-factors and rank them
                for n in self._O[g].dtype.names:
                    if (n in factors) or isinstance(self._O[g][n][0], str):
                        continue
                    self._O[g][n] = rankdata(self._O[g][n])
            O.append(self._O[g])

            # eventually allow for dict of data files for dep_data
            if isinstance(dep_data, dict):
                # the index is just the group into that dict
                dep_ind = g
            else:
                # select the rows based on the group
                dep_ind = ind_ind

            # save feature shape if necessary
            if self._feat_shape is None:
                self._feat_shape = dep_data[dep_ind].shape[1:]

            # handle the mask
            if self._dep_mask is None:
                self._dep_mask = np.ones(self._feat_shape,
                                         dtype=np.bool)

            # create the connectivity (will mask later)
            # if self._do_tfce and self._connectivity is None and \
            #    (len(self._dep_mask.flatten()) > self._dep_mask.sum()):
            #     # create the connectivity
            #     self._connectivity = cluster.sparse_dim_connectivity([cluster.simple_neighbors_1d(n)
            #                                                           for n in self._feat_shape])

            # Save D index into data (apply row and feature masks
            # This will also reshape it
            self._D[g] = dep_data[dep_ind][row_ind][:, self._dep_mask].copy()
            #self._D[g] = dep_data[dep_ind][row_ind].copy()
            #self._D[g][:, ~self._dep_mask] = 0
            # TODO: Make sure that there is still data in _D

            # reshape it
            # self._D[g] = self._D[g].reshape((self._D[g].shape[0], -1))

            if use_ranks:
                if verbose > 0:
                    sys.stdout.write('Ranking %s...' % (str(g)))
                    sys.stdout.flush()

                for i in range(self._D[g].shape[1]):
                    # rank it
                    self._D[g][:, i] = rankdata(self._D[g][:, i])

                    # normalize it
                    self._D[g][:, i] = ((self._D[g][:, i] - 1) /
                                        (len(self._D[g][:, i]) - 1))

            # save M from D so we can have a normalized version
            self._M[g] = self._D[g].copy()

            # remove any NaN's in dep_data
            self._D[g][np.isnan(self._D[g])] = 0.0

            # normalize M
            if use_norm:
                self._M[g] -= self._M[g].mean(0)
                self._M[g] /= np.sqrt((self._M[g]**2).sum(0))

            # determine A from the model.matrix
            rdf = DataFrame({k: (FactorVector(self._O[g][k])
                                 if k in factors else self._O[g][k])
                             for k in self._O[g].dtype.names})

            # model spec as data frame
            ms = r['data.frame'](r_model_matrix(Formula(fe_formula), data=rdf))

            cols = list(r['names'](ms))
            if svd_terms is None:
                self._svd_terms = [c for c in cols
                                   if 'Intercept' not in c]
            else:
                self._svd_terms = svd_terms

            # self._A[g] = np.vstack([ms[c] #np.array(ms.rx(c))
            self._A[g] = np.concatenate([np.array(ms.rx(c))
                                         for c in self._svd_terms]).T

            if use_ranks:
                for i in range(self._A[g].shape[1]):
                    # rank it
                    self._A[g][:, i] = rankdata(self._A[g][:, i])

                    # normalize it
                    self._A[g][:, i] = ((self._A[g][:, i] - 1) /
                                        (len(self._A[g][:, i]) - 1))

            # normalize A
            if True:  # use_norm:
                self._A[g] -= self._A[g].mean(0)
                if not ((self._A[g].max() == self._A[g].min()) & (self._A[g].min() == 0)):
                    self._A[g] /= np.sqrt((self._A[g]**2).sum(0))

            # memmap if desired
            if self._memmap:
                self._M[g] = _memmap_array(self._M[g], memmap_dir,
                                           unique_id=str(g))
                self._D[g] = _memmap_array(self._D[g], memmap_dir,
                                           unique_id=str(g))

        # save the new O
        self._O = O
        if lmer_opts is None:
            lmer_opts = {}
        self._lmer_opts = lmer_opts
        self._factors = factors

        # mask the connectivity
        if self._do_tfce and (self._connectivity is not None) and (len(self._dep_mask.flatten()) > self._dep_mask.sum()):
            self._connectivity = self._connectivity.tolil()[self._dep_mask.flatten()].tocsc()[:,self._dep_mask.flatten()].tocoo()

        # prepare for the perms and boots and jackknife
        self._perms = []
        self._boots = []
        self._bp = []
        self._tp = []
        self._rb = []
        self._bb = []
        self._tb = []
        self._tj = []
        self._pfmask = []

        if verbose > 0:
            sys.stdout.write('Done (%.2g sec)\n' % (time.time()-start_time))
            sys.stdout.write('Processing actual data...')
            sys.stdout.flush()
            start_time = time.time()

        global _global_meld
        _global_meld[id(self)] = self

        # run for actual data (returns both perm and boot vals)
        self._R = None
        self._ss = None
        self._mer = None
        bp, tp, rb, bb, tb, R, feat_mask, ss, mer = _eval_model(id(self), None)
        self._R = R
        self._bp.append(bp)
        self._tp.append(tp)
        self._rb.append(rb)
        self._bb.append(bb)
        self._tb.append(tb)
        self._feat_mask = feat_mask
        self._fmask = ~feat_mask[0]
        self._pfmask.append(~feat_mask[0])
        self._ss = ss
        self._mer = mer

        if verbose > 0:
            sys.stdout.write('Done (%.2g sec)\n' % (time.time()-start_time))
            sys.stdout.flush()

    def __del__(self):
        # get self id
        my_id = id(self)

        # clean up memmapping files
        if self._memmap:
            for g in self._M.keys():
                try:
                    filename = self._M[g].filename
                    del self._M[g]
                    os.remove(filename)
                except OSError:
                    pass
            for g in self._D.keys():
                try:
                    filename = self._D[g].filename
                    del self._D[g]
                    os.remove(filename)
                except OSError:
                    pass

        # clean self out of global model list
        global _global_meld
        if _global_meld and my_id in _global_meld:
            del _global_meld[my_id]

    def run_perms(self, perms, n_jobs=None, verbose=None, backend="multiprocessing"):
        """Run the specified permutations.

        This method will append to the permutations you have already
        run.

        """
        if self._boots:
            raise ValueError("You should not run perms and bootstraps on the same model. "
                             "You are trying to run perms and bootstraps have already been run")
        if n_jobs is None:
            n_jobs = self._n_jobs
        if verbose is None:
            verbose = self._verbose

        if not isinstance(perms, list):
            # perms is nperms
            nperms = perms

            # gen the perms ahead of time
            perms = []
            for p in range(nperms):
                ind = {}
                for k in self._groups:
                    # gen a perm for that subj
                    ind[k] = np.random.permutation(len(self._A[k]))

                perms.append(ind)
        else:
            # calc nperms
            nperms = len(perms)

        if verbose > 0:
            sys.stdout.write('Running %d permutations...\n' % nperms)
            sys.stdout.flush()
            start_time = time.time()

        # save the perms
        self._perms.extend(perms)
        # must use the threading backend
        res = Parallel(n_jobs=n_jobs,
                       backend=backend,
                       verbose=verbose)(delayed(_eval_model)(id(self), perm)
                                        for perm in perms)
        bp, tp, rfs, bfs, tfs, feat_mask = zip(*res)       
        self._bp.extend(bp)
        self._tp.extend(tp)
        self._rb.extend(rfs)
        self._bb.extend(bfs)
        self._tb.extend(tfs)
        self._pfmask.extend(feat_mask)

        if verbose > 0:
            sys.stdout.write('Done (%.2g sec)\n' % (time.time()-start_time))
            sys.stdout.flush()


    def run_boots(self, boots, fvar_nboot=None, n_jobs=None, verbose=None, backend="multiprocessing"):
        """Run the specified bootstraps.

        This method will append to the bootstraps you have already
        run.

        """
        if self._perms:
            raise ValueError("You should not run perms and bootstraps on the same model. "
                             "You are trying to run bootstraps, but perms have already been run")
        # Deal with the number of boots for estimating feature variance
        if self._fvar_nboot is None and fvar_nboot is not None:
            if fvar_nboot < 0:
                raise ValueError("fvar_nboot must be greater than 0")
            self._fvar_nboot =  fvar_nboot
        elif self._fvar_nboot is not None and  fvar_nboot is not None and self._fvar_nboot !=  fvar_nboot:
            raise Exception("You've already set the number of bootsrtaps to use for"
                            "estimating feature vaiance to %d. You can't change it now."
                            "I haven't actually made the attribute read only,"
                            "but if you set it to another value, your ts and ps"
                            "will no longer be accurate."%self._fvar_nboot)
        elif self._fvar_nboot is None and fvar_nboot is None:
            self._fvar_nboot = 1


        if n_jobs is None:
            n_jobs = self._n_jobs
        if verbose is None:
            verbose = self._verbose

        if not isinstance(boots, list):
            # boots is nboots
            nboots = boots
            boots = []
            # add jackknife resamplings for original data
            if not self._boots:
                if self._fvar_nboot > len(self._groups):
                    boots.extend(gen_bal_boots(len(self._groups),self._fvar_nboot))
                elif self._fvar_nboot == 0:
                    pass
                else:
                    boots.extend(gen_jackknives(np.arange(len(self._groups)),self._fvar_nboot))

            bal_boots = gen_bal_boots(len(self._groups),nboots)
            
            for boot in bal_boots:
                boots.append(boot)
                # Bootstrap if more inner than samples
                # otherwise jackknife
                if self._fvar_nboot > len(boot):
                    boots.extend(boot[gen_bal_boots(len(boot),self._fvar_nboot)])
                elif self._fvar_nboot == 0:
                    pass
                else:
                    boots.extend(gen_jackknives(boot,self._fvar_nboot))

        else:
            # calc nboots
            nboots = len(boots)
            if (nboots+1) % (self._fvar_nboot+1) != 0:
                raise ValueError("The list of boostrapts provided is not"
                                 "evenly divisible by fvar_nboot.")

        if verbose > 0:
            sys.stdout.write('Running %d bootstraps'%len(boots))
            sys.stdout.flush()
            start_time = time.time()

        # save the perms
        self._boots.extend(boots)

        # must use the threading backend
        res = Parallel(n_jobs=n_jobs,
                       backend=backend,
                       verbose=verbose)(delayed(_eval_model)(id(self), boot=boot)
                                        for boot in boots)
        bp, tp, rfs, bfs, tfs, feat_mask = zip(*res)
        self._bp.extend(bp)
        self._tp.extend(tp)
        self._rb.extend(rfs)
        self._bb.extend(bfs)
        self._tb.extend(tfs)
        self._pfmask.extend(feat_mask)

        if verbose > 0:
            sys.stdout.write('Done (%.2g sec)\n' % (time.time()-start_time))
            sys.stdout.flush()

    @property
    def terms(self):
        return self._tp[0].dtype.names

    @property
    def t_terms(self):
        return self._tp[0]

    @property
    def t_features(self):
        return self.get_t_features()

    def get_t_features(self, names=None):
        if names is None:
            names = [n for n in self.terms
                     if n != '(Intercept)']
        tfeats = []
        if not self._boots:
            for n in names:
                tfeat = np.zeros(self._feat_shape)
                tfeat[self._dep_mask] = self._tb[0][n][0]
                tfeats.append(tfeat)
        else:
            
            bpf = self._bb[0].__array_wrap__(np.hstack(self._bb))
            nperms = (len(self._boots)+1)//(self._fvar_nboot+1)
            pfmasks = np.array(self._pfmask).transpose((1, 0, 2))
            if self._fvar_nboot > 0:
                for i, n in enumerate(names):
                    fmask = pfmasks[i]
                    bf = bpf[n]
                    bf = bf.reshape(fmask.shape[0], -1)
                    bf[~fmask] = 0
                    bf = bf.reshape(nperms,self._fvar_nboot+1, -1)

                    # Nested bootstrap gives us mean and standard error
                    boot_mean = bf[:,0,:]
                    boot_sterr = ((bf.shape[1]-1)/bf.shape[1])*np.sqrt(np.sum(((bf[:,0,:].reshape(bf.shape[0], 1, bf.shape[-1]) - bf[:,1:,:])**2), 1))
                    tf = stat_helper.boot_stat(boot_mean, boot_sterr)
                    tfeat = np.zeros(self._feat_shape)
                    tfeat[self._dep_mask] = tf[0]
                    tfeats.append(tfeat)
            elif  self._fvar_nboot == 0:
                c = np.identity(len(names))
                O = np.concatenate(self._O)
                
                rsds_all = np.array(self._rb)
                for i, n in enumerate(names):
                    fmask = pfmasks[i]
                    # calculate m
                    m = O[n].astype([(n,np.float64)]).view(dtype=np.float64, type=np.ndarray)
                    m_term = m.T @ m
                    # calculate residual
                    rsds= np.sqrt(rsds_all/(m.shape[0] - np.linalg.matrix_rank(m)))
                    rsds = rsds.reshape(nperms, -1)

                    bf = bpf[n]
                    bf = bf.reshape(fmask.shape[0], -1)
                    bf[~fmask] = 0
                    bf = bf.reshape(nperms, -1)
                    
                    boot_mean = bf
                    boot_sterr = np.sqrt(m_term)*(rsds/(len(m)-np.linalg.matrix_rank(m)))
                    tf = stat_helper.boot_stat(boot_mean, boot_sterr)
                    tfeat = np.zeros(self._feat_shape)
                    tfeat[self._dep_mask] = tf[0]
                    tfeats.append(tfeat)
        return np.rec.fromarrays(tfeats, names=','.join(names))

    @property
    def p_features(self):
        return self.get_p_features()

    def get_p_features(self, names=None, conj=None, do_tfce=True):
        tpf = self._tb[0].__array_wrap__(np.hstack(self._tb))
        pfmasks = np.array(self._pfmask).transpose((1, 0, 2))
        if self._perms:
            nperms = np.int(len(self._perms)+1)
        elif self._boots:
            bpf = self._bb[0].__array_wrap__(np.hstack(self._bb))
            nperms = (len(self._boots)+1)//(self._fvar_nboot+1)


        else:
            raise Exception("Must run some boots or perms before getting ps")

        tfs = []
        if names is None:
            if conj is not None:
                # use the conj if it's not none
                names = conj
            else:
                names = [n for n in tpf.dtype.names
                         if n != '(Intercept)']
        if self._perms:
            pfs = []

            for i, n in enumerate(names):
                fmask = pfmasks[i]
                tf = tpf[n]
                tf = np.abs(tf.reshape(int(nperms), -1))
                fmask = pfmasks[i]
                tf[~fmask] = 0
                nullTdist = tf.max(1)
                nullTdist.sort()
                pf = ((nperms-np.searchsorted(nullTdist, tf.flatten(), 'left')) /
                      nperms).reshape(nperms, -1)
                pfs.append(pf)

            # pfs is terms by perms by features
            pfs = np.array(pfs)

            # handle conjunction
            if conj is not None:
                # get max p-vals across terms for each perm and feature
                pfs = pfs.max(0)[np.newaxis]

                # set the names to be new conj
                names = ['&'.join(names)]
           
            # make null p distribution
            nullPdist = pfs.min(2).min(0)
            nullPdist.sort()

            # get pvalues for each feature for each term
            pfts = np.searchsorted(nullPdist,
                                   pfs[:, 0, :].flatten(),
                                   'right').reshape(len(pfs), -1)/nperms

            pfeats = []
            for n in range(len(names)):
                pfeat = np.ones(self._feat_shape)
                pfeat[self._dep_mask] = pfts[n]
                pfeats.append(pfeat)

            # reconstruct the recarray
            pfts = np.rec.fromarrays(pfeats, names=','.join(names))

            return pfts

        elif self._boots:
            if self._fvar_nboot > 0:
                for i, n in enumerate(names):
                    fmask = pfmasks[i]
                    bf = bpf[n]
                    bf = bf.reshape(fmask.shape[0], -1)
                    bf[~fmask] = 0
                    bf = bf.reshape(nperms,self._fvar_nboot+1, -1)

                    # If the we've got more inner boots than people, do a bootstrap
                    # That means we've got a different formula for standard error and mean
                    if self._fvar_nboot > len(self._groups):
                        boot_mean = bf.mean(1)
                        boot_sterr = (1/self._fvar_nboot) * np.sqrt(np.sum(((boot_mean.reshape(bf.shape[0], 1, bf.shape[-1]) - bf[:,0:,:])**2), 1))
                    else:
                        # Nested bootstrap and jackknife gives us mean and standard error
                        # bootstrap hypothesis test is based on:
                        # http://www.jstor.org/stable/2532163?seq=3#page_scan_tab_contents
                        boot_mean = bf[:,0,:]
                        # Jackknife standard error from http://people.bu.edu/aimcinto/jackknife.pdf
                        boot_sterr = ((bf.shape[1]-1)/bf.shape[1])*np.sqrt(np.sum(((bf[:,0,:].reshape(bf.shape[0], 1, bf.shape[-1]) - bf[:,1:,:])**2), 1))
                    # calculate bootstrap hypothesis test stat taking into account
                    # guidelines from http://www.jstor.org/stable/2532163?seq=2#page_scan_tab_contents
                    tf = stat_helper.boot_stat(boot_mean, boot_sterr)
                    tfeats = np.zeros((nperms, np.product(self._feat_shape)))
                    tfeats[:,self._dep_mask.flatten()] = tf
                    tfs.append(tfeats)
            elif self._fvar_nboot == 0:
                c = np.identity(len(names))
                O = np.concatenate(self._O)
                
                rsds_all = np.array(self._rb)

                for i, n in enumerate(names):
                    fmask = pfmasks[i]
                    # Calculate m for term
                    m = O[n].astype([(n,np.float64)]).view(dtype=np.float64, type=np.ndarray)
                    m_term = m.T @ m
                    # calculate residual for term
                    rsds= np.sqrt(rsds_all/(m.shape[0] - np.linalg.matrix_rank(m)))
                    rsds = rsds.reshape(nperms, -1)

                    bf = bpf[n]
                    bf = bf.reshape(fmask.shape[0], -1)
                    bf[~fmask] = 0
                    bf = bf.reshape(nperms, -1)

                    boot_mean = bf
                    boot_sterr = np.sqrt(m_term)*(rsds)
                    tf = stat_helper.boot_stat(boot_mean, boot_sterr)
                    tfeats = np.zeros((nperms, np.product(self._feat_shape)))
                    tfeats[:,self._dep_mask.flatten()] = tf
                    tfs.append(tfeats)
                    #import pdb; pdb.set_trace() 

            # tfs is terms by boots by features
            tfs = np.array(tfs)

            # handle conjunction
            if conj is not None:
                # get max p-vals across terms for each perm and feature
                tfs = tfs.min(0)[np.newaxis]

                # set the names to be new conj
                names = ['&'.join(names)]
            if do_tfce == True:
                tfces = np.array([[tfce.tfce(tfs[i,j].reshape(self._feat_shape),
                                             param_e=self._E,
                                             param_h=self._H,
                                             pad=True) 
                                   for j in range(tfs.shape[1])] 
                                   for i in range(tfs.shape[0])]).reshape(tfs.shape)
                tfs = tfces

            nullTdist = tfs.max(0).max(-1)
            nullTdist.sort()
            # get pvalues for each feature for each term
            pfts = (nperms-np.searchsorted(nullTdist,
                           tfs[:, 0, :].flatten(),
                           'left').reshape(tfs.shape[0], *self._feat_shape))/nperms

            pfts = np.rec.fromarrays(pfts, names=','.join(names))

            return pfts


if __name__ == '__main__':
    np.random.RandomState(seed= 42)

    # test some MELD
    n_jobs = -1
    verbose = 10

    # generate some fake data
    nobs = 100
    nsubj = 10
    nfeat = (10, 30)
    nperms = 200
    use_ranks = False
    smoothed = False
    memmap = False

    s = np.concatenate([np.array(['subj%02d' % i] * nobs)
                        for i in range(nsubj)])
    beh = np.concatenate([np.array([1] * (nobs/2) + [0]*(nobs / 2))
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
            dep_data_s[:, 4, i+j] += (ind_data['beh'] * (i+1)/50.)
            dep_data_s[:, 5, i+j] += (ind_data['beh'] * (i+1)/50.)
            dep_data_s[:, 5, i+j] += (ind_data['beh2'] * (i+1)/50.)
            dep_data_s[:, 6, i+j] += (ind_data['beh2'] * (i+1)/50.)

    # smooth the data
    if smoothed:
        import scipy.ndimage
        dep_data = scipy.ndimage.gaussian_filter(dep_data, [0, 1, 1])
        dep_data_s = scipy.ndimage.gaussian_filter(dep_data_s, [0, 1, 1])

    print("Starting MELD test")
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
