import numpy as np

def _ecdf(x):
    '''no frills empirical cdf used in fdrcorrection
    '''
    nobs = len(x)
    return np.arange(1, nobs + 1) / float(nobs)

def fdr_correction(pvals, alpha=0.05, method='indep'):
    """P-value correction with False Discovery Rate (FDR)

    Correction for multiple comparison using FDR.

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.

    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate
    method : 'indep' | 'negcorr'
        If 'indep' it implements Benjamini/Hochberg for independent or if
        'negcorr' it corresponds to Benjamini/Yekutieli.

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not
    pval_corrected : array
        pvalues adjusted for multiple hypothesis testing to limit FDR

    Notes
    -----
    Reference:
    Genovese CR, Lazar NA, Nichols T.
    Thresholding of statistical maps in functional neuroimaging using the false
    discovery rate. Neuroimage. 2002 Apr;15(4):870-8.
    """
    pvals = np.asarray(pvals)
    shape_init = pvals.shape
    pvals = pvals.ravel()

    pvals_sortind = np.argsort(pvals)
    pvals_sorted = pvals[pvals_sortind]
    sortrevind = pvals_sortind.argsort()

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1. / np.arange(1, len(pvals_sorted) + 1))
        ecdffactor = _ecdf(pvals_sorted) / cm
    else:
        raise ValueError("Method should be 'indep' and 'negcorr'")

    reject = pvals_sorted < (ecdffactor * alpha)
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
    else:
        rejectmax = 0
    reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    pvals_corrected[pvals_corrected > 1.0] = 1.0
    pvals_corrected = pvals_corrected[sortrevind].reshape(shape_init)
    reject = reject[sortrevind].reshape(shape_init)
    return reject, pvals_corrected

def boot_stat(stat, sterr):
    """Calculate a boostrap hypothesis testing statistic.
    Based on recomendations for bootstrap hypothesis testing from
    Hall and Wilson, 1991.

    Parameters
    ----------
    stat: array
        Array of statistic to be tested for each bootstrap. Nboostraps x [feature dimensions].
        Statistic for original data is expected to be first.
    standard error: array
        Array of standard errors of each statistic. Nboostraps x [feature dimensions].

    Returns
    -------
    bjt: array
        Array of bootstrap statistic Nboostraps x [feature dimensions].

    Notes
    -----
    Reference:
    Hall P, Wilson SR, Two Guidelines for Bootstrap Hypothesis Testing. 
    Biometrics. June 1991 47, 757-762
    """
    orig_shape = stat.shape[1:]
    stat = stat.reshape(stat.shape[0], -1)
    sterr = sterr.reshape(nperms, -1)

    # bootstrap hypothesis test is based on:
    # http://www.jstor.org/stable/2532163?seq=3#page_scan_tab_contents
    bjt = np.zeros(stat.shape)
    # bootjack t for original data is just the original data divided by its bootstraps
    bjt[0][sterr[0] != 0] = np.abs(((stat[0])[sterr[0] != 0])/sterr[0][sterr[0] != 0])
    # bootjack t for rest of null distribution is stat_repl - stat_orig/standarderror_repl
    bjt[1:][sterr[1:] != 0] = np.abs(((stat[1:] - stat[0])[sterr[1:] != 0])/sterr[1:][sterr[1:] != 0])
    bjt = bjt.reshape(-1, *orig_shape)

    return bjt


