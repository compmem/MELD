# coding: utf-8

# In[1]:

import os
import sys
import time
import numpy as np
#import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
#import process_log as pl
import pandas as pd
import pickle
from joblib import Parallel,delayed
from pathlib import Path
from meld import meld
from meld import tfce
from meld import signal_gen as msg
from meld import stat_helper as msh
from meld.nonparam import gen_perms
import argparse
from meld.meld import LMER

# #Defining functions

def get_error_rates(bThr,brs,signal,terms,pvals,full=False):
    """calculate error rate at a single test statistic threshold (bThr)"""
    if full == True:
        pvalThr=0.05
    else:
        pvalThr = 1-bThr
    no_signal = abs(abs(signal) -1.0)
    tp = 0.
    fp = 0.
    fn = 0.
    tn = 0.
    #set postitive and negative to the the number of signal or noise voxels * the number of terms
    if 'Intercept' in terms or '(Intercept)' in terms:        
        p = (len(terms)-1) * np.count_nonzero(signal)
        n = (len(terms)-1) * np.count_nonzero(no_signal)
    else:
        p = (len(terms)) * np.count_nonzero(signal)
        n = (len(terms)) * np.count_nonzero(no_signal)
    for i in range(len(terms)):
        #meld calls the intercept (Intercept), GLM doesn't use parentheses
        if terms[i] != '(Intercept)' and terms[i] != 'Intercept' :
            if pvals[terms[i]] <= pvalThr :
                tp += np.count_nonzero(((abs(brs[terms[i]])>=bThr))*signal)
                fp += np.count_nonzero(((abs(brs[terms[i]])>=bThr))*no_signal)
                if full == True:
                    fn += np.count_nonzero(((abs(brs[terms[i]])<bThr))*signal)
                    tn += np.count_nonzero(((abs(brs[terms[i]])<bThr))*no_signal)
            elif full == True:
                fn += np.count_nonzero(signal)
                tn += np.count_nonzero(no_signal)
    #true positive rate = true positives found divided by potential number of true positives
    try:
        tpr = tp / p
    except:
        tpr = np.NaN
    #false negative rate = false negatives divided by potential number of false negatives
    try:
        fnr = fn / p
    except:
        fnr = np.NaN
    #false positive rate = false positives found divided by potential number of false positives
    fpr = fp / n
    tnr = tn / n
    if full == True:
        return(tpr,fpr,tnr,fnr)
    else:
        return (tpr,fpr)

#new error rate function that just gets error rate for the map it's passed, 
#doesn't care about terms
def get_error_terms(bmap,bsig):
    tp = np.float64(np.sum(bmap*bsig))
    fp = np.float64(np.sum(bmap*~bsig))
    fn = np.float64(np.sum(~bmap*bsig))
    tn = np.float64(np.sum(~bmap*~bsig))
    mccD =(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    if mccD == 0.:
        mccD = 1.
    else:
        mccD = np.sqrt(mccD)
    
    mcc = ((tp*tn)-(fp*fn))/mccD
    
    error_terms = np.array([(tp, 
                             fp,
                             fn,
                             tn,
                             tp/(tp+fn),
                             tn/(fp+tn),
                             tp/(tp+fp),
                             tn/(tn+fn),
                             fp/(fp+tn),
                             fp/(fp+tp),
                             fn/(fn+tp),
                             (tp+tn)/((tp+fn)+(fp+tn)),
                             (2*tp)/(2*tp+fp+fn),
                             mcc)],
                           dtype=np.dtype([('tp', 'float64'),
                                           ('fp', 'float64'),
                                           ('fn', 'float64'),
                                           ('tn', 'float64'),
                                           ('tpr', 'float64'),
                                           ('spc', 'float64'),
                                           ('ppv', 'float64'),
                                           ('npv', 'float64'),
                                           ('fpr', 'float64'),
                                           ('fdr', 'float64'),
                                           ('fnr', 'float64'),
                                           ('acc', 'float64'),
                                           ('f1', 'float64'),
                                           ('mcc', 'float64')]))

    return error_terms

def get_ROC(statmap,signal,start=0.,stop=1.,num=1000,thr=None, ret_thr = False,ret_mcc = False):
    """evaluate true and false positive rates for 1000 values from [0:1],
       calcualtes area under tpr vs fpr (ROC) curve based on trapezoidal method,
       expects positive results to be greater than threshold"""
    if thr is None:
        thr = np.linspace(start,stop,num)
        thr = np.append(thr,thr[-1]+thr[1])

    tprs = np.zeros(len(thr))
    fprs = np.zeros(len(thr))
    mcc = np.zeros(len(thr))
    for i in range(len(thr)):
        #get error rates for each bThr
        et=get_error_terms(statmap>=thr[i],abs(signal)>0)
        tprs[i] = et['tpr']
        fprs[i] = et['fpr']
        mcc[i] = et['mcc']
    data = np.array(sorted(zip(fprs,tprs)))
    #find area under curve using trapezoidal method
    under = np.trapz(data[:,1],data[:,0])
    if ((ret_thr == True)&(ret_mcc==True)):
        return (under, tprs,fprs,thr,mcc)
    elif ret_thr == True:
        return (under, tprs,fprs,thr)
    elif ret_mcc == True:
        return (under, tprs,fprs,mcc)
    else:
        return (under,tprs,fprs)


def eval_bthrs(brs,signal,terms,pvals):
    """evaluate true and false positive rates for a variety of thresholds,
       calcualtes area under tpr vs fpr (ROC) curve based on trapezoidal method"""
    bThr = []
    for t in terms:
        if t != '(Intercept)' and t != 'Intercept' :
            bThr = np.append(bThr,brs[t].flatten())
    bThr = np.unique(bThr)
    tprs = np.zeros(len(bThr))
    fprs = np.zeros(len(bThr))
    for i in range(len(bThr)):
        #get error rates for each bThr
        (tprs[i],fprs[i])=get_error_rates(bThr[i],brs,signal,terms,pvals)
    data = np.array(sorted(zip(fprs,tprs)))
    #find area under curve using trapezoidal method
    under = np.trapz(data[:,1],data[:,0])
    return (under,tprs,fprs)


# In[7]:

def eval_glm(fe_formula,ind_data,dep_data):
    """evaluate one subject's worth of GLM data at each feature"""
    betas = {}
    #get shape of data
    nfeat = np.shape(dep_data[0])
    #iterate over each feature in data
    for p in range(nfeat[0]):
        for q in range(nfeat[1]):
            #set val in ind_data equal to dep_data for that feature
            ind_data['val']=dep_data[:,p,q]
            #fit glm based on model given in fe_formula
            modl= smf.glm(formula=fe_formula, data=ind_data).fit()
            #save beta's for each factor to a dict
            for fac in modl.pvalues.keys():
                if fac not in betas:
                    betas[fac]=np.zeros(nfeat)                
                betas[fac][p,q]=modl.params[fac]
    return betas


# In[8]:

#execution time: 
#size,    pthresh, real,    user
#100x100, p= 0.05, 63.999s, 463.004s
#100x100, p= 0.01, 32.149s, 249.616s
#100x100, p=0.10, 101.395s, 729.352s
#set up alphasim lists
#alphasim value for clusters of index+1 voxels in size
asims_dd={'32x32':{'0.01':[0.999974,
               0.175825,
               0.005355,
               0.000164,
               0.000007,
               0.000001],
       '0.05':[0.999999,
               0.987283,
               0.444893,
               0.081238,
               0.012722,
               0.002005,
               0.000322,
               0.00005,
               0.000008],
       '0.1':[0.999999,
              0.999999,
              0.979591,
              0.64708,
              0.255378,
              0.083128,
              0.025962,
              0.008131,
              0.002557,
              0.000791,
              0.000238,
              0.000072,
              0.000017,
              0.000007,
              0.000003]},
       '100x100':{'0.01':[1.000000,
                          0.853355,
                          0.054625,
                          0.001695,
                          0.000040],
                  '0.05':[1.000000,
                          1.000000,
                          0.997517,
                          0.585414,
                          0.127460,
                          0.021857,
                          0.003581,
                          0.000600,
                          0.000106,
                          0.000011,
                          0.000004,
                          0.000001],
                  '0.10':[1.000000,
                           1.000000,
                           1.000000,
                           0.999983,
                           0.955854,
                           0.607394,
                           0.251411,
                           0.086920,
                           0.028865,
                           0.009338,
                           0.003028,
                           0.000993,
                           0.000328,
                           0.000114,
                           0.000043,
                           0.000013,
                           0.000003,
                           0.000001,
                           0.000001]}}


# In[9]:

def get_alphamap(tvals,pvals,pthr,makez=False,asims=asims_dd['32x32']):
    """code for correction of glm pvals based on frequency of finding
    a cluster of a given size in a field of randomly generated noise.
    Data generated useing Alphasim with 1000000 simulations with no smoothing."""
    #pthr should be equal to 0.1,0.05, or 0.01

    clustmap={}
    numclust={}
    #things get weird in here dealing with p-vals and positive and negative tvals
    #my solution for positive and negative is to cluster them separately and combine them
    #    
    clustmap['pos'],numclust['pos']=ndimage.measurements.label(np.logical_and(pvals<=pthr,tvals > 0))
    clustmap['neg'],numclust['neg']=ndimage.measurements.label(np.logical_and(pvals<=pthr,tvals < 0))
    minp=1
    alphazmap= {k:clustmap[k].astype(np.float) for k in clustmap.keys()}
    for tsign in clustmap.keys():
        for numvox in range(1,numclust[tsign]+1):
            if np.sum(clustmap[tsign]==numvox) > len(asims[str(pthr)]):
                if makez == True:
                    alphazmap[tsign][clustmap[tsign]==numvox]=stats.norm.ppf(asims[str(pthr)][-1]/2,0,1)
                else:
                    alphazmap[tsign][clustmap[tsign]==numvox]=asims[str(pthr)][-1]
                minp=asims[str(pthr)][-1]
            else:
                if makez==True:
                    alphazmap[tsign][clustmap[tsign]==numvox]=stats.norm.ppf(asims[str(pthr)][np.sum(clustmap[tsign]==numvox)-1]/2,0,1)
                else:
                    alphazmap[tsign][clustmap[tsign]==numvox]=asims[str(pthr)][np.sum(clustmap[tsign]==numvox)-1]
                if asims[str(pthr)][np.sum(clustmap[tsign]==numvox)-1] < minp:
                    minp = asims[str(pthr)][np.sum(clustmap[tsign]==numvox)-1]
        
    
    #The p-val returrned is the minimum alphasim corrected pvalue for any cluster
    if makez == True:
        alphamap = alphazmap['neg'] - alphazmap['pos']
    else:
        alphamap=alphazmap['neg'] + alphazmap['pos']
        alphamap[alphamap==0]=1
    #out = dict(alphamap=alphamap,
    #           pval=minp)
    return alphamap


# In[10]:

def get_fdrmap(ts,ret ='qval',stat='normal',center=True):
    tshape = ts.shape
    if center == True:
        fachist = np.histogram(ts,bins=500)
        peak = fachist[1][fachist[0]==np.max(fachist[0])][0]
        ts -= peak
    ts = FloatVector(ts.flatten())
    results = fdrtool.fdrtool(ts,statistic=stat,
                              plot=False, verbose=False)
    return np.array(results.rx(ret)).reshape(tshape)


# In[11]:

#functions needed for ttest
# calc stats for each perm
def eval_perm(perm,ind_data,dep_data):   
    # get the permed ind_data
    pdat = ind_data[perm]
    
    # get the condition indicies based on the permed ind_data
    behA_ind =pdat['beh']==-0.5
    behB_ind = pdat['beh']==0.5
    
    # loop over subjects and calc diff
    vals = []
    for s in np.unique(list(pdat['subj'])):
        # get the subject index
        subj_ind = (pdat['subj']==s)
        
        # get diff in means for that index
        valdiff = (np.average(dep_data[behA_ind&subj_ind],axis=0)-np.average(dep_data[behB_ind&subj_ind],axis=0))
        
        vals.append(valdiff)
    vals = np.array(vals)
    
    # perform stats across subjects
    t,p = stats.ttest_1samp(vals, 0.0, axis=0)
    
    return t[np.newaxis,:]

# def run_tfce(t,con,dt=0.05,E=2/3.,H=2.0):
#     # enhance clusters in both directions, then combine
#     ttemp = cl.tfce(t, tail=1,connectivity=con,dt=dt,E=E, H=H,)
#     ttemp += cl.tfce(t, tail=-1, connectivity=con,dt=dt,E=E, H=H,)
#     # prepend axis (for easy concatenation) and return
#     return ttemp[np.newaxis,:]


def get_metrics(statmap, fac, p_boot_thr, signal):
    me_et=get_error_terms((statmap)>=(1.-p_boot_thr),np.abs(signal*slope)>0)
    metrics = {k:val[0] for k,val in pd.DataFrame(me_et).to_dict('list').items()}
    metrics['under'],metrics['tprs'],metrics['fprs'],metrics['thr'],metrics['roc_mcc']=get_ROC(statmap, signal*slope,num=1000,ret_thr = True,ret_mcc=True)
    metrics['factor'] = fac
    metrics['thr']= 1-metrics['thr']
    
    metrics['peak_mcc'] = np.max(metrics['roc_mcc'])
    metrics['peak_mcc_thr'] = metrics['thr'][np.argmax(metrics['roc_mcc'])]
    
    return metrics

def eval_glm_subjs(fe_formula, ind_data, dep_data, single_subject_res=None, boot=None,
                   n_jobs=2, backend="multiprocessing"):
    subj_res = []
    subjs = np.unique(ind_data['subj'])
    if boot is not None:
        subjs = subjs[boot]
    if single_subject_res is not None:
        for subj in subjs:
            subj_res.append(single_subject_res[subj])
    else:
        tmp_res = Parallel(n_jobs=n_jobs,
            backend=backend,
            verbose=10)(delayed(eval_glm)(fe_formula, 
                                     ind_data[ind_data.subj == subj],
                                     dep_data[ind_data.subj == subj])
                        for subj in subjs)
        subj_res = [tr['beh'] for tr in tmp_res]
    #if subj_res is not None:
    #    subj_res = np.array(np.array([sr['beh'] for sr in subj_res]))
    return subj_res

def run_lmer(lmer_id, vals, variables):
    lm = _global_lmer[lmer_id]
    betas = np.zeros(len(variables))
    tvals = np.zeros(len(variables))
    for i,b in (enumerate(tvals)):
        _betas, _tvals, _log_likes = lm.run(vals=vals)
        for j, var in enumerate(variables):
            tvals[j] = _tvals[var]
            betas[j] = _betas[var]
    return betas, tvals

def run_lmer_boot(boot, ind_data, lmer_dep_data, formula, variables):
    boot_ind = []
    boot_dep = []
    for i,bs in enumerate(boot):
        tmp_inds = ind_data[ind_data['subj']==bs].copy()
        tmp_inds['subj'] = i
        boot_ind.append(tmp_inds)
        boot_dep.append(lmer_dep_data[ind_data['subj']==bs])
    boot_ind = np.concatenate(boot_ind)
    boot_dep = np.concatenate(boot_dep)

    lmer_betas = []
    lmer_tvals = []
    lm = LMER(formula, boot_ind)
    flat_dep_data = boot_dep.reshape(boot_dep.shape[0], -1)
    res = []
    feat_betas = []
    feat_tvals = []
    for i in np.arange(flat_dep_data.shape[-1]):
        betas = np.zeros(len(variables))
        tvals = np.zeros(len(variables))
        _resds, _betas, _tvals, _log_likes = lm.run(vals=flat_dep_data[:,i])
        for j, var in enumerate(variables):
            betas[j] = _betas[var]
            tvals[j] = _tvals[var]
        feat_betas.append(betas)
        feat_tvals.append(tvals)
    lmer_betas.extend(np.array(feat_betas).squeeze().T.reshape( *boot_dep.shape[1:]))
    lmer_tvals.extend(np.array(feat_tvals).squeeze().T.reshape( *boot_dep.shape[1:]))
    return lmer_betas, lmer_tvals

def run_lmer_perm(perm, ind_data, lmer_dep_data, formula, variables):
    perm_ind = []
    perm_dep = []
    flat_perm = []
    for sid in np.unique(ind_data['subj']):
        flat_perm.extend(perm[sid])

    lmer_tvals = []
    lm = LMER(formula, ind_data)
    flat_dep_data = lmer_dep_data.reshape(lmer_dep_data.shape[0], -1)
    res = []
    feat_tvals = []
    for i in np.arange(flat_dep_data.shape[-1]):
        tvals = np.zeros(len(variables))
        _tvals, _log_likes = lm.run(vals=flat_dep_data[flat_perm,i])
        for j, var in enumerate(variables):
            tvals[j] = _tvals[var]
        feat_tvals.append(tvals)
    lmer_tvals.extend(np.array(feat_tvals).squeeze().T.reshape( *lmer_dep_data.shape[1:]))
    return lmer_tvals


def get_boot_p(stat, nperms, fvar_nboot, dep_data, do_tfce=False, E=0.6666666666, H=2):
    stat = stat.copy()
    tfs = []
    
    stat = stat.reshape(nperms,fvar_nboot+1, -1)
    # Nested bootstrap and jackknife gives us mean and standard error
    # bootstrap hypothesis test is based on:
    # http://www.jstor.org/stable/2532163?seq=3#page_scan_tab_contents
    boot_mean = stat[:,0,:]
    # Jackknife standar error from http://people.bu.edu/aimcinto/jackknife.pdf
    boot_sterr = ((stat.shape[1]-1)/stat.shape[1])*np.sqrt(np.sum(((stat[:,0,:].reshape(stat.shape[0], 1, stat.shape[-1]) - stat[:,1:,:])**2), 1))

    tf = msh.boot_stat(boot_mean, boot_sterr)
    tf = tf.reshape(-1, *dep_data.shape[1:])

    tfs.append(tf)
    tfs = np.array(tfs)

    if do_tfce == True:
        tfces = np.array([[tfce.tfce(tfs[i,j].reshape(self._feat_shape),
                                     param_e=self._E,
                                     param_h=self._H,
                                     pad=True) 
                           for j in range(tfs.shape[1])] 
                           for i in range(tfs.shape[0])]).reshape(tfs.shape)
        tfs = tfces
    nullTdist = tfs.max(0).max(-1).max(-1)
    nullTdist.sort()
    # get pvalues for each feature for each term
    pfts = (nperms-np.searchsorted(nullTdist,
                   tfs[:, 0, :].flatten(),
                   'left').reshape(tfs.shape[0], *dep_data.shape[1:]))/nperms
    return tfs[:,0,:].reshape(tfs.shape[0], *dep_data.shape[1:]), pfts

#variable for path to bootstraps boots?
def test_sim_dat(nsubj,nobs,slope,signal,signal_name,run_n,prop,mnoise=False,contvar=False,item_inds=False,
                 mod_mnoise=False,mod_cont=False,nfeat=(32,32),
                 n_jobs = 2,pthr=0.05,nperms=200,nboots=50,fvar_nboot = 10,I=0.0,S=0.0,
                 feat_nboot=1000,feat_thresh=0.1,connectivity=None,asims_dd=asims_dd,E=2/3.,
                 H=2.0, data=None, backend="multiprocessing"):

#generate data, now with the option to accept external data
    if data is None:
        (ind_data,dep_data,sn_stats) = msg.gen_data(nsubj,nobs,nfeat,slope,signal,
                                            mnoise=mnoise,contvar=contvar,
                                            item_inds=item_inds,mod_cont=mod_cont,I=I,S=S)
    else:
        (ind_data,dep_data,sn_stats) = data

    #set up formulas
    #This structure models subject level noise on the slope and intercept
    re_formula = '(0+beh|subj)'
    if mod_cont == True:
        fe = 'beh*cont'
        re_formula += ' + (0+cont|subj) + (0+beh:cont|subj)'
    else:
        fe = 'beh'

    fe_formula = 'val ~ %s'%fe
    if mod_mnoise == False:
        re_formula='(1|subj)'
    else:
        #re_formula +=' + (1|subj)'
        re_formula = '(beh|subj)'

    #only intercept effects for item because items aren't repeated
    #see http://talklab.psy.gla.ac.uk/KeepItMaximalR2.pdf page 13
    if item_inds == True:
        re_formula += ' + (1|item)'
        fact_dict ={'subj': None, 'item': None}
    else:
        fact_dict={'subj': None}

    # make list of subjects
    subjs = np.unique(ind_data['subj'])

    # instatiate result list
    all_res =  []
    res_base = {'slope':slope,
               'signal_shape':signal.shape,
               'signal_name': signal_name,
               'prop': prop,
               'run_n': run_n,
               'nsubj':nsubj,
               'nobs':nobs,
               'mnoise':mnoise,
               'inoise':I,
               'snoise':S,
               'contvar':contvar,
               'mod_mnoise':mod_mnoise,
               'mod_cont':mod_cont,
               'sn_stats':sn_stats,
               'glm_model':fe_formula,
               'nperms':nperms,
               'alpha':pthr,
               }

    # Run all the flavors of meld
    meld_run_settings = [{'method': 'meld_perm_sspn_no_mask','ss_perm_norm':True,'feat_thresh':1.0,'nperms':nperms, 'do_tfce': True},
                         {'method': 'meld_perm_sspn_no_mask_notfce','ss_perm_norm':True,'feat_thresh':1.0,'nperms':nperms, 'do_tfce': False},
                         {'method': 'meld_perm_no_mask','ss_perm_norm':False,'feat_thresh':1.0,'nperms':nperms, 'do_tfce': True},
                         {'method': 'meld_perm_no_mask_notfce','ss_perm_norm':False,'feat_thresh':1.0,'nperms':nperms, 'do_tfce': False}]
    perms = None
    for mrs in meld_run_settings:
        try:
            perms = me_s._perms
        except NameError:
            perms = mrs['nperms']
        start = time.time()
        # Run Meld
        me_s = meld.MELD(fe_formula, re_formula, 'subj',
                dep_data, ind_data, factors = fact_dict,
                use_ranks=False, re_cross_group='item',
                feat_nboot=500, feat_thresh=mrs['feat_thresh'],
                do_tfce=mrs['do_tfce'], ss_perm_norm=mrs['ss_perm_norm'],
                E=E, H=H,
                n_jobs=n_jobs)

        res_base['model'] = me_s._formula_str,
        if 'fe_flip' in mrs['method']:
            me_s.run_perms(perms)
        else:
            me_s.run_perms(mrs['nperms'])
        method_time = time.time() - start
        # Save out meld maps and error terms
        me_terms = me_s.terms
        me_t_terms = me_s.t_terms
        me_tfs = me_s.t_features
        me_pfs = me_s.get_p_features()
        me_bfs = me_s.b_features
        statmap = 1-me_pfs['beh']

        res = res_base.copy()
        res.update({'method': mrs['method'],
                    'tfce': False,
                    })
        res.update(get_metrics(statmap, 'beh', pthr, signal))
        res['tfs'] = me_tfs['beh']
        res['pfs'] = me_pfs['beh']
        res['betas'] = me_bfs['beh']
        res['tterm'] = me_s.t_terms['beh']
        res['time'] = method_time
        all_res.append(res)

        me_pfs = me_s.get_p_features(use_b = True)
        statmap = 1-me_pfs['beh']

        res = res_base.copy()
        res.update({'method': mrs['method']+'_bsig',
                    'tfce': False,
                    })
        res.update(get_metrics(statmap, 'beh', pthr, signal))
        res['tfs'] = me_tfs['beh']
        res['pfs'] = me_pfs['beh']
        res['betas'] = me_bfs['beh']
        res['tterm'] = me_s.t_terms['beh']
        res['time'] = method_time
        all_res.append(res)

    # Run GLM analysis
    print("fitting GLM", flush=True)
    glm_subj_res = eval_glm_subjs(fe_formula, ind_data, dep_data, n_jobs=n_jobs, backend=backend)
    glm_boot_res = []
    glm_boot_res.append(glm_subj_res)

    method_time = time.time() - start
    glm_boot_t_res = np.array([stats.ttest_1samp(gbr, 0, axis=0)[0] for gbr in glm_boot_res])
    glm_boot_p_res = np.array([stats.ttest_1samp(gbr, 0, axis=0)[1] for gbr in glm_boot_res])

    glm_boot_bnf_res = glm_boot_p_res[0]*10000
    glm_boot_bnf_res[glm_boot_bnf_res > 1] = 1
    statmap = 1-glm_boot_bnf_res

    res = res_base.copy()
    res.update({'method': 'glm',
                'tfce': False,
                })
    res.update(get_metrics(statmap,'beh', pthr, signal))
    res['tfs'] = glm_boot_t_res[0]
    res['pfs'] = glm_boot_p_res[0]
    res['betas'] = np.array(glm_subj_res).mean(0)
    res['time'] = method_time
    all_res.append(res)

    alpha_p = get_alphamap(glm_boot_t_res[0], glm_boot_p_res[0], 0.05, asims=asims_dd['100x100'])
    statmap = 1 - alpha_p

    res = res_base.copy()
    res.update({'method': 'glm_alphasim',
                'tfce': True,
                })
    res.update(get_metrics(statmap,'beh', pthr, signal))
    res['tfs'] = glm_boot_t_res[0]
    res['pfs'] = alpha_p
    res['time'] = method_time
    all_res.append(res)

    # Run LMER on original data
    lmer_dep_data = dep_data
    lmer_betas = []
    lmer_tvals = []
    # Run lmer with real data
    global _global_lmer
    _global_lmer = {}
    lm = LMER(fe_formula+' + ' + re_formula, ind_data)
    lm_id = id(lm)
    _global_lmer[lm_id] = lm
    flat_dep_data = lmer_dep_data.reshape(lmer_dep_data.shape[0], -1)
    res = Parallel(n_jobs=n_jobs,
            backend=backend,
            verbose=10)(delayed(run_lmer)(lm_id, flat_dep_data[:,i], ['beh'])
                        for i in np.arange(np.product(lmer_dep_data.shape[1:])))
    lmer_betas.append(np.array(res).squeeze().T.reshape(*lmer_dep_data.shape[1:])[0])
    lmer_tvals.append(np.array(res).squeeze().T.reshape(*lmer_dep_data.shape[1:])[1])

    print("Starting LMERs", flush=True)
    # Run LMER on bootstraps
    lmer_betas, lmer_tvals = Parallel(n_jobs=n_jobs, backend=backend,verbose=10)(delayed(run_lmer_perm)
               (perm, ind_data, lmer_dep_data, fe_formula+' + ' + re_formula, ['beh'])
               for perm in perms)
    lmer_betas.extend(lmer_betas)
    lmer_tvals.extend(lmer_tvals)

    del _global_lmer
    lmer_betas = np.array(lmer_betas)
    lmer_tvals = np.array(lmer_tvals)

    # get lmer feature pvalues
    nullTdist = lmer_tvals.reshape(lmer_tvals.shape[0], -1).max(1)
    nullTdist.sort()
    lmer_p = (((nperms+1)-np.searchsorted(nullTdist, lmer_tvals[0].flatten(), 'left')) /
                          (nperms+1)).reshape(lmer_tvals.shape[1:])
    statmap = 1-lmer_p

    res = res_base.copy()
    res.update({'method': 'lmer',
                'tfce': False,
                })
    res.update(get_metrics(statmap,'beh', pthr, signal[50:52,29:56]))
    res['tfs'] = lmer_tvals[0]
    res['betas'] = lmer_betas[0]
    res['pfs'] = lmer_p
    all_res.append(res)
    return all_res

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Run meld simtulations')
    parser.add_argument('outdir')
    parser.add_argument('batch_name',default = 'batch_0')
    parser.add_argument('run_n')
    parser.add_argument('signal_name')
    parser.add_argument('--nsubjs', nargs = "+")
    parser.add_argument('--nobs', nargs = "+")
    parser.add_argument('--slopes', nargs = "+")
    parser.add_argument('--Is', nargs = "+")
    parser.add_argument('--Ss', nargs = "+")
    parser.add_argument('--prop', default=0.5)
    parser.add_argument('--nboots', default = 500)
    parser.add_argument('--nperms', default = 500)
    parser.add_argument('--fvar_nboot', default = "nsubj")
    parser.add_argument('--nruns', default = 1)
    parser.add_argument('--n_jobs', default = 2)
    parser.add_argument('--backend', default = "multiprocessing")


    args = parser.parse_args()
    print(args)
    # Deal with most args
    outdir = Path(args.outdir)
    if not outdir.exists():
        outdir.mkdir()
    run_n = int(args.run_n)
    batch_name = str(args.batch_name)
    signal_name = str(args.signal_name)
    prop = float(args.prop)
    nboots = int(args.nboots)
    nperms = int(args.nperms)
    nruns = int(args.nruns)
    n_jobs = int(args.n_jobs)
    backend = str(args.backend)

    mod_mnoise = True
    if not args.slopes is  None:
        try:
            slopes = np.array(args.slopes[0].split(' ')).astype(float)
        except AttributeError:
            slopes = np.array(args.slopes).astype(float)
    else:
        slopes = np.array([1.])

    if not args.Is is  None:
        try:
            Is = np.array(args.Is[0].split(' ')).astype(float)
        except AttributeError:
            Is = np.array(args.Is).astype(float)
    else:
        Is = np.array([1.0])

    if not args.Ss is  None:
        try:
            Ss = np.array(args.Ss[0].split(' ')).astype(float)
        except AttributeError:
            Ss = np.array(args.Ss).astype(float)
    else:
        Ss = np.array([0.1])

    if not args.nsubjs is  None:
        try:
            nsubjs = np.array(args.nsubjs[0].split(' ')).astype(int)
        except AttributeError:
            nsubjs = np.array(args.nsubjs).astype(int)
    else:
        nsubjs = np.array([10]).astype(int)

    if not args.nobs is  None:
        try:
            nobses = np.array(args.nobs[0].split(' ')).astype(int)
        except AttributeError:
            nobses = np.array(args.nobs).astype(int)
    else:
        nobses = np.array([10]).astype(int)

    outname = (outdir/('run_%s_%03d_tb.npy'%(batch_name, run_n))).as_posix()
    contvar = False
    item_inds = True
    mnoise = True
    mod_mnoise = True
    nfeat = (100,100)
    feat_thresh = 0.05
    write_each = True
    
    i = 0
    j = 0

    total_runs = len(slopes)*len(Is)*len(Ss)*len(nobses)*len(nsubjs)*nruns
    start_time = time.time()
    ltime = time.localtime(start_time)
    z = 0
    #make it harder to overwrite the default outputname
    while os.path.exists(outname+'.pickle'):
        z +=1
        outname += '_%d_%d_%d_%d_%d_%d_%d'%(ltime.
            tm_year,ltime.tm_mon,ltime.tm_mday,
            ltime.tm_hour,ltime.tm_min,ltime.tm_sec,z)
    outname += '.pickle'

    sys.stdout.write("Starting %3d runs\n"%(total_runs))
    sys.stdout.flush()

    if contvar == True:
        mod_cont=True
    else:
        mod_cont=False

    #if mnoise == True:
    #    mod_mnoise=True
    #else:
    #    mod_mnoise=False

    #for signal in signals:
    results = []
    for nobs in nobses:
        for slope in slopes:
            for I in Is:
                for S in Ss:
                    for nsubj in nsubjs:
                        for run in range(nruns):
                            if args.fvar_nboot == 'nsubj':
                                fvar_nboot=nsubj
                            else:
                                fvar_nboot = int(args.fvar_nboot)

                            signal = msg.hide_blobs(cfrac=np.multiply(prop,100))
                            run_time = time.time()
                            res = test_sim_dat(nsubj,nobs,slope,signal,signal_name,run,prop,mnoise=mnoise,
                                               contvar=contvar,item_inds=item_inds,
                                               mod_mnoise=mod_mnoise,mod_cont=mod_cont,
                                               nfeat=nfeat,n_jobs=n_jobs,nboots=nboots,fvar_nboot=fvar_nboot,nperms=nperms,
                                               I=I,S=S,
                                               feat_thresh=feat_thresh,
                                               data=None, backend="multiprocessing")
                            results.extend(res)
                            if write_each:
                                with open(outname,'wb') as handle:
                                    pickle.dump(results, handle, protocol=2)
                            i += 1
                            sys.stdout.write('Run %d finished in %.2f sec\n'%(i,time.time()-run_time))
                            sys.stdout.write('%.2f sec elapsed since start\n'%(time.time()-start_time))
                            sys.stdout.flush()

                            print("Completed %3d runs of %3d total"%(i,total_runs), flush=True)
                        j += 1


    #print "AUC Preview"
    #print "MELD\tGLM\tGLM"
    #for i in range(len(results)):
    #    print "%0.2f\t%0.2f\t%0.2f"%(results[i]['meld_auc'],results[i]['glm_auc'],results[i]['glm_fdr_auc'])
    if not write_each:
        if not save_lfud:
            with open(outname,'wb') as handle:
                pickle.dump(results, handle, protocol=2)