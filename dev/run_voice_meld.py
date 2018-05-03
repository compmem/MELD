from nilearn import image, datasets, plotting, masking
from pathlib import Path
from numba import jit

import pandas as pd
import numpy as np
import nilearn
import meld
import subprocess
import os
import argparse

def run(command, env={}, shell=False):
    merged_env = os.environ
    merged_env.update(env)
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=shell,
                               env=merged_env)
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() is not None:
            break
    if process.returncode != 0:
        raise Exception("Non zero return code: %d"%process.returncode)


def stbmeld(outdir, batch_name, run_n, subjects, stimuli, no_tfce, no_downsample, no_item, njobs, nperms, maskpath, badlist):
    """
    run meld on the single trial beta samples
    """
  
    jj = 0
    dfs = []
    dat_list = []

    mask_img = image.load_img(maskpath)
    if downsample:
        mask_rs = image.resample_to_img(mask_img, dat_list[0], interpolation='nearest')
    else:
        mask_rs = mask_img
    mask = mask_rs.get_data().flatten().astype(bool)
    connectivity = meld.cluster.sparse_dim_connectivity([meld.cluster.simple_neighbors_1d(n)
                                                   for n in mask_rs.shape])
    meta = pd.concat(dfs).reset_index(drop = True)
    masked = masking.apply_mask(dat_list, mask_rs)
    dat_ar = np.zeros((masked.shape[0], mask.shape[0]))
    dat_ar[:,mask.astype(bool)] = masked

    # Handle stimuli and subjects
    if subjects is None:
        subjects = meta.subject.unique()

    if stimuli is None:
        stimuli = meta.stimuli.unique()
    else:
        try:
            stimuli = set(stimuli[0].split(' '))
        except AttributeError:
            stimuli = set(stimuli)
    print(stimuli, flush = True)

    # Filter input
    keep_ind = (~meta['subject'].isin(bad_list)) & (meta['subject'].isin(subjects)) & (meta['stimuli'].isin(stimuli))
    meta = meta.loc[keep_ind,:].reset_index(drop = True)
    dep_data = dat_ar[keep_ind,:]
    # filter bad values in dep_data
    dep_data[dep_data > filter_limit] = filter_limit
    dep_data[dep_data < -filter_limit] = -filter_limit
    for sub_dir in sorted(proc_dir.glob('sub-*')):
        sub = sub_dir.parts[-1]
        if (sub not in bad_list) & ((subjects is None) or (sub in subjects)):
            print(sub, end = ', ', flush = True)
            subn = sub_dir.parts[-1][4:]
            stb_dir = sub_dir/'passivelistening/stb/'
            tmp_df = pd.read_csv('task-passivelistening_events.tsv', sep = '\t')
            tmp_df.stimuli = tmp_df.stimuli.str.split('/').str[-1]
            tmp_df['subject'] = sub
            for ii in tmp_df.index:
                ii_img = image.load_img((stb_dir/"REML_{stim}_{n}.nii.gz".format(stim = '_'.join(tmp_df.loc[ii,'stimuli'].split('_')[0:-1]), n = int(tmp_df.loc[ii,'stimuli'].split('_')[-1]))).as_posix())
                if downsample:
                    ii_rs = image.resample_img(ii_img,target_affine= np.diag((4,4,4)))
                else:
                    ii_rs = ii_img
                try:
                    dat_list.append(ii_rs)
                except:
                    bad_list.append(subj)
                    break
                jj +=1
                #print(ii, end = ' ')
            dfs.append(tmp_df)
            #print()
    print("downsampling is", downsample, flush = True)
    if not subjects is  None:
        try:
            subjects = set(subjects[0].split(' '))
        except AttributeError:
            subjects = set(subjects)
    else:
        subjects = {}
    print(subjects, subjects, flush = True)

    base_dir = Path('/meld_root/work/data/voice/')
    proc_dir = base_dir/'derivatives/afni_proc2'

    meta = pd.concat(dfs).reset_index(drop = True)
    masked = masking.apply_mask(dat_list, mask_rs)
    dat_ar = np.zeros((masked.shape[0], mask.shape[0]))
    dat_ar[:,mask.astype(bool)] = masked

    # Filter input
    keep_ind = (~meta['subject'].isin(bad_list)) & (meta['subject'].isin(subjects)) & (meta['stimuli'].isin(stimuli))
    meta = meta.loc[keep_ind,:].reset_index(drop = True)
    dep_data = dat_ar[keep_ind,:]
    
    print("ind_data shape: ",ind_data.shape," dep_data_shape: ", dep_data.shape," connectivity shape: ", connectivity.shape, " mask_shape: ", mask.shape, " mask_sum ", mask.sum(), flush = True)
    # Run MELD
    me_s = meld.meld.MELD('val ~ vocal', re_formula, 'subj',
                    dep_data, ind_data, factors=factors,
                    use_ranks=False,
                    feat_nboot=500, feat_thresh=0.05,
                    dep_mask = mask,
                    do_tfce=do_tfce,
                    connectivity=connectivity, #shape=img_dat.shape,
                    dt=.01, E=2/3., H=2.0,
                    n_jobs=njobs, verbose=10,
                    memmap=False
                   )

    me_s.run_perms(nperms)


    # Write out results

    pfmasks = np.array(me_s._pfmask).transpose((1, 0, 2))
    t = me_s.get_t_features()
    t = np.array(t['vocal']).reshape(dat_list[0].shape)
    timg = image.new_img_like(dat_list[0], t)
    timg.to_filename((outdir/('run_%s_%03d_t.nii.gz'%(batch_name, run_n))).as_posix())
    p = me_s.p_features
    p = np.array(me_s.p_features['vocal']).reshape(dat_list[0].shape)
    p_img = image.new_img_like(dat_list[0], t*(p<0.05))
    p_img.to_filename((outdir/('run_%s_%03d_p.nii.gz'%(batch_name, run_n))).as_posix())
    t_thresh_img = image.new_img_like(dat_list[0], t*(p<0.05))
    t_thresh_img.to_filename((outdir/('run_%s_%03d_t_thresh.nii.gz'%(batch_name, run_n))).as_posix())

    np.save((outdir/('run_%s_%03d_tb.npy'%(batch_name, run_n))).as_posix(), np.concatenate(me_s._tb))
    np.save((outdir/('run_%s_%03d_pfmask.npy'%(batch_name, run_n))).as_posix(), np.concatenate(me_s._pfmask))


if __name__ == "__main__":
    """
    Initialize the script with the arguments.
    """

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Run meld on some set of subjects and stims')
    parser.add_argument('--outdir', default = "sample_output", help="output directory")
    parser.add_argument('--batch_name', default = 'batch_0', help="batch name")
    parser.add_argument('--run_n', help="number of runs")
    parser.add_argument('--subjects', nargs = "+", help="list of subjects to use")
    parser.add_argument('--stimuli', nargs = "+", help="list of stimuli input")
    parser.add_argument('--no_tfce', action='store_true', help="whether to use tfce")
    parser.add_argument('--no_downsample', action='store_true', help="whether to use downsample")
    parser.add_argument('--njobs', default = 10)
    parser.add_argument('--nperms', default = 100)
    parser.add_argument('--maskpath', help="path to the mask file")
    parser.add_argument('--badlist', default = ['sub-007', 'sub-023', 'sub-072', 'sub-077', 'sub-137', 'sub-165', 'sub-045'])
    parser.add_argument('--filterlimit', default = 100)
    parser.add_argument('--formula', default="val ~ 1", help = "formula to use for predicting")
    
    args = parser.parse_args()
  

    outdir = Path(args.outdir)
    if not outdir.exists():
        outdir.mkdir()
    
    # Input the arguments to variables
    run_n = int(args.run_n)
    batch_name = str(args.batch_name)
    do_tfce = not bool(args.no_tfce)
    downsample = not bool(args.no_downsample)
    njobs = int(args.njobs)
    nperms = int(args.nperms)
    subjects = args.subjects
    bad_list = args.badlist.split(" ")
    filter_limit = int(args.filterlimit)
    maskpath = args.maskpath
    stimuli = args.stimuli
