import nilearn
from nilearn import image, datasets, plotting, masking
from pathlib import Path
import pandas as pd
import numpy as np
import meld
from matplotlib import pyplot as plt
from numba import jit
from pathlib import Path
import subprocess
import os
import argparse


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
pd.set_option('display.max_columns', 500)
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

        
        
parser = argparse.ArgumentParser(description='Run meld on some set of subjects and stims')
parser.add_argument('outdir')
parser.add_argument('batch_name',default = 'batch_0')
parser.add_argument('run_n')
parser.add_argument('--subjects', nargs = "+")
parser.add_argument('--stimuli', nargs = "+")
parser.add_argument('--no_tfce', action='store_true')
parser.add_argument('--no_downsample',action='store_true')
parser.add_argument('--no_item',action='store_true')
parser.add_argument('--njobs', default = 10)
parser.add_argument('--nperms', default = 100)

args = parser.parse_args()
print(args)
# Deal with most args
outdir = Path(args.outdir)
if not outdir.exists():
    outdir.mkdir()
run_n = int(args.run_n)
batch_name = str(args.batch_name)
do_tfce = not bool(args.no_tfce)
downsample = not bool(args.no_downsample)
item_re = not bool(args.no_item)
njobs = int(args.njobs)
nperms = int(args.nperms)

print("downsampling is", downsample, flush = True)
if not args.subjects is  None:
    try:
        subjects = set(args.subjects[0].split(' '))
    except AttributeError:
        subjects = set(args.subjects)
else:
    subjects = {}
print(args.subjects, subjects, flush = True)

base_dir = Path('/meld_root/work/data/voice/')
proc_dir = base_dir/'derivatives/afni_proc2'
bad_list = ['sub-007', 'sub-023', 'sub-072', 'sub-077', 'sub-137', 'sub-165', 'sub-045']

jj = 0
dfs = []
dat_list = []
for sub_dir in sorted(proc_dir.glob('sub-*')):
    sub = sub_dir.parts[-1]
    if (sub not in bad_list) & ((args.subjects is None) or (sub in subjects)):
        print(sub, end = ', ', flush = True)
        subn = sub_dir.parts[-1][4:]
        stb_dir = sub_dir/'passivelistening/stb/'
        tmp_df = pd.read_csv('/meld_root/work/data/voice/task-passivelistening_events.tsv', sep = '\t')
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
        
mask_path = '/meld_root/work/data/voice/derivatives/afni_proc2/group_full_mask.nii.gz'
mask_img = image.load_img(mask_path)
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
if args.subjects is None:
    subjects = meta.subject.unique()    
        
if args.stimuli is None:
    stimuli = meta.stimuli.unique()
else:
    try:
        stimuli = set(args.stimuli[0].split(' '))
    except AttributeError:
        stimuli = set(args.stimuli)
print(stimuli, flush = True)

# Filter input
keep_ind = (~meta['subject'].isin(bad_list)) & (meta['subject'].isin(subjects)) & (meta['stimuli'].isin(stimuli))
meta = meta.loc[keep_ind,:].reset_index(drop = True)
dep_data = dat_ar[keep_ind,:]
# filter bad values in dep_data
dep_data[dep_data > 100] = 100.
dep_data[dep_data < -100] = -100.

# Make ind_data rec array and rename columns
meta['vocal'] = 0
meta.loc[meta.trial_type == 'nonvocal', 'vocal'] = -1
ind_data = meta.loc[:, ['subject','stimuli','vocal']]
ind_data.columns = ['subj', 'item', 'vocal']
ind_data['val'] = 0
print(ind_data.item.unique(), flush = True)
print(ind_data.subj.unique(), flush = True)
ind_data = ind_data.to_records(index = False)

if item_re:
    re_formula = '(vocal|subj) + (1|item)'
    factors = factors={'subj': None, 'item': None}
else:
    re_formula = '(vocal|subj)'
    factors={'subj': None}
    
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