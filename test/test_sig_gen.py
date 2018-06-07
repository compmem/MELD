import numpy as np
import pandas as pd
from os import path
from meld import signal_gen as msg

def test_hide_blobs():
    st1 = msg.hide_blobs((10,10))
    # Make sure hide blobs only generates 0s and 1s
    assert (st1[st1 != 0] == 1).all()
    assert (st1 == np.array([[ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [ 0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [ 1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.],
                             [ 0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                             [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])).all()
    cfrac_test_path =path.abspath(path.dirname(__file__))+'/cfrac_test.csv'
    cfrac_test = np.loadtxt(cfrac_test_path, delimiter=',').reshape((100,100))
    cfrac_try = msg.hide_blobs(cfrac=50)
    # Make sure hide blobs only generates 0s and 1s
    assert (cfrac_try[cfrac_try != 0] == 1).all()
    # Check the cfrac pattern against a reference
    assert (cfrac_try == cfrac_test).all()

def test_gen_data():
    for test_n in range(20):
        ind_data, dep_data, sn_stats = msg.gen_data(8,50,(1,1),0.5,np.array([[1]]), item_inds=True,I=1,S=0.1)
        dat_df = pd.DataFrame(ind_data)
        dat_df['dep'] = dep_data.squeeze()
        # Ensure that items are consistently assigned to a single behavioral group
        assert np.array([len(val)==1 for val in dat_df.groupby('item').beh.unique()]).all()

        signal = msg.hide_blobs(cfrac=50)
        # Create data with a huge signal in it to make sure that it's getting hidden correctly
        ind_data, dep_data, sn_stats = msg.gen_data(8,50,signal.shape,100,signal, item_inds=True,I=1,S=0.1)
        # Make sure all the positive behavioral trials have large positive values in signal elements
        assert (dep_data[ind_data['beh'] > 0][:,signal==1] > 30).all()
        # Make sure all the negative behavioral trials have large negative values in signal elements
        assert (dep_data[ind_data['beh'] < 0][:,signal==1] < -30).all()
        # Make sure all the nonsignal elements have small values
        assert (np.abs(dep_data[:,signal==0])<30).all()

        ind_data, dep_data, sn_stats = msg.gen_data(8,100,(1,1),1,np.array([[1]]), item_inds=True,I=100,S=0.0, field_noise=0)
        dat_df = pd.DataFrame(ind_data)
        dat_df['dep'] = dep_data.squeeze()
        # Ensure that standard deviation within behavioral effects is high
        assert (dat_df.groupby('beh').dep.describe()['std']>30).all()
        # but that each item is consistent (since we took out all otther noise)
        assert (dat_df.groupby('item').dep.describe()['std']==0).all()

        ind_data, dep_data, sn_stats = msg.gen_data(100,8,(1,1),1,np.array([[1]]), item_inds=True,I=0.0,S=100.0, field_noise=0)
        dat_df = pd.DataFrame(ind_data)
        dat_df['dep'] = dep_data.squeeze()
        # Ensure that standard deviation within behavioral effects is high
        assert (dat_df.groupby('beh').dep.describe()['std']>30).all()
        # but that each item is consistent (since we took out all otther noise)
        assert (dat_df.groupby(['subj','beh']).dep.describe()['std'] == 0).all()

        ind_data, dep_data, sn_stats = msg.gen_data(8,100,(1,1),1,np.array([[1]]), item_inds=True,I=1,S=1,mnoise=False, field_noise=0)
        dat_df = pd.DataFrame(ind_data)
        dat_df['dep'] = dep_data.squeeze()
        # make sure that slope noise is absent when it should be absent
        test_slope_noise = (dat_df.query('beh == 0.5').groupby('subj').dep.mean() - dat_df.query('beh == -0.5').groupby('subj').dep.mean()).std()
        assert test_slope_noise < 1e-12

        ind_data, dep_data, sn_stats = msg.gen_data(1000,10,(1,1),1,np.array([[1]]), item_inds=True,I=0,S=1,mnoise=True, field_noise=0)
        dat_df = pd.DataFrame(ind_data)
        dat_df['dep'] = dep_data.squeeze()
        # and present when it should be present
        test_slope_noise = (dat_df.query('beh == 0.5').groupby('subj').dep.mean() - dat_df.query('beh == -0.5').groupby('subj').dep.mean()).std()
        assert test_slope_noise > 0.9 and test_slope_noise < 1.1

