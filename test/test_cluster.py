import numpy as np
import meld

def test_find_clusters():
    test = np.zeros((6,6))
    for i in range(2):
        for j in range (2):
            test[i*3:(i*3)+2,j*3:(j*3)+2] = 0.1 + i * 0.2 + j * 0.2

    test_conn = meld.cluster.sparse_dim_connectivity([meld.cluster.simple_neighbors_1d(n)
                                                   for n in test.shape])     

    conn_res = {}
    conn_res[(0.1,1)] = (([np.array([ True,  True, False, False, False, False,  True,  True, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False], dtype=bool),
      np.array([False, False, False,  True,  True, False, False, False, False,
              True,  True, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False], dtype=bool),
      np.array([False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
              True,  True, False, False, False, False,  True,  True, False,
             False, False, False, False, False, False, False, False, False], dtype=bool),
      np.array([False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False,  True,  True, False, False, False, False,
              True,  True, False, False, False, False, False, False, False], dtype=bool)],
     np.array([ 0.4,  1.2,  1.2,  2. ])))
    conn_res[(0.1,-1)] = ([np.array([ True,  True,  True, False, False,  True,  True,  True,  True,
             False, False,  True,  True,  True,  True,  True,  True,  True,
             False, False,  True, False, False,  True, False, False,  True,
             False, False,  True,  True,  True,  True,  True,  True,  True], dtype=bool)],
     np.array([ 0.4]))
    conn_res[(0.3,0)] = ([np.array([False, False, False,  True,  True, False, False, False, False,
              True,  True, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False], dtype=bool),
      np.array([False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
              True,  True, False, False, False, False,  True,  True, False,
             False, False, False, False, False, False, False, False, False], dtype=bool),
      np.array([False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False,  True,  True, False, False, False, False,
              True,  True, False, False, False, False, False, False, False], dtype=bool)],
     np.array([ 1.2,  1.2,  2. ]))


    for thresh, tail in conn_res.keys():
        clust_res, sums_res = meld.cluster.find_clusters(test.flatten(),thresh,tail,test_conn)
        assert (np.array(conn_res[(thresh,tail)][0]) == np.array(clust_res)).all()
        assert np.allclose(conn_res[(thresh,tail)][1],sums_res)

        clust_res, sums_res = meld.cluster.find_clusters(test,thresh,tail)
        assert (np.array(conn_res[(thresh,tail)][0]).flatten() == np.array(clust_res).flatten()).all()
        assert np.allclose(conn_res[(thresh,tail)][1],sums_res)

def test_tfce():    
    test = np.zeros((6,6))
    for i in range(2):
        for j in range (2):
            test[i*3:(i*3)+2,j*3:(j*3)+2] = 0.1 + i * 0.2 + j * 0.2

    test_conn = meld.cluster.sparse_dim_connectivity([meld.cluster.simple_neighbors_1d(n)
                                                   for n in test.shape])
    test[0:2,3:5] *= -1

    tests = []
    tests.append({'dt': 0.1, 'e': 2/3, 'h': 2, 'tail': 0, 'result':np.array([ 0.00251984,  0.00251984,  0.        ,  0.03527779,  0.03527779,
            0.        ,  0.00251984,  0.00251984,  0.        ,  0.03527779,
            0.03527779,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.03527779,  0.03527779,
            0.        ,  0.13859132,  0.13859132,  0.        ,  0.03527779,
            0.03527779,  0.        ,  0.13859132,  0.13859132,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ])})
    tests.append({'dt': 0.1, 'e': 2/3, 'h': 2, 'tail': -1, 'result':np.array([ 0.        ,  0.        ,  0.        , -0.02267858, -0.02267858,
            0.        ,  0.        ,  0.        ,  0.        , -0.02267858,
           -0.02267858,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ])})
    tests.append({'dt': 0.1, 'e': 2/3, 'h': 2, 'tail': 1, 'result':np.array([ 0.00251984,  0.00251984,  0.        ,  0.        ,  0.        ,
            0.        ,  0.00251984,  0.00251984,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.03527779,  0.03527779,
            0.        ,  0.13859132,  0.13859132,  0.        ,  0.03527779,
            0.03527779,  0.        ,  0.13859132,  0.13859132,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ])})
    tests.append({'dt': 0.1, 'e': 2/3, 'h': 3, 'tail': 0, 'result':np.array([ 0.00025198,  0.00025198,  0.        ,  0.00907143,  0.00907143,
            0.        ,  0.00025198,  0.00025198,  0.        ,  0.00907143,
            0.00907143,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.00907143,  0.00907143,
            0.        ,  0.05669645,  0.05669645,  0.        ,  0.00907143,
            0.00907143,  0.        ,  0.05669645,  0.05669645,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ])})
    tests.append({'dt': 0.1, 'e': 2/3, 'h': 1, 'tail': 0, 'result':np.array([ 0.02519842,  0.02519842,  0.        ,  0.15119053,  0.15119053,
            0.        ,  0.02519842,  0.02519842,  0.        ,  0.15119053,
            0.15119053,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.15119053,  0.15119053,
            0.        ,  0.37797631,  0.37797631,  0.        ,  0.15119053,
            0.15119053,  0.        ,  0.37797631,  0.37797631,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ])})
    tests.append({'dt': 0.1, 'e': 1/3, 'h': 2, 'tail': 0, 'result':np.array([ 0.0015874 ,  0.0015874 ,  0.        ,  0.02222361,  0.02222361,
            0.        ,  0.0015874 ,  0.0015874 ,  0.        ,  0.02222361,
            0.02222361,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.02222361,  0.02222361,
            0.        ,  0.08730706,  0.08730706,  0.        ,  0.02222361,
            0.02222361,  0.        ,  0.08730706,  0.08730706,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ])})
    tests.append({'dt': 0.1, 'e': 1/3, 'h': 2, 'tail': 0, 'result':np.array([ 0.0015874 ,  0.0015874 ,  0.        ,  0.02222361,  0.02222361,
            0.        ,  0.0015874 ,  0.0015874 ,  0.        ,  0.02222361,
            0.02222361,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.02222361,  0.02222361,
            0.        ,  0.08730706,  0.08730706,  0.        ,  0.02222361,
            0.02222361,  0.        ,  0.08730706,  0.08730706,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ])})
    tests.append({'dt': 0.1, 'e': 4/5, 'h': 2, 'tail': 0, 'result':np.array([ 0.00303143,  0.00303143,  0.        ,  0.04244006,  0.04244006,
            0.        ,  0.00303143,  0.00303143,  0.        ,  0.04244006,
            0.04244006,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.04244006,  0.04244006,
            0.        ,  0.16672882,  0.16672882,  0.        ,  0.04244006,
            0.04244006,  0.        ,  0.16672882,  0.16672882,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ])})
    tests.append({'dt': 0.2, 'e': 2/3, 'h': 2, 'tail': 0, 'result':np.array([ 0.        ,  0.        ,  0.        ,  0.02015874,  0.02015874,
            0.        ,  0.        ,  0.        ,  0.        ,  0.02015874,
            0.02015874,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.02015874,  0.02015874,
            0.        ,  0.10079368,  0.10079368,  0.        ,  0.02015874,
            0.02015874,  0.        ,  0.10079368,  0.10079368,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ])})
    tests.append({'dt': 0.05, 'e': 2/3, 'h': 2, 'tail': 0, 'result':np.array([ 0.0015749,  0.0015749,  0.       ,  0.0286632,  0.0286632,
            0.       ,  0.0015749,  0.0015749,  0.       ,  0.0286632,
            0.0286632,  0.       ,  0.       ,  0.       ,  0.       ,
            0.       ,  0.       ,  0.       ,  0.0286632,  0.0286632,
            0.       ,  0.1212674,  0.1212674,  0.       ,  0.0286632,
            0.0286632,  0.       ,  0.1212674,  0.1212674,  0.       ,
            0.       ,  0.       ,  0.       ,  0.       ,  0.       ,  0.       ])})
    for t in tests:
        tres = meld.cluster.tfce(test.flatten(),dt=t['dt'], E=t['e'], H=t['h'],tail=t['tail'], connectivity = test_conn)
        assert np.allclose(tres,t['result'])