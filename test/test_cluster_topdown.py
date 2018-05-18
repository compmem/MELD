import meld
import numpy as np

d = 20
pixelSize = 1
E = 2/3
H = 2
test = np.random.rand(d,d,d)*5
#test_conn = meld.cluster.sparse_dim_connectivity([meld.cluster.simple_neighbors_1d(n)
#                                                  for n in test.shape])
colData = test.flatten()
areaData = pixelSize * np.ones_like(colData)
test_conn = meld.cluster.sparse_dim_connectivity([meld.cluster.simple_neighbors_1d(n, triu=False)
                                                  for n in test.shape], row_list=True)

res = meld.cluster_topdown.tfce_pos(colData, pixelSize, test_conn, param_e = E, param_h = H)
