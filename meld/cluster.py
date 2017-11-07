#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

# global imports
import numpy as np
from scipy import stats, sparse, ndimage, spatial
from scipy.sparse import csgraph
from numba import jit


cs_graph_components = csgraph.connected_components

def deg2rad(degrees):
    """Convert degrees to radians."""
    return degrees/180.*np.math.pi

def pol2cart(theta, radius, z=None, radians=True):
    """Converts corresponding angles (theta), radii, and (optional) height (z)
    from polar (or, when height is given, cylindrical) coordinates
    to Cartesian coordinates x, y, and z.
    Theta is assumed to be in radians, but will be converted
    from degrees if radians==False."""
    if radians:
        x = radius*np.cos(theta)
        y = radius*np.sin(theta)
    else:
        x = radius*np.cos(deg2rad(theta))
        y = radius*np.sin(deg2rad(theta))
    if z is not None:
        # make sure we have a copy
        z = z.copy()
        return x, y, z
    else:
        return x, y

# some functions from MNE
def _get_components(x_in, connectivity):
    """get connected components from a mask and a connectivity matrix"""
    crow = connectivity.row
    ccol = connectivity.col
    cdata = connectivity.data
    shape = connectivity.shape
    data, row, col = do_masky_things(crow, ccol, cdata, x_in)
    connectivity = sparse.coo_matrix((data, (row,col)), shape=shape)
    _, components = cs_graph_components(connectivity)
    # print "-- number of components : %d" % np.unique(components).size
    return components

@jit(nopython=True)
def do_masky_things(crow, ccol, cdata, x_in):
    mask = np.logical_and(x_in[crow], x_in[ccol])
    data = cdata[mask]
    row = crow[mask]
    col = ccol[mask]
    idx = np.where(x_in)[0]
    row = np.concatenate((row, idx))
    col = np.concatenate((col, idx))
    data = np.concatenate((data, np.ones(len(idx), dtype=data.dtype)))
    return data, row, col

def find_clusters(x, threshold, tail=0, connectivity=None):
    """For a given 1d-array (test statistic), find all clusters which
    are above/below a certain threshold. Returns a list of 2-tuples.

    Parameters
    ----------
    x: 1D array
        Data
    threshold: float
        Where to threshold the statistic
    tail : -1 | 0 | 1
        Type of comparison
    connectivity : sparse matrix in COO format
        Defines connectivity between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Defaut is None, i.e, no connectivity.

    Returns
    -------
    clusters: list of slices or list of arrays (boolean masks)
        We use slices for 1D signals and mask to multidimensional
        arrays.

    sums: array
        Sum of x values in clusters
    """
    if tail not in [-1, 0, 1]:
        raise ValueError('invalid tail parameter')

    x = np.asanyarray(x)

    if tail == -1:
        x_in = x <= threshold
    elif tail == 1:
        x_in = x >= threshold
    else:
        x_in = np.abs(x) >= threshold

    if connectivity is None:
        labels, n_labels = ndimage.label(x_in)

        if x.ndim == 1:
            clusters = ndimage.find_objects(labels, n_labels)
            sums = ndimage.measurements.sum(x, labels,
                                            index=range(1, n_labels + 1))
        else:
            clusters = list()
            sums = np.empty(n_labels)
            for l in range(1, n_labels + 1):
                c = labels == l
                clusters.append(c)
                sums[l - 1] = np.sum(x[c])
    else:
        if x.ndim > 1:
            raise Exception("Data should be 1D when using a connectivity "
                            "to define clusters.")
        if np.sum(x_in) == 0:
            return [], np.empty(0)

        components = _get_components(x_in, connectivity)
        comp_inx = components[x_in]
        comp_inx_ind = np.arange(0,len(components))[x_in]
        labels = {c:k for k,c in enumerate(np.unique(comp_inx))}
        clusters = np.zeros((len(labels), len(components)), dtype=bool)
        sums = np.zeros(len(labels))
        for i,comp in enumerate(components):
            try:
                clusters[labels[comp],i] = True
                sums[labels[comp]] += x[i]
            except KeyError:
                pass
        clusters = list(clusters)

    return clusters, sums

def find_tfce_clusters(x, threshold, sign, E, H, dt, tail=0, connectivity=None):
    """For a given 1d-array (test statistic), find all clusters which
    are above/below a certain threshold. Returns a list of 2-tuples.

    Parameters
    ----------
    x: 1D array
        Data
    threshold: float
        Where to threshold the statistic
    tail : -1 | 0 | 1
        Type of comparison
    connectivity : sparse matrix in COO format
        Defines connectivity between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Defaut is None, i.e, no connectivity.

    Returns
    -------
    clust_sums: size of cluster at each element of x
    """
    if tail not in [-1, 0, 1]:
        raise ValueError('invalid tail parameter')

    x = np.asanyarray(x)

    if tail == -1:
        x_in = x <= threshold
    elif tail == 1:
        x_in = x >= threshold
    else:
        x_in = np.abs(x) >= threshold

    if connectivity is None:
        labels, n_labels = ndimage.label(x_in)

        if x.ndim == 1:
            clusters = ndimage.find_objects(labels, n_labels)
            sums = ndimage.measurements.sum(x, labels,
                                            index=range(1, n_labels + 1))
        else:
            clusters = list()
            sums = np.empty(n_labels)
            for l in range(1, n_labels + 1):
                c = labels == l
                clusters.append(c)
                sums[l - 1] = np.sum(x[c])
    else:
        if x.ndim > 1:
            raise Exception("Data should be 1D when using a connectivity "
                            "to define clusters.")
        if np.sum(x_in) == 0:
            return np.zeros(len(x))

        components = _get_components(x_in, connectivity)
        

    return get_clust_sums(components, x_in, x, sign, E, H, dt, threshold)

@jit(nopython=True)
def get_clust_sums(components, x_in, x, sign, E, H, dt, threshold):
    comp_inx = components[x_in]
    comp_inx_ind = np.arange(0,len(components))[x_in]
    sums = np.zeros(len(components))
    clust_sums = np.zeros(len(x))
    clust_sums_inx = np.zeros(len(comp_inx))
    for i,comp in enumerate(comp_inx):
        sums[comp] += 1
    for i,comp in enumerate(comp_inx):
        clust_sums_inx[i] = sums[comp_inx[i]]
        
    clust_sums[x_in] = sign * np.power(clust_sums_inx, E) * np.power(sign*threshold, H) * dt
    
    return clust_sums


def pval_from_histogram(T, H0, tail):
    """Get p-values from stats values given an H0 distribution

    For each stat compute a p-value as percentile of its statistics
    within all statistics in surrogate data
    """
    if tail not in [-1, 0, 1]:
        raise ValueError('invalid tail parameter')

    # from pct to fraction
    if tail == -1:  # up tail
        pval = np.array([np.sum(H0 <= t) for t in T])
    elif tail == 1:  # low tail
        pval = np.array([np.sum(H0 >= t) for t in T])
    elif tail == 0:  # both tails
        pval = np.array([np.sum(H0 >= abs(t)) for t in T])
        pval += np.array([np.sum(H0 <= -abs(t)) for t in T])

    pval = (pval + 1.0) / (H0.size + 1.0)  # the init data is one resampling
    return pval


def sparse_dim_connectivity(dim_con):
    """
    Create a sparse matrix capturing the connectivity of a conjunction
    of dimensions.
    """
    # get length of each dim by looping over the connectivity matrices
    # passed in
    dlen = [d.shape[0] for d in dim_con]

    # prepare for full connectivity matrix
    nelements = np.prod(dlen)

    # get the indices
    ind = np.indices(dlen)

    # reshape them
    dind = [ind[i].reshape((nelements, 1)) for i in range(ind.shape[0])]

    # fill the rows and columns
    rows = []
    cols = []

    # loop to create mix of
    for i in range(len(dind)):
        # get the connected elements for that dimension
        r, c = np.nonzero(dim_con[i])

        # loop over them
        for j in range(len(r)):
            # extend the row/col connections
            rows.extend(np.nonzero(dind[i] == r[j])[0])
            cols.extend(np.nonzero(dind[i] == c[j])[0])

    # create the sparse connectivity matrix
    data = np.ones(len(rows))
    cmat = sparse.coo_matrix((data, (rows, cols)),
                             shape=(nelements, nelements))

    return cmat

def simple_neighbors_1d(n):
    """
    Return connectivity for simple 1D neighbors.
    """
    c = np.zeros((n, n))
    c[np.triu_indices(n, 1)] = 1
    c[np.triu_indices(n, 2)] = 0
    return c


def sensor_neighbors(sensor_locs):
    """
    Calculate the neighbor connectivity based on Delaunay
    triangulation of the sensor locations.

    sensor_locs should be the x and y values of the 2-d flattened
    sensor locs.
    """
    # see if loading from file
    if isinstance(sensor_locs, str):
        # load from file
        locs = np.loadtxt(sensor_locs)
        theta = -locs[0] + 90
        radius = locs[1]
        x, y = pol2cart(theta, radius, radians=False)
        sensor_locs = np.vstack((x, y)).T

    # get info about the sensors
    nsens = len(sensor_locs)
    
    # do the triangulation
    d = spatial.Delaunay(sensor_locs)

    # determine the neighbors
    n = [np.unique(d.vertices[np.nonzero(d.vertices == i)[0]])
         for i in range(nsens)]

    # make the symmetric connectivity matrix
    cn = np.zeros((nsens, nsens))
    for r in range(nsens):
        cn[r, n[r]] = 1

    # only keep the upper
    cn[np.tril_indices(nsens)] = 0

    # return it
    return cn


def tfce(x, dt=.1, E=2/3., H=2.0, tail=0, connectivity=None):
    """
    Threshold-Free Cluster Enhancement.
    """
    # test tail value
    if tail not in [-1, 0, 1]:
        raise ValueError('Invalid tail parameter.')

    # make sure array
    x = np.asanyarray(x)

    # figure out thresh range based on tail and the data
    trange = []
    if tail == -1:
        sign = -1.0
        if (x < 0).sum() > 0:
            trange = np.arange(x[x < 0].max(), x.min()-dt, -dt)
    elif tail == 1:
        sign = 1.0
        if (x > 0).sum() > 0:
            trange = np.arange(x[x > 0].min(), x.max()+dt, dt)
    else:
        sign = 1.0
        trange = np.arange(np.abs(x).min(), np.abs(x).max()+dt, dt)

    # make own connectivity if not provided so that we have consistent
    # return values
    if connectivity is None:
        xr = x
        connectivity = sparse_dim_connectivity([simple_neighbors_1d(n)
                                                for n in x.shape])
    else:
        # integrate in steps of dt over the threshold
        # do reshaping once
        xr = x.reshape(np.prod(x.shape))
    

    # get starting values for data (reshaped if needed)
    xt = np.zeros_like(xr)
    for thresh in trange:
        # get the clusters (reshape as necessary)
        xt += find_tfce_clusters(xr, thresh, sign, E, H, dt, 
                                     tail=tail,
                                     connectivity=connectivity)

    # return the enhanced data, reshaped back
    if connectivity is None:
        return xt
    else:
        return xt.reshape(*(x.shape))
