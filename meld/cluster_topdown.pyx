from __future__ import print_function
#cython: linetrace=True
#cython: binding=True

import numpy as np
import heapq
from .cluster import sparse_dim_connectivity
from numba import float64, jitclass, jit
from libc.math cimport pow
from cpython cimport array
import array

cimport cython


#@jit(nopython=True)
def tfce_calc(extent, param_e, upper_limit, h_plus1, lower_limit):

    return np.power(extent, param_e) * \
                (np.power(upper_limit, h_plus1)
                 - np.power(lower_limit, h_plus1)) / h_plus1


cdef array.array long_array_template = array.array('l', [])
cdef array.array double_array_template = array.array('d', [])

cdef class Cluster:
    """ TFCE cluster class
    """
    cdef public float param_e, h_plus1, accum_val, extent, upper_limit, lower_limit, new_slice_val
    cdef public long n_mems
    cdef public array.array members, node_ar

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, float param_e, float param_h):
        """
            Initialize empty cluster
        """
        self.param_e = param_e
        self.h_plus1 = param_h + 1
        # summed TFCE over all cluster members
        self.accum_val = 0.
        # extent of cluster; called "totalArea" in Cpp implementation
        self.extent = 0.
        # list of cluster members
        self.n_mems = 0
        self.members = array.clone(long_array_template, 0, zero=False)
        self.node_ar = array.clone(long_array_template, 1, zero=False)

        # upper limit of TFCE integral; called "lastVal" in Cpp implementation
        self.upper_limit = 0.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update(self, float lower_limit):
        """
            Perform an incremental update to the summed TFCE integral over
            cluster members

            Args
            ----
            lower_limit : float
                lower limit of TFCE integral
                called "bottomVal" in Cpp implementation
        """
        cdef float new_slice_val
        if self.n_mems == 0:
            self.upper_limit = lower_limit
        elif not lower_limit == self.upper_limit:
            assert lower_limit < self.upper_limit
            new_slice_val = pow(self.extent, self.param_e) * (pow(self.upper_limit, self.h_plus1) - pow(lower_limit, self.h_plus1)) / self.h_plus1
            self.accum_val += new_slice_val
            self.upper_limit = lower_limit

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void add_member(self, long node, float height, float extent):
        """
            Add a new node to cluster

            Args
            ----
            node : int
                Node index
            height : float
                Test statistic value at node
                called "val" in Cpp implementation
            extent : float
                Extent of node
                called "area" in Cpp implementation
        """
        self.node_ar[0] = node
        self.update(height)
        # append node to list of members
        array.extend(self.members, self.node_ar)
        self.n_mems += 1
        self.extent += extent

    cpdef void clear_members(self):
        """Remove all members from cluster
        """
        array.resize(self.members, 0)
        self.n_mems = 0


def alloc_cluster(cluster_list, dead_clusters, param_e, param_h):
    """
        Generate a valid (new) cluster number
        If available, number for a dead cluster will be recycled
        Cluster and dead cluster lists will be updated

        Args
        ----
            cluster_list : list of Clusters
            dead_clusters : set of Clusters
            param_e : float
            param_h : float
    """
    # verify that the lists are actually being modified in main code
    if len(dead_clusters) == 0:
        cluster_list.append(Cluster(param_e, param_h))
        return len(cluster_list) - 1
    else:
        # removes and returns element from set
        ret = dead_clusters.pop()
        cluster_list[ret] = Cluster(param_e, param_h)
        return ret


def get_node_neighbors(node, connectivity):
    """
        Given a connectivity matrix, get neighbors of a node

        Args
        ----
            node : int
                Node index
            connectivity : sparse matrix in COO format

        Returns
        -------
            neighbors : numpy array of integers
                array of node indices for neighboring nodes

    """
    # since connectivity matrix is triangular, we need to check both along rows
    # and cols
    # diagonals are all 0 in connectivity matrix (no self-neighboring) so we
    # don't need to worry about that
    neighbors = connectivity[node]

    return neighbors


#@jit(nopython=True)
cdef get_touching(long [:] neighbors, long [:] membership):
    """ Numba function to get cluster membership of neighboring clusters

    Paramters
    ---------
    neighbors : array of int
        Node indicies of neighboring nodes
    membership : array of float
        Array storing membership for every node

    Returns
    -------
    touching_clusters: set of int
        Set of node indicies for touching clusters
    """
    #cdef long nbr
    cdef long n_nbrs = len(neighbors)
    cdef long num_touching = 0
    cdef long mbr
    cdef int i
    
    cdef set touching_clusters = set()
    for i in range(n_nbrs):
        mbr = membership[neighbors[i]]
        if mbr != -1:
            if mbr not in touching_clusters:
                touching_clusters.add(mbr)
                num_touching += 1
    return (touching_clusters, num_touching)

cpdef write_test():
    print("written")
    print("docker build only")

cpdef tfce_pos(col_data, double area, connectivity, double param_e=0.5, double param_h=2.0):
    """
        Compute TFCE for each node (i.e., vertex, pixel, voxel, ...)

        TFCE at node p is given by
            \integral_{0}^{h_p} e(h,p)^E h^H dh ,
        where h is the test statistic, h_p is the test statistic at node p,
        e(h,p) is the extent of the cluster (i.e., the area/volume of all nodes
        that are part of the same connected component with node p when
        thresholded at h), and E and H are parameters.

        This implementation capitalizes on the discrete nature of test statistic
        data in the image.

        Sort the test statistics for the n nodes in the image in descending
        order such that:
        h_{(0)} >= h_{(1)} >= ... >= h{(n)}

        The TFCE integral can be written as a sum of integrals by splitting the
        integration limits to multiple intervals. Thus, the TFCE at node p_{(0)}
        with the maximum test statistic h_{(0)} is given by
            \integral_{h_{(1)}}^{h_{(0)}}   e(h,p_{(0)})^E h^H dh +
            \integral_{h_{(2)}}^{h_{(1)}}   e(h,p_{(0)})^E h^H dh +
            ... +
            \integral_{h_{(n)}}^{h_{(n-1)}} e(h,p_{(0)})^E h^H dh,
        where h_{(n)} is assumed to be 0 (if not, another integral is needed
        to capture this bottom layer).
        For each of these integrals, e(h,p_{(0)}) is a constant given the range
        of h in the integral, and therefore can be moved outside the integral,
        allowing for the integral to be evaluated analytically. Let the value
        of e(h,p_{(0)}) between h_{(1)} < h <= h_{(0)} be e(h_{(0)}, p_{(0)}).
        Then, TFCE at node p_{(0)} is given by:
            e(h_{(0)}, p_{(0)}) \frac{ h_{(0)}^{H+1} - h_{(1)}^{H+1} }{ H+1 } +
            e(h_{(1)}, p_{(0)}) \frac{ h_{(1)}^{H+1} - h_{(2)}^{H+1} }{ H+1 } +
            ... +
            e(h_{(n-1)}, p_{(0)}) \frac{ h_{(n-1)}^{H+1} - h_{(n)}^{H+1} }{ H+1 }.


        Args
        ----
            col_data : 1-D numpy array of floats
            area_data : 1-D numpy array of floats
            param_e : float
            param_h : float

        Returns
        -------
            accum_data : 1-D numpy array of floats
                array of TFCE values at each node
    """
    # col_data must be n-by-1
    # area_data must be n-by-1
    # connectivity must be a sparse n-by-n matrix; upper triangular only

    #num_nodes = len(col_data)
    
    
    cdef long biggest_size = 0
    cdef long merged_indes = -1
    cdef long tclust, node, mbr, #which_cluster
    cdef long n_cd_vals = len(col_data)
    cdef double correction_val, value, negvalue, num_touching
    cdef long [:] neighbors
    accum_data =  np.zeros_like(col_data)
    cdef double [:] accum_data_view = accum_data
    cdef set touching_clusters, dead_clusters
    
    membership = -np.ones_like(col_data, dtype=np.intp)
    cdef long [:] membership_view = membership
    #accum_data = np.zeros_like(col_data)
    cluster_list = []
    dead_clusters = set()
    node_heap = []

    for i, coldt in enumerate(col_data):
        if coldt > 0:
            # first element is the priority (determines heap order)
            heapq.heappush(node_heap, (-coldt, i))

    while len(node_heap) > 0:
        negvalue, node = heapq.heappop(node_heap)
        value = -negvalue
        neighbors = connectivity[node]
        #num_neigh = len(neighbors)

        touching_clusters, num_touching = get_touching(neighbors, membership_view)

        #num_touching = len(touching_clusters)
        # make new cluster
        if num_touching == 0:
            new_cluster = alloc_cluster(cluster_list,
                                        dead_clusters,
                                        param_e,
                                        param_h)
            cluster_list[new_cluster].add_member(node, value, area)
            membership_view[node] = new_cluster
        # add to cluster
        elif num_touching == 1:
            which_cluster = touching_clusters.pop()
            cluster_list[which_cluster].add_member(node, value, area)
            membership_view[node] = which_cluster
            accum_data_view[node] -= cluster_list[which_cluster].accum_val
        # merge all touching clusters
        else:
            # find the biggest cluster (i.e., with most members)
            #  and use as merged cluster
            merged_index = -1
            biggest_size = 0

            for tclust in touching_clusters:
                if cluster_list[tclust].n_mems > biggest_size:
                    merged_index = tclust
                    biggest_size = cluster_list[tclust].n_mems

            # assert vector index .. ?
            assert (merged_index >= 0) and (merged_index < len(cluster_list))

            merged_cluster = cluster_list[merged_index]
            # recalculate to align cluster bottoms
            merged_cluster.update(value)

            for tclust in touching_clusters:
                # if we are the largest cluster, don't modify the per-node accum
                # for members, so merges between small and large clusters are cheap
                if tclust != merged_index:
                    this_cluster = cluster_list[tclust]
                    # recalculate to align cluster bottoms
                    this_cluster.update(value)

                    correction_val = this_cluster.accum_val - merged_cluster.accum_val
                    for mbr in this_cluster.members:
                        accum_data_view[mbr] += correction_val
                        membership_view[mbr] = merged_index

                    merged_cluster.members.extend(this_cluster.members)
                    merged_cluster.extent += this_cluster.extent
                    # designate (old) cluster as dead
                    dead_clusters.add(tclust)
                    # deallocate member list
                    cluster_list[tclust].clear_members()

            # will not trigger recomputation; we already recomputed at this value
            merged_cluster.add_member(node, value, area)

            # the node they merge on must not get the peak value of the cluster,
            # so again, record its difference from peak
            accum_data_view[node] -= merged_cluster.accum_val
            membership_view[node] = merged_index

            # do not reset the accum value of the merged cluster,
            # we specifically avoided modifying the per-node accum for its
            # members, so the cluster accum is still in play

    # final clean up of accum data
    for i, this_cluster in enumerate(cluster_list):
        # ignore clusters that don't exist
        if i not in dead_clusters:
            # update to include the to-zero slice
            this_cluster.update(0.0)
            for mbr in this_cluster.members:
                # add the resulting slice to all members -
                # their stored data contains the offset between
                # the cluster peak and their corect value
                accum_data_view[mbr] += this_cluster.accum_val

    return accum_data
