# distutils: language=c++
# distutils: libraries = m
from meld.ccluster cimport Cluster, VoxelIJK, allocCluster
from meld.caret_heap cimport CaretSimpleMaxHeap
from libcpp.vector cimport vector
from libcpp.set cimport set
from libc.stdint cimport int64_t
from libc.stdio cimport printf
cimport numpy as np
from libcpp cimport bool
import numpy as np
cimport cython
from cython.operator cimport dereference as deref

ctypedef np.npy_intp SIZE_t 
ctypedef vector[int64_t] int_vec

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bool index_valid(int64_t[4]& vind, int64_t[4]& dims):
    """Check if a given set of indices is valid in
       a dataset of the provided dimensions.
    """
    if ((vind[0] < 0) | (vind[0] >= dims[0])):
        return False
    if (vind[1] < 0) | (vind[1] >= dims[1]):
        return False
    if ((vind[2] < 0) | (vind[2] >= dims[2])):
        return False
    if ((vind[3] < 0) | (vind[3] >= dims[3])):
        return False
    return True

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void four_d_TFCE(float [:,:,:,::1] data_view, float [::1] accum_data_view, float volume, float param_e=0.666666666, float param_h=2.0, bint negate = False, bint pad = False):
    
    # define all the variables to keep cython happy
    cdef int64_t sx = data_view.shape[0]
    cdef int64_t sy = data_view.shape[1]
    cdef int64_t sz = data_view.shape[2]
    cdef int64_t st = data_view.shape[3]
    cdef int64_t size = sx * sy * sz * st
    cdef int64_t[4] dims = [sx, sy, sz, st]
    cdef int64_t dim_start = 0
    # set the dimensions so we avoid looping over the pad elements
    if pad:
        dim_start = 1
    cdef int64_t[3] m_mult = [dims[1]*dims[2]*dims[3], dims[2]*dims[3], dims[3]]
    if pad:
         dims = [sx - 1, sy - 1, sz - 1, st - 1]
    cdef int_vec membership = int_vec(size, -1)
    # Not preallocating the list of clusters takes advantage of c++ relatively optimized vector reallocation
    # Preallocating is much less efficient. In otherwords, this is already optimized, don't mess with it
    cdef vector[Cluster] cluster_list
    cdef CaretSimpleMaxHeap[VoxelIJK, float] voxel_heap
    cdef size_t i, j, k, t
    cdef VoxelIJK neigh_voxel, voxel
    cdef float value
    cdef double correction_val
    cdef int64_t voxel_index, neigh_index, which_cluster, num_touching, new_cluster
    cdef int64_t merged_index, biggest_size, tclust, num_members, member_index
    cdef set[int64_t] dead_clusters, touching_clusters
    
    # Define the index offsets for spatial neighbors
    # Currently only face neighbors are implemented
    cdef int64_t STENCIL_SIZE = 32
    cdef int_vec stencil = int_vec(STENCIL_SIZE)
    stencil = [0, 0, 0, -1,
               0, 0, -1, 0,
               0, -1, 0, 0,
               -1, 0, 0, 0,
               1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1]
    
    # Make the negative values positive if negate is true
    # This code duplication moves the if statement out of the inner loop
    # Saves a few percent in run time
    if negate:
        # Load data into heap
        for i in range(dim_start, dims[0]):
            for j in range(dim_start, dims[1]):
                for k in range(dim_start, dims[2]):
                    for t in range(dim_start, dims[3]):
                        if data_view[i,j,k,t] < 0:
                            voxel_heap.push(VoxelIJK(i,j,k,t), -data_view[i,j,k,t])
    else:
        # Load data into heap
        for i in range(dim_start, dims[0]):
            for j in range(dim_start, dims[1]):
                for k in range(dim_start, dims[2]):
                    for t in range(dim_start, dims[3]):
                        if data_view[i,j,k,t] > 0:
                            voxel_heap.push(VoxelIJK(i,j,k,t), data_view[i,j,k,t])

    # Loop over heap
    # I know the code duplication is attrocious, but I don't want to have
    # the if pad inside the voxel loop, it adds 2% to the run time there
    if pad:
        while not voxel_heap.isEmpty():
            # Reset the set of touching clusters
            touching_clusters.clear()
            # Get the top voxel and it's value
            # The heap is max sorted, so this is the max remaining value in the data
            voxel = voxel_heap.pop(&value)
            
            # Convert 4 d indicies to 1 dimensional indicies
            # This is ugly, but moving this out of a function saves 0.2 microseconds per hit.
            # And it gets hit a lot
            voxel_index = voxel.m_ijk[0] * m_mult[0] + voxel.m_ijk[1] * m_mult[1] + voxel.m_ijk[2] * m_mult[2] + voxel.m_ijk[3]

            # Populate touching clusters
            for i in range(0, STENCIL_SIZE, 4):
                neigh_voxel = VoxelIJK(voxel.m_ijk[0] + stencil[i],
                                       voxel.m_ijk[1] + stencil[i + 1],
                                       voxel.m_ijk[2] + stencil[i + 2],
                                       voxel.m_ijk[3] + stencil[i + 3])
                # Don't have to check if it's a valid neighbor, the padding garantees that.
                neigh_index = neigh_voxel.m_ijk[0] * m_mult[0] + neigh_voxel.m_ijk[1] * m_mult[1] + neigh_voxel.m_ijk[2] * m_mult[2] + neigh_voxel.m_ijk[3]
                if membership[neigh_index] != -1:
                    touching_clusters.insert(membership[neigh_index])

            num_touching = touching_clusters.size()
            
            # Make a new cluster if no clusters neighbor this element
            if num_touching == 0:
                new_cluster = allocCluster(cluster_list, dead_clusters)
                cluster_list[new_cluster].addMember(voxel, value, volume, param_e, param_h)
                membership[voxel_index] = new_cluster
            # Add to a cluster if this element is touching a single cluster
            elif num_touching == 1:
                which_cluster = deref(touching_clusters.begin())
                cluster_list[which_cluster].addMember(voxel, value, volume, param_e, param_h)
                membership[voxel_index] = which_cluster
                accum_data_view[voxel_index] -= cluster_list[which_cluster].accumVal
            # If touching multiple clusters, merge them all, biggest first
            else:
                merged_index = -1
                biggest_size = 0
                # Find biggest cluster in terms of number of members and merge everything to that
                # This limits the number of cluster reassignments necessary
                for clust in touching_clusters:
                    if cluster_list[clust].members.size() > biggest_size:
                        merged_index = clust
                        biggest_size = cluster_list[clust].members.size()
                # update the selected cluster with the current value to align cluster bottoms
                cluster_list[merged_index].update(value, param_e, param_h)
                for clust in touching_clusters:
                    # if we are the largest cluster, don't modify the per-node accum
                    # for members, so merges between small and large clusters are cheap
                    if clust != merged_index:
                        # Recalculate to align cluster bottoms
                        cluster_list[clust].update(value, param_e, param_h)
                        num_members = cluster_list[clust].members.size()
                        # fix the accum values in the side cluster so we can add the 
                        # merged cluster's accum to everything at the end
                        correction_val = cluster_list[clust].accumVal - cluster_list[merged_index].accumVal
                        
                        #add the correction value to every member so that we have the current integrated values correct
                        for j in range(num_members):
                            member_index = cluster_list[clust].members[j].m_ijk[0] * m_mult[0] + cluster_list[clust].members[j].m_ijk[1] * m_mult[1] + cluster_list[clust].members[j].m_ijk[2] * m_mult[2] + cluster_list[clust].members[j].m_ijk[3]
                            # apply correction
                            accum_data_view[member_index] += correction_val
                            # update membership
                            membership[member_index] = merged_index
                        # copy all members
                        cluster_list[merged_index].members.insert(cluster_list[merged_index].members.end(), cluster_list[clust].members.begin(), cluster_list[clust].members.end())
                        cluster_list[merged_index].totalVolume += cluster_list[clust].totalVolume
                        # mark the old cluster as dead and eligble for reassignment
                        dead_clusters.insert(clust)
                        # clear member list
                        vector[VoxelIJK]().swap(cluster_list[clust].members)
                # will not trigger recomputation, we already recomputed at this value        
                cluster_list[merged_index].addMember(voxel, value, volume, param_e, param_h)
                # the element they merge on must not get the peak value of the cluster,
                # so again, record its difference from peak
                accum_data_view[voxel_index] -= cluster_list[merged_index].accumVal
                membership[voxel_index] = merged_index
                
                # do not reset the accum value of the merged cluster,
                # we specifically avoided modifying the per-node accum for its
                # members, so the cluster accum is still in play
    else:
        while not voxel_heap.isEmpty():
            touching_clusters.clear()
            voxel = voxel_heap.pop(&value)
            # This is ugly, but moving this out of a function saves 0.2 microseconds per hit.
            voxel_index = voxel.m_ijk[0] * m_mult[0] + voxel.m_ijk[1] * m_mult[1] + voxel.m_ijk[2] * m_mult[2] + voxel.m_ijk[3]

            # Populate touching clusters
            for i in range(0, STENCIL_SIZE, 4):
                neigh_voxel = VoxelIJK(voxel.m_ijk[0] + stencil[i],
                                       voxel.m_ijk[1] + stencil[i + 1],
                                       voxel.m_ijk[2] + stencil[i + 2],
                                       voxel.m_ijk[3] + stencil[i + 3])
                if index_valid(neigh_voxel.m_ijk, dims):
                    neigh_index = neigh_voxel.m_ijk[0] * m_mult[0] + neigh_voxel.m_ijk[1] * m_mult[1] + neigh_voxel.m_ijk[2] * m_mult[2] + neigh_voxel.m_ijk[3]
                    if membership[neigh_index] != -1:
                        touching_clusters.insert(membership[neigh_index])

            num_touching = touching_clusters.size()

            if num_touching == 0:
                new_cluster = allocCluster(cluster_list, dead_clusters)
                cluster_list[new_cluster].addMember(voxel, value, volume, param_e, param_h)
                membership[voxel_index] = new_cluster
            elif num_touching == 1:
                which_cluster = deref(touching_clusters.begin())
                cluster_list[which_cluster].addMember(voxel, value, volume, param_e, param_h)
                membership[voxel_index] = which_cluster
                accum_data_view[voxel_index] -= cluster_list[which_cluster].accumVal
            else:
                merged_index = -1
                biggest_size = 0
                # Find biggest cluster in terms of number of members and merge everything to that
                for clust in touching_clusters:
                    if cluster_list[clust].members.size() > biggest_size:
                        merged_index = clust
                        biggest_size = cluster_list[clust].members.size()
                cluster_list[merged_index].update(value, param_e, param_h)
                for clust in touching_clusters:
                    if clust != merged_index:
                        cluster_list[clust].update(value, param_e, param_h)
                        num_members = cluster_list[clust].members.size()
                        correction_val = cluster_list[clust].accumVal - cluster_list[merged_index].accumVal
                        for j in range(num_members):
                            member_index = cluster_list[clust].members[j].m_ijk[0] * m_mult[0] + cluster_list[clust].members[j].m_ijk[1] * m_mult[1] + cluster_list[clust].members[j].m_ijk[2] * m_mult[2] + cluster_list[clust].members[j].m_ijk[3]
                            accum_data_view[member_index] += correction_val
                            membership[member_index] = merged_index
                        cluster_list[merged_index].members.insert(cluster_list[merged_index].members.end(), cluster_list[clust].members.begin(), cluster_list[clust].members.end())
                        cluster_list[merged_index].totalVolume += cluster_list[clust].totalVolume
                        dead_clusters.insert(clust)
                        vector[VoxelIJK]().swap(cluster_list[clust].members)
                cluster_list[merged_index].addMember(voxel, value, volume, param_e, param_h)
                accum_data_view[voxel_index] -= cluster_list[merged_index].accumVal
                membership[voxel_index] = merged_index
                
    for clust in range(cluster_list.size()):
        if dead_clusters.find(clust) != dead_clusters.end():
            continue
        cluster_list[clust].update(0, param_e, param_h)
        num_members = cluster_list[clust].members.size()
        for j in range(num_members):
            member_index = cluster_list[clust].members[j].m_ijk[0] * m_mult[0] + cluster_list[clust].members[j].m_ijk[1] * m_mult[1] + cluster_list[clust].members[j].m_ijk[2] * m_mult[2] + cluster_list[clust].members[j].m_ijk[3]
            accum_data_view[member_index] += cluster_list[clust].accumVal
            

def tfce(data, float volume=1.0, float param_e=0.66666666, float param_h=2.0, pad=False):
    """Threshold-Free Cluster Enhancement (TFCE) as desribed in Smith & Nichols, 2009.
    Rescales statistical maps so that values represent both magnitude of effect 
    and base of support. Currently only face touching neighbors are considered.
    Positive and negative values are each TFCEd separately and recombined.
    
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
            e(h_{(n-1)}, p_{(0)}) \frac{ h_{(n-1)}^{H+1} - h_{(n)}^{H+1} }{ H+1 }
    
    Parameters
    ----------
    data: numpy.array, up to 4-d
        Array of values up to 4 dimensional.
    volume: float
        Volume of a single elements in the input dataset.
    param_e: float
        E parameter that determines the contribution of spatial extent
    param_h: float
        H parameter that determines contribution of magnitude
    pad: Bool
        Whether or not to use padding instead of index checking. Developmental flag,
        TFCE is slightly faster with pad=True, at the cost of somewhat higher memory.
    
    Returns
    -------
    result: numpy.array
        Array of TFCEd values of same shape as input data.
        
    
    Implementation is based on code by Tim Coalson in the HCP Connectome Workbench.
    Initial python implmentation and above TFCE description by Murat Bilgel. 
    Cython wrapping by Dylan Nielson.
    
    Citations
    ---------
    Smith SM, Nichols TE., "Threshold-free cluster enhancement: addressing problems of 
      smoothing, threshold dependence and localisation in cluster inference." 
      Neuroimage. 2009 Jan 1;44(1):83-98. PMID: 18501637"
    """
    
    # Add axes to get data to 4 dimensional
    orig_shape = data.shape
    if len(data.shape) == 1:
        data = data[np.newaxis, np.newaxis,np.newaxis,:]
    elif len(data.shape) == 2:
        data = data[np.newaxis, np.newaxis,:,:]
    elif len(data.shape) == 3:
        data = data[np.newaxis, : ,: ,:]
    elif len(data.shape) >4 :
        raise NotImplementedError
    
    # Pad the data to obviate the need for index checking for a small speedup
    if pad:
        cntg_data = np.ascontiguousarray(np.pad(data, 1, mode="constant", constant_values=0), dtype=np.float32)
        data_shape = np.array(data.shape)+2
    else:
        cntg_data = np.ascontiguousarray(data, dtype=np.float32)
        data_shape = data.shape
    
    accum_data = np.zeros(np.product(data_shape), dtype = np.float32)
    # These will modify the values in accum_data in place
    four_d_TFCE(cntg_data, accum_data, volume=volume, param_e=param_e, param_h=param_h, negate=False, pad=pad)
    four_d_TFCE(cntg_data, accum_data, volume=volume, param_e=param_e, param_h=param_h, negate=True, pad=pad)
    
    # Reshape accum_data
    accum_data = accum_data.reshape(data_shape)
    # Strip the padding
    if pad:
        accum_data = accum_data[1:-1, 1:-1, 1:-1, 1:-1]
    # Make the negative values negative
    accum_data[data<0] *= -1
    # Strip off any added dimensions
    accum_data = accum_data.reshape(orig_shape)
    return accum_data        


def tfce_smoketest(d, pad = False):
    data = np.random.randn(d, d ,d ,d).astype(np.float32)
    accum_data = tfce(data, pad = pad)
    assert accum_data.shape == data.shape
    assert (accum_data == 0).all() == False
