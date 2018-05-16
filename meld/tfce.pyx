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
cdef four_d_TFCE(float [:,:,:,::1] data_view, float [::1] accum_data_view, float volume, float param_e=0.666666666, float param_h=2.0, bint negate = False):
    
    cdef int64_t sx = data_view.shape[0]
    cdef int64_t sy = data_view.shape[1]
    cdef int64_t sz = data_view.shape[2]
    cdef int64_t st = data_view.shape[3]
    cdef int64_t size = sx * sy * sz * st
    cdef int64_t[4] dims = [sx, sy, sz, st]
    cdef int64_t[4] m_mult
    cdef int_vec membership = int_vec(size, -1)
    #membership = -1
    cdef vector[Cluster] cluster_list
    cdef CaretSimpleMaxHeap[VoxelIJK, float] voxel_heap
    #cdef float [:, :, :, ::1] dataview = data
    #cdef float [:] accum_data_view = accum_data
    cdef size_t i, j, k, t
    cdef VoxelIJK neigh_voxel, voxel
    cdef float value
    cdef double correction_val
    cdef int64_t voxel_index, neigh_index, which_cluster, num_touching, new_cluster
    cdef int64_t merged_index, biggest_size, tclust, num_members, member_index
    cdef set[int64_t] dead_clusters, touching_clusters
    #cdef Cluster* merged_cluster

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

    # Initialize Multiplies for unraveling multiindex
    m_mult[0] = dims[0]
    for i in range(1, 4):
        m_mult[i] = m_mult[i - 1] * dims[i]

    #print(m_mult)

    # Load data into heap
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                for t in range(dims[3]):
                    if negate:
                        if data_view[i,j,k,t] < 0:
                            voxel_heap.push(VoxelIJK(i,j,k,t), -data_view[i,j,k,t])
                    else:
                        if data_view[i,j,k,t] > 0:
                            voxel_heap.push(VoxelIJK(i,j,k,t), data_view[i,j,k,t])

    # Loop over heap
    while not voxel_heap.isEmpty():
        touching_clusters.clear()
        voxel = voxel_heap.pop(&value)
        # This is ugly, but moving this out of a function saves ).2 us per hit.
        voxel_index = voxel.m_ijk[0] + m_mult[0] * voxel.m_ijk[1] + m_mult[1] * voxel.m_ijk[2] + m_mult[2] * voxel.m_ijk[3]
        #print(voxel.m_ijk, value, voxel_index)

        # Populate touching clusters
        for i in range(0, STENCIL_SIZE, 4):
            neigh_voxel = VoxelIJK(voxel.m_ijk[0] + stencil[i],
                                   voxel.m_ijk[1] + stencil[i + 1],
                                   voxel.m_ijk[2] + stencil[i + 2],
                                   voxel.m_ijk[3] + stencil[i + 3])
            #print(neigh_voxel.m_ijk)
            if index_valid(neigh_voxel.m_ijk, dims):
                neigh_index = neigh_voxel.m_ijk[0] + m_mult[0] * neigh_voxel.m_ijk[1] + m_mult[1] * neigh_voxel.m_ijk[2] + m_mult[2] * neigh_voxel.m_ijk[3]
                #print("neigh_index:", neigh_index, "neigh_voxel:", neigh_voxel.m_ijk)
                if membership[neigh_index] != -1:
                    #print("membership:", membership[neigh_index], flush=True)
                    touching_clusters.insert(membership[neigh_index])

        num_touching = touching_clusters.size()

        if num_touching == 0:
            #print("0 touching", flush=True)
            new_cluster = allocCluster(cluster_list, dead_clusters)
            cluster_list[new_cluster].addMember(voxel, value, volume, param_e, param_h)
            #print("new_cluster:", new_cluster, flush=True)
            membership[voxel_index] = new_cluster
        elif num_touching == 1:
            #print("1 touching", flush=True)
            which_cluster = deref(touching_clusters.begin())
            #print("which_cluster:", which_cluster, flush=True)
            #print(which_cluster, voxel.m_ijk, value, volume, param_e, param_h)

            cluster_list[which_cluster].addMember(voxel, value, volume, param_e, param_h)
            membership[voxel_index] = which_cluster
        else:
            #print("%d touching"% num_touching, flush=True)
            merged_index = -1
            biggest_size = 0
            # Find biggest cluster in terms of number of members and merge everything to that
            #print("a")
            for clust in touching_clusters:
                if cluster_list[clust].members.size() > biggest_size:
                    merged_index = clust
                    biggest_size = cluster_list[clust].members.size()
            #print("b")
            #print(cluster_list.size(), merged_index, num_touching, touching_clusters)
            cluster_list[merged_index].update(value, param_e, param_h)
            #print("a", flush=True)
            for clust in touching_clusters:
                if clust != merged_index:
                    #print("a1", flush=True)
                    cluster_list[clust].update(value, param_e, param_h)
                    num_members = cluster_list[clust].members.size()
                    correction_val = cluster_list[clust].accumVal - cluster_list[merged_index].accumVal
                    #print("a2", flush=True)
                    for j in range(num_members):
                        member_index = cluster_list[clust].members[j].m_ijk[0] + m_mult[0] * cluster_list[clust].members[j].m_ijk[1] + m_mult[1] * cluster_list[clust].members[j].m_ijk[2] + m_mult[2] * cluster_list[clust].members[j].m_ijk[3]
                        accum_data_view[member_index] += correction_val
                        membership[member_index] = merged_index
                    #print("a3", flush=True)
                    cluster_list[merged_index].members.insert(cluster_list[merged_index].members.end(), cluster_list[clust].members.begin(), cluster_list[clust].members.end())
                    cluster_list[merged_index].totalVolume += cluster_list[clust].totalVolume
                    #print("a4", flush=True)
                    dead_clusters.insert(clust)
                    vector[VoxelIJK]().swap(cluster_list[clust].members)
            #print("b", flush=True)
            cluster_list[merged_index].addMember(voxel, value, volume, param_e, param_h)
            accum_data_view[voxel_index] = cluster_list[merged_index].accumVal
            membership[voxel_index] = merged_index
    for clust in range(cluster_list.size()):
        if dead_clusters.find(clust) != dead_clusters.end():
            continue
        cluster_list[clust].update(0, param_e, param_h)
        num_members = cluster_list[clust].members.size()
        for j in range(num_members):
            member_index = cluster_list[clust].members[j].m_ijk[0] + m_mult[0] * cluster_list[clust].members[j].m_ijk[1] + m_mult[1] * cluster_list[clust].members[j].m_ijk[2] + m_mult[2] * cluster_list[clust].members[j].m_ijk[3]
            accum_data_view[member_index] += cluster_list[clust].accumVal
            

def tfce(data, float volume=1.0, float param_e=0.66666666, float param_h=2.0):
    if len(data.shape) == 1:
        data = data[np.newaxis, np.newaxis,np.newaxis,:]
    elif len(data.shape) == 2:
        data = data[np.newaxis, np.newaxis,:,:]
    elif len(data.shape) == 3:
        data = data[np.newaxis, : ,: ,:]
    elif len(data.shape) >4 :
        raise NotImplementedError
    cdef np.ndarray[float, ndim=4] cntg_data = np.ascontiguousarray(data, dtype=np.float32)
    accum_data = np.zeros(np.product(data.shape), dtype = np.float32)
    # These will modify the values in accum_data in place
    four_d_TFCE(cntg_data, accum_data, volume=volume, param_e=param_e, param_h=param_h, negate=False)
    four_d_TFCE(cntg_data, accum_data, volume=volume, param_e=param_e, param_h=param_h, negate=True)
    # These transposes are messy, but they're a quick fix for an indexing order mistake
    accum_data = accum_data.reshape(data.T.shape).T
    accum_data[data<0] *= -1
    return accum_data
