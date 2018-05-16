# distutils: language=c++
# distutils: libraries = m
# distutils: sources = cluster.h
# distutils: sources = Common/VoxelIJK.h
# distutils: sources = Common/CaretHeap.h
# distutils: include_dirs = Common

from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp cimport bool
from libc.stdint cimport int64_t
from caret_heap cimport CaretSimpleMaxHeap

cdef extern from "Common/VoxelIJK.h" namespace "caret":
    cdef cppclass VoxelIJK:
        int64_t m_ijk[4]
        VoxelIJK()
        VoxelIJK(int64_t i, int64_t j, int64_t k, int64_t) except +

cdef extern from "cluster.h" namespace "Cluster":
    cdef cppclass Cluster:
        Cluster() except +
        double accumVal, totalVolume
        vector[VoxelIJK] members
        bool first
        void addMember(const VoxelIJK& voxel, const float& val, const float& area, const float& param_e, const float& param_h) except +
        void update(const float& bottomVal, const float& param_e, const float& param_h) except +

cdef extern from "cluster.h" namespace "Cluster":
    cdef int64_t allocCluster(vector[Cluster]& clusterList, set[int64_t]& deadClusters)

ctypedef struct VoxelHeap:
    CaretSimpleMaxHeap[VoxelIJK, float] _heap

