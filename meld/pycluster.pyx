# distutils: language=c++
# distutils: libraries = m

cimport cython
from libc.stdint cimport int64_t
from ccluster cimport Cluster, VoxelIJK, VoxelHeap
#from caret_heap cimport CaretSimpleMaxHeap
from libcpp.vector cimport vector

cdef class PyVoxelIJK:
    cdef VoxelIJK _vox
    def __cinit__(PyVoxelIJK self, int64_t i, int64_t j, int64_t k, int64_t t):
        self._vox = VoxelIJK(i, j, k, t)
    @property
    def ijkt(self):
        return self._vox.m_ijk
    def __repr__(self):
        return self.ijkt.__repr__()


cdef class PyCluster:
    cdef Cluster clust
    def addmember(PyCluster self, PyVoxelIJK voxel, const float& val, const float& voxel_volume, const float& param_e, const float& param_h):
        self.clust.addMember(voxel._vox, val, voxel_volume, param_e, param_h)

    def update(PyCluster self, const float& bottomVal, const float& param_e, const float& param_h):
        self.clust.update(bottomVal, param_e, param_h)

    @property
    def members(self):
        return [vox.m_ijk for vox in self.clust.members]

    @property
    def accumVal(self):
        return self.clust.accumVal
    @property
    def totalVolume(self):
        return self.clust.totalVolume
    @property
    def first(self):
        return self.clust.first

cdef class PyVoxelHeap:

    cdef VoxelHeap _h

    def push(VoxelHeap self, PyVoxelIJK t, float k):
        self._h._heap.push(t._vox, k)

    def pop(VoxelHeap self):
        cdef float* value
        return PyVoxelIJK(*self._h._heap.pop(value).m_ijk)

    def clear(VoxelHeap self):
        self._h._heap.clear()

    @property
    def size(VoxelHeap self):
        return self._h._heap.size()

    @property
    def isEmpty(VoxelHeap self):
        return self._h._heap.isEmpty()
