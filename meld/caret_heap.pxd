# distutils: language=c++
# distutils: libraries = m
# distutils: sources = Common/CaretHeap.h
# distutiles: include_dirs = Common

from libcpp.vector cimport vector
from libc.stdint cimport int64_t

cdef extern from "Common/CaretHeap.h" namespace "caret":

    cdef cppclass CaretSimpleMaxHeap[T,K]:
        CaretSimpleMaxHeap()
        CaretSimpleMaxHeap(T t, K k)
        struct DataStruct:
            K m_key
            T m_data
            DataStruct(const K& key, const T& data)
        vector[DataStruct] m_heap
        void push(const T& data, const K& key)
        #look at the data of the top element
        T& top(K* )
        
        #remove and return the top element
        T pop(K* key)
        
        #preallocate for efficiency, if you know about how big it will be
        void reserve(int64_t expectedSize)
        
        #check for empty
        bint isEmpty() const
        
        #get number of elements
        int64_t size() const
        
        #reset the heap
        void clear()

        bint mycompare(const K& left, const K& right)
