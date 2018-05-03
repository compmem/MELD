import numpy as np
import heapq
from .cluster import sparse_dim_connectivity


class Cluster():
    def __init__(self, param_E, param_H):
        '''
            Initialize empty cluster
        '''
        self.param_E = param_E
        self.H_plus1 = param_H + 1

        self.accumVal = 0. # summed TFCE over all cluster members
        self.extent = 0. # extent of cluster; called "totalArea" in Cpp implementation
        self.members = [] # list of cluster members
        self.upperLimit = 0. # upper limit of TFCE integral; called "lastVal" in Cpp implementation

    def update(self, lowerLimit):
        '''
            Perform an incremental update to the summed TFCE integral over cluster members

            Args
            ----
            lowerLimit : float
                lower limit of TFCE integral
                called "bottomVal" in Cpp implementation
        '''
        if len(self.members)==0:
            self.upperLimit = lowerLimit
        elif not (lowerLimit==self.upperLimit):
            assert(lowerLimit < self.upperLimit)
            newSliceVal = np.power(self.extent, self.param_E) * \
                (np.power(self.upperLimit, self.H_plus1) - np.power(lowerLimit, self.H_plus1)) / self.H_plus1
            self.accumVal += newSliceVal
            self.upperLimit = lowerLimit

    def addMember(self, node, height, extent):
        '''
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
        '''
        self.update(height)
        self.members.append(node) # append node to list of members
        self.extent += extent

def allocCluster(clusterList, deadClusters, param_E, param_H):
    '''
        Generate a valid (new) cluster number
        If available, number for a dead cluster will be recycled
        Cluster and dead cluster lists will be updated

        Args
        ----
            clusterList : list of Clusters
            deadClusters : set of Clusters
            param_E : float
            param_H : float
    '''
    # verify that the lists are actually being modified in main code
    if len(deadClusters)==0:
        clusterList.append( Cluster(param_E, param_H) )
        return len(clusterList) - 1
    else:
        ret = deadClusters.pop() # removes and returns element from set
        clusterList[ret] = Cluster(param_E, param_H)
        return ret

def getNodeNeighbors(node, connectivity):
    '''
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

    '''
    # since connectivity matrix is triangular, we need to check both along rows
    # and cols
    # diagonals are all 0 in connectivity matrix (no self-neighboring) so we
    # don't need to worry about that
    neighbors = np.append(connectivity.col[connectivity.row==node],
                          connectivity.row[connectivity.col==node])
    return neighbors

def tfce_pos(colData, areaData, connectivity, param_E = 0.5, param_H = 2.0):
    '''
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
            colData : 1-D numpy array of floats
            areaData : 1-D numpy array of floats
            param_E : float
            param_H : float

        Returns
        -------
            accumData : 1-D numpy array of floats
                array of TFCE values at each node
    '''
    # colData must be n-by-1
    # areaData must be n-by-1
    # connectivity must be a sparse n-by-n matrix; upper triangular only

    numNodes = len(colData)
    membership = -np.ones_like(colData, dtype=np.int)
    accumData = np.zeros_like(colData)
    clusterList = []
    deadClusters = set()
    nodeHeap = []

    for i, coldt in enumerate(colData):
        if coldt > 0:
            heapq.heappush(nodeHeap, (-coldt,i)) # first element is the priority (determines heap order)

    while len(nodeHeap)>0:
        negvalue, node = heapq.heappop(nodeHeap)
        value = -negvalue
        neighbors = getNodeNeighbors(node, connectivity)
        numNeigh = len(neighbors)

        touchingClusters = set()
        for nbr in neighbors:
            if membership[nbr] != -1:
                touchingClusters.add(membership[nbr])

        numTouching = len(touchingClusters)
        if numTouching==0: # make new cluster
            newCluster = allocCluster(clusterList, deadClusters, param_E, param_H)
            clusterList[newCluster].addMember(node, value, areaData[node])
            membership[node] = newCluster
        elif numTouching==1: # add to cluster
            whichCluster = touchingClusters.pop()
            clusterList[whichCluster].addMember(node, value, areaData[node])
            membership[node] = whichCluster
            accumData[node] -= clusterList[whichCluster].accumVal
        else: # merge all touching clusters
            # find the biggest cluster (i.e., with most members) and use as merged cluster
            mergedIndex = -1
            biggestSize = 0

            for tclust in touchingClusters:
                if len(clusterList[tclust].members) > biggestSize:
                    mergedIndex = tclust
                    biggestSize = len(clusterList[tclust].members)

            # assert vector index .. ?
            assert((mergedIndex>=0) and (mergedIndex<len(clusterList)))

            mergedCluster = clusterList[mergedIndex]
            mergedCluster.update(value) # recalculate to align cluster bottoms

            for tclust in touchingClusters:
                # if we are the largest cluster, don't modify the per-node accum
                # for members, so merges between small and large clusters are cheap
                if tclust != mergedIndex:
                    thisCluster = clusterList[tclust]
                    thisCluster.update(value) # recalculate to align cluster bottoms

                    correctionVal = thisCluster.accumVal - mergedCluster.accumVal
                    for mbr in thisCluster.members:
                        accumData[mbr] += correctionVal
                        membership[mbr] = mergedIndex

                    mergedCluster.members.extend(thisCluster.members)
                    mergedCluster.extent += thisCluster.extent

                    deadClusters.add(tclust) # designate (old) cluster as dead
                    clusterList[tclust].members = [] # deallocate member list

            mergedCluster.addMember(node, value, areaData[node]) # will not trigger recomputation; we already recomputed at this value

            # the node they merge on must not get the peak value of the cluster,
            # so again, record its difference from peak
            accumData[node] -= mergedCluster.accumVal
            membership[node] = mergedIndex

            # do not reset the accum value of the merged cluster,
            # we specifically avoided modifying the per-node accum for its
            # members, so the cluster accum is still in play

    # final clean up of accum data
    for i, thisCluster in enumerate(clusterList):

        if not (i in deadClusters): # ignore clusters that don't exist
            thisCluster.update(0.0) # update to include the to-zero slice
            for mbr in thisCluster.members:
                # add the resulting slice to all members -
                # their stored data contains the offset between the cluster peak and their corect value
                accumData[mbr] += thisCluster.accumVal

    return accumData
