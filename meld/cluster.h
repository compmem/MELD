#ifndef __CLUSTER_H__
#define __CLUSTER_H__

/*LICENSE_START*/
/*
 *  Copyright (C) 2014  Washington University School of Medicine
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */
/*LICENSE_END*/

#include <vector>
#include <cmath>
#include <set>
#include "Common/VoxelIJK.h"
//#include "Common/CaretAssertion.h"

using namespace caret;
using namespace std;

namespace Cluster
{

    class Cluster
    {
        public:
            float accumVal, totalVolume;
            vector<VoxelIJK> members;
            float lastVal;
            bool first;
            Cluster()
            {
                first = true;
                accumVal = 0.0;
                totalVolume = 0.0;
            }
            void addMember(const VoxelIJK& voxel, const float& val, const float& voxel_volume, const float& param_e, const float& param_h)
            {
                update(val, param_e, param_h);
                members.push_back(voxel);
                totalVolume += voxel_volume;
            }
            void update(const float& bottomVal, const float& param_e, const float& param_h)
            {
                if (first)
                {
                    lastVal = bottomVal;
                    first = false;
                } else {
                    if (bottomVal != lastVal)//skip computing if there is no difference
                    {
                        //CaretAssert(bottomVal < lastVal);
                        float integrated_h = param_h + 1.0f;//integral(x^h) = (x^(h + 1))/(h + 1) + C
                        double newSlice = pow(totalVolume, (double)param_e) * (pow((double)lastVal, integrated_h) - pow((double)bottomVal, integrated_h)) / integrated_h;
                        accumVal += newSlice;
                        lastVal = bottomVal;//computing in double precision, with float for inputs, puts the smallest difference between values far greater than the instability of the computation
                    }
                }
            }
    };
    
    int64_t allocCluster(vector<Cluster>& clusterList, set<int64_t>& deadClusters)
    {
        if (deadClusters.empty())
        {
            clusterList.push_back(Cluster());
            return (int64_t)(clusterList.size() - 1);
        } else {
            set<int64_t>::iterator iter = deadClusters.begin();
            int64_t ret = *iter;
            deadClusters.erase(iter);
            clusterList[ret] = Cluster();//reinitialize
            return ret;
        }
    }
};

#endif // __CLUSTER_H__