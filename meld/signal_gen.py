import os
import sys
import time
import numpy as np
#import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import stats

def sig_by_prop(p,size=(100,100)):
    field = np.zeros(size)
    n = (field.shape[0]*field.shape[1])*p
    randx= np.random.randint(0,field.shape[0],n)
    randy= np.random.randint(0,field.shape[1],n)

    field[randx,randy] = 1
    return field

def hide_blobs(nfeats=(100,100), x=0, y=0,blob_pat='big',zfield = None,cfrac = None):
    """ puts a (artisinally hand crafted) set of 1's surrounded by 0's
        into a field of dims nfeats
        currently supported patterns are small, final, big, vbig,ten,hundo,
        for the MELD paper, I've added central, split, and dispersed, 
        these hijack your x and y and assume you've got 100 features""" 

    #check input
    if x > nfeats[0] or y > nfeats[1]:
        print("You tried to place the blobs outside the indicies of the feature space you asked for.")
        print("setting x and y to 0")
        x = y = 0
    
    if zfield is None:
        zfield = np.zeros(nfeats)
    
    if cfrac is not None:
    #make center blob
        facs = np.sort(list(factors(cfrac)))
        if len(facs)%2==0:
            cdim = facs[len(facs)//2-1:len(facs)//2+1]
        else:
            cdim = np.array([facs[len(facs)//2],facs[len(facs)//2]])
        #figure out how to center center blob
        cyx = ((np.array(nfeats)//2)-(cdim//2)).astype(int)
        cdim = cdim.astype(int)

        #hide center
        zfield[cyx[0]:cyx[0]+cdim[0],cyx[1]:cyx[1]+cdim[1]] = 1

        dfrac = 100-cfrac

        dyxlist = [10,30,50,70,90]

        dsum = 0
        for y in dyxlist:
            for x in dyxlist:
                if (x != 50) | (y!=50):
                    if dfrac-dsum>=4:
                        ddim = np.array([2,2])
                    elif dfrac-dsum>0:
                        ddim = np.array([dfrac-dsum,1])
                    else:
                        break
                    ddim = ddim.astype(int)
                    dsum += np.ones(ddim).sum()
                    zfield[y:y+ddim[0],x:x+ddim[1]]=1
        return zfield
    
    xs = None
    #This is what our signal will eventually be multiplied by to give a spatially distributed signal
    
    if blob_pat =='small':
        blobs = np.array ([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,1.,1.,0.,0.,1.,1.,0.,0.,1.,0.,0.,0.,1.],
                           [0.,0.,0.,1.,1.,0.,0.,1.,1.,0.,0.,1.,1.,0.,0.,1.],
                           [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,1.,1.,0.,0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,1.,1.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,0.],
                           [-1.,0.,0.,-1.,-1.,0.,0.,-1.,-1.,0.,0.,-1.,-1.,-1.,-1.,-1.],
                           [-1.,0.,0.,0.,-1.,0.,0.,-1.,-1.,0.,0.,0.,-1.,-1.,-1.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.]])
        
    elif blob_pat =='final':
        blobs = np.array ([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],
                            [1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                            [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,1,1,0,0,1,1],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0.166666667,0.166666667,0,0,0.333333333,0.333333333,0,0,0.5,0.5,0,0,0.666666667,0.666666667,0,0,0.833333333,0.833333333,0,0,1,1],
                            [0.166666667,0.166666667,0,0,0.333333333,0.333333333,0,0,0.5,0.5,0,0,0.666666667,0.666666667,0,0,0.833333333,0.833333333,0,0,1,1],
                            [0.166666667,0.166666667,0,0,0.333333333,0.333333333,0,0,0.5,0.5,0,0,0.666666667,0.666666667,0,0,0.833333333,0.833333333,0,0,1,1],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [-0.166666667,-0.166666667,0,0,-0.333333333,-0.333333333,0,0,-0.5,-0.5,0,0,-0.666666667,-0.666666667,0,0,-0.833333333,-0.833333333,0,0,-1,-1],
                            [-0.166666667,-0.166666667,0,0,-0.333333333,-0.333333333,0,0,-0.5,-0.5,0,0,-0.666666667,-0.666666667,0,0,-0.833333333,-0.833333333,0,0,-1,-1],
                            [-0.166666667,-0.166666667,0,0,-0.333333333,-0.333333333,0,0,-0.5,-0.5,0,0,-0.666666667,-0.666666667,0,0,-0.833333333,-0.833333333,0,0,-1,-1],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,-1],
                            [-1,0,0,0,-1,-1,0,0,-1,-1,0,0,-1,-1,0,0,-1,-1,0,0,-1,-1],
                            [0,0,0,0,0,0,0,0,0,-1,0,0,-1,-1,0,0,-1,-1,0,0,-1,-1]])
    elif blob_pat =='vbig':
        blobs = np.array ([[1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,-1.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,-1.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,-1.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,-1.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,-1.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,-1.]])
        
    elif blob_pat =='ten':
        blobs = np.array ([[0.,1.,1.,0.],
                           [0.,1.,1.,1.],
                           [0.,1.,1.,1.],
                           [0.,1.,1.,0.]])
        
    elif blob_pat =='hundo':
        blobs = np.array ([[0.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [0.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [0.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [0.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [0.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,1.,1.,1.,1.,0.],
                           [0.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,0.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,0.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,0.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,0.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,0.],
                           [0.,-1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.,0.]])
    elif blob_pat == 'central01':
        x = 49
        y = 47
        blobs = np.ones((5,2))
    
    elif blob_pat == 'central':
        x = y = 45
        blobs = np.ones((10,10))
        
    elif blob_pat == 'central10':
        x = 37
        y = 30
        blobs = np.ones((25,40))
        
    elif blob_pat == 'split01':
        xs = [40,60]
        ys = [47]
        blobs = np.ones((5,1))

    elif blob_pat == 'split':
        xs = ys = [40,60]
        blobs = np.ones((5,5))
        
    elif blob_pat == 'split10':
        xs = [15,60]
        ys = [30,60]
        blobs = np.ones((10,25))
    
    elif blob_pat == 'dispersed01':
        xs = [40,60]
        ys = [10,30,50,70,90]
        blobs = np.ones((1,1))

    elif blob_pat == 'dispersed':
        xs = ys = [10,30,50,70,90]
        blobs = np.ones((2,2))
    
    elif blob_pat == 'dispersed10':
        xs = ys = [6,26,46,66,86]
        blobs = np.ones((5,8))

    else:
        if blob_pat != 'big':
            print("I don't know that shape, setting shape to big")
        blobs = np.array ([[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.],
                           [1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.],
                           [0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,0.],
                           [0.,0.,-1.,-1.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,-1.,-1.],
                           [0.,0.,-1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.]])
    #this sets blobs equal to the dims of nfeat-placement indices
    if nfeats[0] - x < np.shape(blobs)[0]:  
        blobs = blobs[0:nfeats[0]-x,:]
    if nfeats[1] - y  < np.shape(blobs)[1]:
        blobs = blobs[:,0:nfeats[1]-y]
        
        
    if xs is not None:
        for y in ys:
            for x in xs:
                zfield[y:y+blobs.shape[0],x:x+blobs.shape[1]] = blobs

    else:
        zfield[x:x+blobs.shape[0],y:y+blobs.shape[1]] = blobs
    return zfield


# In[4]:

def factors(n):
    res = []
    for i in  range(1, int(n**0.5) + 1):
        if n % i == 0:
            res.extend([i,n//i])
    return set(res)


# In[5]:

def gen_data(nsubj,nobs,nfeat,slope,sigfield,mnoise=False,contvar=False,item_inds=False,mod_cont=None,
             I = 0.0, S = 0.0, field_noise=1.0):
    """Generate a set of random data with a independent data and dependent data matricies
       Data will have nsubj*nobs rows and dep data will have 1 column for each feature
       sigfield is the matrix the signal should be multiplied by to give it a "spatial" extent
       Independent by default has a factor called beh which represents some behavioral condition
       if mnoise,  then the slopes have subject and item level noise added to them
                ie, this is the type of noise you are modeling with (x|subj) or (x|item)
       if contvar, then there is a continuous variable in the data that interacts with the
                beh factor
       if mod_cont, then the continuous variable is included in the output independent data
                This is here incase I want to look at the effects of including a continuous
                variable in your data when its not there and vice versa
        ##As of 3-7-14, item level noise is used only for slopes.##
       if item_inds is true then the independent variable table is output with a column for item
                item level noise is used to generate the model regardless of this option
                """
    
    if mod_cont is None:
        mod_cont = contvar
    
    #talked about implementing blur, but haven't
    #blur = 2
    #define slopes, they are all set to slope for simplicity for now
    #could set it to accept a dict of slopes or some such
    beh_slope = slope
    cont_slope = slope
    bxc_slope = slope
    
    #set up the behavioral array, this is set up for two behavioral conditions currently
    beh_list = np.array([-0.5,0.5]*int(nobs/2))
    if nobs != len(beh_list):
        nobs = len(beh_list)
        print('The number of observations you provided was not divisible by 2,')
        print('it has been set to', nobs)
    stim_list = np.vstack((beh_list, np.arange(nobs)))

    #generate independent variables and put it in a rec_array
    #setting up the columns for ind_data
    #subject number
    s = []
    #happy panda or sad panda?
    beh = []
    #which specific item was presented? list of item numbers
    items = []
    for i in range(nsubj):
        stmp = np.array([i]*nobs)
        rand_ind = np.random.permutation(np.arange(nobs, dtype=int))
        stim_list_tmp = stim_list[:,rand_ind]
        s.extend(list(stmp))
        beh.extend(stim_list_tmp[0])
        items.extend(stim_list_tmp[1])

    ind_data = np.rec.fromarrays((np.random.randn(len(s)),
                                  np.random.randn(len(s)),
                                  beh,
                                  np.array(items).astype(int),
                                  s),
                                  names=['val','cont','beh','item','subj'])
    
    #set up noise, noise is all random standard normal
    #noise has 6 columns, 2 for each of beh, cont, bxc
    snoise = np.random.standard_normal((nsubj,6))*S
    itemnoise = np.random.standard_normal((nobs,6))*I
    #n_denom = 1+inoise
    #noisy field to which the signal y will be added
    dep_data = np.random.randn(len(s),*nfeat) * field_noise
    
    
    #save out stats on noisy field before signal is added to it
    sn_stats = dict(noise=dict(ave=np.mean(dep_data[:,sigfield!=0]),
                               sigma=np.std(dep_data[:,sigfield!=0]),
                               high=np.max(dep_data[:,sigfield!=0]),
                               low=np.min(dep_data[:,sigfield!=0])))
   
    ys = np.zeros((len(ind_data),sigfield.shape[0],sigfield.shape[1]))
    #loop through ind_data to created corresponding dep_data for each row
    for i in range(len(ind_data)):
        
        if mnoise == True:
            #mx is the ind data * slope * average
            #with slope noise
            #Previous methods of generating slope
            #beh_mx = ind_data['beh'][i]*beh_slope*((snoise[ind_data['subj'][i],1]+itemnoise[ind_data['subj'][i],1])/n_denom)
            #cont_mx = ind_data['cont'][i]*cont_slope*((snoise[ind_data['subj'][i],3]+itemnoise[ind_data['subj'][i],3])/n_denom)
            #beh_mx = ind_data['beh'][i]*(beh_slope+((snoise[ind_data['subj'][i],1]+itemnoise[ind_data['subj'][i],1])/n_denom))
            #cont_mx = ind_data['cont'][i]*(cont_slope+((snoise[ind_data['subj'][i],3]+itemnoise[ind_data['subj'][i],3])/n_denom))
            
            #beh_mx = ind_data['beh'][i]*(beh_slope+(snoise[ind_data['subj'][i],1]+itemnoise[ind_data['subj'][i],1]))
            #cont_mx = ind_data['cont'][i]*(cont_slope+(snoise[ind_data['subj'][i],3]+itemnoise[ind_data['subj'][i],3]))
            #bxc_mx = ind_data['beh'][i]*ind_data['cont'][i]*(beh_slope*cont_slope+(snoise[ind_data['subj'][i],5]+itemnoise[ind_data['subj'][i],5]))
            
            #Took out the item level slope noise after talking with Per on 3-7-14
            beh_mx = ind_data['beh'][i]*(beh_slope+(snoise[ind_data['subj'][i],1]))
            cont_mx = ind_data['cont'][i]*(cont_slope+(snoise[ind_data['subj'][i],3]))
            bxc_mx = ind_data['beh'][i]*ind_data['cont'][i]*(beh_slope*cont_slope+(snoise[ind_data['subj'][i],5]))
        else:
            #no  slope noise
            beh_mx = ind_data['beh'][i]*beh_slope
            cont_mx = ind_data['cont'][i]*cont_slope
            bxc_mx = ind_data['beh'][i]*ind_data['cont'][i]*bxc_slope
        
        #b is the average of item and subject intercept noise, Old methods
        #beh_b = (snoise[ind_data['subj'][i],0] + itemnoise[ind_data['subj'][i],0])/n_denom
        #cont_b = (snoise[ind_data['subj'][i],2] +itemnoise[ind_data['subj'][i],2])/n_denom        
        #beh_b = (snoise[ind_data['subj'][i],0] + itemnoise[ind_data['subj'][i],0])
        #cont_b = (snoise[ind_data['subj'][i],2] +itemnoise[ind_data['subj'][i],2])
        #bxc_b = (snoise[ind_data['subj'][i],4] +itemnoise[ind_data['subj'][i],4])
        
        #Took out the item level intercept noise after talking with Per on 3-7-14
        #Put it back in after looking at http://talklab.psy.gla.ac.uk/KeepItMaximalR2.pdf
        #page 13
        beh_b = (snoise[ind_data['subj'][i],0] + itemnoise[ind_data['item'][i],0])
        cont_b = (snoise[ind_data['subj'][i],2] + itemnoise[ind_data['item'][i],2])
        bxc_b = (snoise[ind_data['subj'][i],4] + itemnoise[ind_data['item'][i],4])
        
        #set up the signal for the ith row of ind data
        if contvar == True:
            ys[i,:,:] = sigfield * (beh_mx + beh_b + cont_mx + cont_b + bxc_mx + bxc_b)
        else:
            ys[i,:,:] = sigfield * (beh_mx + beh_b)
        
        #add signal to noisy field
        dep_data[i,:,:] = dep_data[i,:,:] + (ys[i,:,:])
    
    #save stats on signal before it is added to noise
    sn_stats['ys']=dict(ave=np.mean(abs(ys[:,sigfield!=0])),
                        sigma=np.std(abs(ys[:,sigfield!=0])),
                        high=np.max(abs(ys[:,sigfield!=0])),
                        low=np.min(abs(ys[:,sigfield!=0])))
    #save stats on combined signal and noise
    sn_stats['S_N']=dict(ave=np.mean(dep_data[:,sigfield!=0]),
                        sigma=np.std(dep_data[:,sigfield!=0]),
                        high=np.max(dep_data[:,sigfield!=0]),
                        low=np.min(dep_data[:,sigfield!=0]))
    
    #modify the ind_data table as necessary for output
    if item_inds == True:
        if mod_cont == True:
            ind_data = ind_data
        else:
            ind_data = ind_data[['val','beh','subj','item']]
    else:
        if mod_cont == True:
            ind_data = ind_data[['val','beh','subj','cont']]
        else:
            ind_data = ind_data[['val','beh','subj']]
    
    #data = (ind_data,ndimage.gaussian_filter(dep_data,2))
    data = (ind_data,dep_data,sn_stats)
    return data