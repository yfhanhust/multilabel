import numpy as np 
import scipy as sp 
from sklearn.linear_model import Ridge 
import downhill
import theano
from sklearn.metrics import roc_auc_score

def LogitCoupledMFInductiveSim(X,Y,fea_loc,prob,nrank,lambda0,lambda1,lambda2,lambda3):
    mask = np.ones(X.shape)
    mask[fea_loc] = 0.
    
    ######### label matrix factorization using reweighted logit loss 
    U = theano.shared(np.random.random((X.shape[0],nrank)),name='U')
    V = theano.shared(np.random.random((nrank,X.shape[1])),name='V')  
    ######### UVT = X 
    W = theano.shared(np.random.random((X.shape[1],nrank)),name='W')
    H = theano.shared(np.random.random((nrank,Y.shape[1])),name='H')
    ######### corr_entropy(MN + XWH) = Y
    M = theano.shared(np.random.random((X.shape[0],nrank)),name='M')
    N = theano.shared(np.random.random((nrank,Y.shape[1])),name='N')
    ######### corr_entropy(MN + XWH) = Y
    
      
    #W = theano.shared(W_init)
    #H = theano.shared(H_init)
    pos_loc_num = len(np.where(Y > 0)[0])
    neg_loc_num = len(np.where(Y < 1)[0])
    factor = float(pos_loc_num) / float(neg_loc_num)
    tX = theano.tensor.matrix(name='X')
    tL = theano.tensor.matrix(name='Y')
    #pos_mask = np.zeros(Y.shape)
    #pos_mask[np.where(Y_flipped > 0)] = 1.
    #neg_mask = np.zeros(Y.shape)
    #neg_mask[np.where(Y_flipped < 0)] = 1. 
    #pmask = theano.tensor.matrix('pmask')
    #nmask = theano.tensor.matrix('nmask')
    ##### lsq error == learn U and V to approximate X 
    feature_mask = theano.tensor.matrix('feature_mask')
    UV = theano.tensor.dot(U,V)
    lsqloss = theano.tensor.sqr((tX - UV) * feature_mask).sum()
    
    #factor = float(pos_loc_num) / float(neg_loc_num)
    WH = theano.tensor.dot(W,H)
    XWH = theano.tensor.dot(UV,WH)
    MN = theano.tensor.dot(M,N)
    logit_pos = (1.-prob) * theano.tensor.gt(tL,0) * theano.tensor.log(1. + theano.tensor.exp(-1.* XWH - MN))
    logit_neg = prob * theano.tensor.lt(tL,1) * theano.tensor.log(1. + theano.tensor.exp(XWH + MN))
    #logit_pos = (1.-prob) * theano.tensor.log(1. + theano.tensor.exp(-1.* theano.tensor.dot(W,H)))
    #logit_neg = prob * theano.tensor.log(1. + theano.tensor.exp(theano.tensor.dot(W,H)))
    logit_loss = logit_pos.sum() + logit_neg.sum() #+ lambda0 * ((W * W).mean() + (H * H).mean())
    lse_loss = lsqloss + lambda0 * logit_loss + lambda1 * ((U*U).mean() + (V*V).mean()) + lambda2 * ((W*W).mean() + (H*H).mean())  + lambda3 * ((M*M).mean() + (N*N).mean())
    nbatch = Y.shape[0]
    
    downhill.minimize(
            loss= lse_loss,
            params = [U,V,W,H,M,N],
            train = [X,Y,mask],
            inputs = [tX,tL,feature_mask],
            patience=0,
            algo='adagrad',
            batch_size= nbatch,
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.001)
    
    return U.get_value(),V.get_value(),W.get_value(),H.get_value(),M.get_value(),N.get_value()
    
def LogitCoupledMFInductiveSimNL(X,Y,fea_loc,prob,nrank,lambda0,lambda1,lambda2,lambda3,random_weight,random_offset):
    mask = np.ones(X.shape)
    mask[fea_loc] = 0.
    no_of_components = random_weight.shape[0]
    
    ######### label matrix factorization using reweighted logit loss 
    U = theano.shared(np.random.random((X.shape[0],nrank)),name='U')
    V = theano.shared(np.random.random((nrank,X.shape[1])),name='V')  
    ######### UVT = X 
    W = theano.shared(np.random.random((no_of_components,nrank)),name='W')
    H = theano.shared(np.random.random((nrank,Y.shape[1])),name='H')
    ######### corr_entropy(MN + XWH) = Y
    M = theano.shared(np.random.random((X.shape[0],nrank)),name='M')
    N = theano.shared(np.random.random((nrank,Y.shape[1])),name='N')
    ######### corr_entropy(MN + XWH) = Y
    
      
    #W = theano.shared(W_init)
    #H = theano.shared(H_init)
    pos_loc_num = len(np.where(Y > 0)[0])
    neg_loc_num = len(np.where(Y < 1)[0])
    tX = theano.tensor.matrix(name='X')
    tL = theano.tensor.matrix(name='Y')
    #pos_mask = np.zeros(Y.shape)
    #pos_mask[np.where(Y_flipped > 0)] = 1.
    #neg_mask = np.zeros(Y.shape)
    #neg_mask[np.where(Y_flipped < 0)] = 1. 
    #pmask = theano.tensor.matrix('pmask')
    #nmask = theano.tensor.matrix('nmask')
    ##### lsq error == learn U and V to approximate X 
    feature_mask = theano.tensor.matrix('feature_mask')
    UV = theano.tensor.dot(U,V)
    lsqloss = theano.tensor.sqr((tX - UV) * feature_mask).sum()
    
    #factor = float(pos_loc_num) / float(neg_loc_num)
    WH = theano.tensor.dot(W,H)
    cos_coremat = theano.tensor.cos(theano.tensor.dot(UV,random_weight.T) + random_offset)
    P = (np.sqrt(2.) / np.sqrt(no_of_components)) * cos_coremat #### nsample * no_of_components
    XWH = theano.tensor.dot(P,WH)
    MN = theano.tensor.dot(M,N)
    logit_pos = (1.-prob) * theano.tensor.gt(tL,0) * theano.tensor.log(1. + theano.tensor.exp(-1.* XWH - MN))
    logit_neg = prob * theano.tensor.lt(tL,1) * theano.tensor.log(1. + theano.tensor.exp(XWH + MN))
    #logit_pos = (1.-prob) * theano.tensor.log(1. + theano.tensor.exp(-1.* theano.tensor.dot(W,H)))
    #logit_neg = prob * theano.tensor.log(1. + theano.tensor.exp(theano.tensor.dot(W,H)))
    logit_loss = logit_pos.sum() + logit_neg.sum() #+ lambda0 * ((W * W).mean() + (H * H).mean())
    lse_loss = lsqloss + lambda0 * logit_loss + lambda1 * ((U*U).mean() + (V*V).mean()) + lambda2 * ((W*W).mean() + (H*H).mean())  + lambda3 * ((M*M).mean() + (N*N).mean())
    nbatch = Y.shape[0]
    
    downhill.minimize(
            loss= lse_loss,
            params = [U,V,W,H,M,N],
            train = [X,Y,mask],
            inputs = [tX,tL,feature_mask],
            patience=0,
            algo='adagrad',
            batch_size= nbatch,
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.001)
    
    return U.get_value(),V.get_value(),W.get_value(),H.get_value(),M.get_value(),N.get_value()
    
def LogitMF(Y,lambda0,prob,nrank):
    #M = theano.tensor.matrix('Feature')
    W = theano.shared(np.random.random((Y.shape[0],nrank)),name='W')
    H = theano.shared(np.random.random((nrank,Y.shape[1])),name='H')
    
    
    Y_flipped = np.copy(Y)
    Y_flipped[np.where(Y < 1)] = -1
    pos_loc_num = len(np.where(Y_flipped > 0)[0])
    neg_loc_num = len(np.where(Y_flipped < 0)[0])
    tL = theano.tensor.matrix(name='Y')
    #pos_mask = np.zeros(Y.shape)
    #pos_mask[np.where(Y_flipped > 0)] = 1.
    #neg_mask = np.zeros(Y.shape)
    #neg_mask[np.where(Y_flipped < 0)] = 1. 
    #pmask = theano.tensor.matrix('pmask')
    #nmask = theano.tensor.matrix('nmask')
    
    factor = float(pos_loc_num) / float(neg_loc_num)
    WH = theano.tensor.dot(W,H)
    logit_pos = (1.-prob) * theano.tensor.gt(tL,0) * theano.tensor.log(1. + theano.tensor.exp(-1.* WH))
    logit_neg = prob * theano.tensor.lt(tL,0) * theano.tensor.log(1. + theano.tensor.exp(WH))
    #logit_pos = (1.-prob) * theano.tensor.log(1. + theano.tensor.exp(-1.* theano.tensor.dot(W,H)))
    #logit_neg = prob * theano.tensor.log(1. + theano.tensor.exp(theano.tensor.dot(W,H)))
    logit_loss = logit_pos.mean() + factor * logit_neg.mean() + lambda0 * ((W * W).mean() + (H * H).mean()) 
    
    nbatch = Y.shape[0]
    
    downhill.minimize(
            loss= logit_loss,
            params = [W,H],
            train = [Y_flipped],
            inputs = [tL],
            patience=0,
            algo='adagrad',
            batch_size= nbatch,
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.01)
            
    return W.get_value(), H.get_value()
 
train_file = open('Mediamill_data.txt','r')
train_file_lines = train_file.readlines(100000000000000000)
train_file.close()
train_fea = np.zeros((43907,120),dtype=float)
train_label = np.zeros((43907,101),dtype=int)
for k in range(1,len(train_file_lines)):
    data_segs = train_file_lines[k].split(' ')
    label_line = data_segs[0]
    labels = label_line.split(',')
    if (len(labels) == 0) or (labels[0] == ''):
        train_label[k-1,0] = 0
    else:
        for i in range(len(labels)):
            train_label[k-1,int(labels[i])-1] = 1

    for i in range(1,len(data_segs)):
        fea_pair = data_segs[i].split(':')
        fea_idx = int(fea_pair[0])
        fea_val = float(fea_pair[1])
        train_fea[k-1,fea_idx] = fea_val

fraction = []

for i in range(train_label.shape[1]):
    single_label = train_label[:,i]
    fraction.append(np.where(single_label > 0)[0].shape[0] / float(len(single_label)))

sort_fraction = np.argsort(-1*np.array(fraction))
train_label_sub = train_label[:,sort_fraction[0:20]]

fea_fraction = 0.6
label_fraction = 0.8
fea_mask = np.random.random(train_fea.shape)
fea_loc = np.where(fea_mask < fea_fraction) ### indexes of the unobserved entries 

pos_entries = np.where(train_label_sub == 1)
pos_ind = np.array(range(len(pos_entries[0])))
np.random.shuffle(pos_ind)
labelled_ind = pos_ind[0:int(float(len(pos_ind))*(1-label_fraction))] # 20% of 1s are preserved 
labelled_mask = np.zeros(train_label_sub.shape)
for i in labelled_ind:
    labelled_mask[pos_entries[0][i],pos_entries[1][i]] = 1

label_loc = np.where(labelled_mask == 0) #### label_loc: observed entries 
train_label_masked = train_label_sub.copy()
train_label_masked[label_loc] = 0. #### weak label assignments

auc_score_test_gridsearch = []
auc_score_train_gridsearch = []
nrank = 10

for prob in [0.0001,0.001,0.01,0.1,0.2]: # 3
    for lambda0 in [0.001,0.005,0.01,0.05,0.1,0.5,1]: #7
        for lambda1 in [0.5,1,5,10,15,20,30]: #7 
            for lambda2 in [0.5,1,5,10,15,20,30]: # 7
                for lambda3 in [0.5,1,5,10,15,20,25,30]:#8
                    U,V,W,H,M,N = LogitCoupledMFInductiveSim(train_data,train_label_masked,fea_loc,prob,nrank,lambda0,lambda1,lambda2,lambda3)
                    WH = np.dot(W,H)
                    score_test = np.ndarray.flatten(np.dot(test_data,WH))
                    score_test = 1./(1. + np.exp(-1*score_test))
                    ground_truth_test = np.ndarray.flatten(test_label)
                    auc_score_test_logit = roc_auc_score(ground_truth_test,score_test)
                    print str(prob) + ' ' + str(lambda0) + ' ' + str(lambda1) + ' ' + str(lambda2) + ' ' + str(lambda3) + ' ' + str(auc_score_test_logit)
                    auc_score_test_gridsearch.append(auc_score_test_logit)

                    UV = np.dot(U,V)
                    MN = np.dot(M,N)
                    score_train = np.ndarray.flatten(np.dot(UV,WH)+MN)
                    score_train = 1./(1. + np.exp(-1*score_train))
                    ground_truth_train = np.ndarray.flatten(train_label)
                    auc_score_train_logit = roc_auc_score(ground_truth_train,score_train)
                    print str(prob) + ' ' + str(lambda0) + ' ' + str(lambda1) + ' ' + str(lambda2) + ' ' + str(lambda3) + ' ' + str(auc_score_train_logit)
                    auc_score_train_gridsearch.append(auc_score_train_logit)


parameters = []
for prob in [0.0001,0.001,0.01,0.1,0.2]: # 3
    for lambda0 in [0.001,0.005,0.01,0.05,0.1,0.5,1]: #7
        for lambda1 in [0.5,1,5,10,15,20,30]: #7 
            for lambda2 in [0.5,1,5,10,15,20,30]: # 7
                for lambda3 in [0.5,1,5,10,15,20,25,30]:#8
                    parameters.append((prob,lambda0,lambda1,lambda2,lambda3))

import pickle
with open('mediamill_inductive_grid_search_08.pickle','wb') as f:
       pickle.dump([parameters,auc_score_train_gridsearch,auc_score_test_gridsearch],f)

fea_fraction = 0.6
label_fraction = 0.5
fea_mask = np.random.random(train_fea.shape)
fea_loc = np.where(fea_mask < fea_fraction) ### indexes of the unobserved entries 

pos_entries = np.where(train_label_sub == 1)
pos_ind = np.array(range(len(pos_entries[0])))
np.random.shuffle(pos_ind)
labelled_ind = pos_ind[0:int(float(len(pos_ind))*(1-label_fraction))] # 20% of 1s are preserved 
labelled_mask = np.zeros(train_label_sub.shape)
for i in labelled_ind:
    labelled_mask[pos_entries[0][i],pos_entries[1][i]] = 1

label_loc = np.where(labelled_mask == 0) #### label_loc: observed entries 
train_label_masked = train_label_sub.copy()
train_label_masked[label_loc] = 0. #### weak label assignments

auc_score_test_gridsearch = []
auc_score_train_gridsearch = []
nrank = 10

for prob in [0.0001,0.001,0.01,0.1,0.2]: # 3
    for lambda0 in [0.001,0.005,0.01,0.05,0.1,0.5,1]: #7
        for lambda1 in [0.5,1,5,10,15,20,30]: #7 
            for lambda2 in [0.5,1,5,10,15,20,30]: # 7
                for lambda3 in [0.5,1,5,10,15,20,25,30]:#8
                    U,V,W,H,M,N = LogitCoupledMFInductiveSim(train_data,train_label_masked,fea_loc,prob,nrank,lambda0,lambda1,lambda2,lambda3)
                    WH = np.dot(W,H)
                    score_test = np.ndarray.flatten(np.dot(test_data,WH))
                    score_test = 1./(1. + np.exp(-1*score_test))
                    ground_truth_test = np.ndarray.flatten(test_label)
                    auc_score_test_logit = roc_auc_score(ground_truth_test,score_test)
                    print str(prob) + ' ' + str(lambda0) + ' ' + str(lambda1) + ' ' + str(lambda2) + ' ' + str(lambda3) + ' ' + str(auc_score_test_logit)
                    auc_score_test_gridsearch.append(auc_score_test_logit)

                    UV = np.dot(U,V)
                    MN = np.dot(M,N)
                    score_train = np.ndarray.flatten(np.dot(UV,WH)+MN)
                    score_train = 1./(1. + np.exp(-1*score_train))
                    ground_truth_train = np.ndarray.flatten(train_label)
                    auc_score_train_logit = roc_auc_score(ground_truth_train,score_train)
                    print str(prob) + ' ' + str(lambda0) + ' ' + str(lambda1) + ' ' + str(lambda2) + ' ' + str(lambda3) + ' ' + str(auc_score_train_logit)
                    auc_score_train_gridsearch.append(auc_score_train_logit)


parameters = []
for prob in [0.0001,0.001,0.01,0.1,0.2]: # 3
    for lambda0 in [0.001,0.005,0.01,0.05,0.1,0.5,1]: #7
        for lambda1 in [0.5,1,5,10,15,20,30]: #7 
            for lambda2 in [0.5,1,5,10,15,20,30]: # 7
                for lambda3 in [0.5,1,5,10,15,20,25,30]:#8
                    parameters.append((prob,lambda0,lambda1,lambda2,lambda3))

import pickle
with open('mediamill_inductive_grid_search_05.pickle','wb') as f:
       pickle.dump([parameters,auc_score_train_gridsearch,auc_score_test_gridsearch],f)
       
fea_fraction = 0.6
label_fraction = 0.3
fea_mask = np.random.random(train_fea.shape)
fea_loc = np.where(fea_mask < fea_fraction) ### indexes of the unobserved entries 

pos_entries = np.where(train_label_sub == 1)
pos_ind = np.array(range(len(pos_entries[0])))
np.random.shuffle(pos_ind)
labelled_ind = pos_ind[0:int(float(len(pos_ind))*(1-label_fraction))] # 20% of 1s are preserved 
labelled_mask = np.zeros(train_label_sub.shape)
for i in labelled_ind:
    labelled_mask[pos_entries[0][i],pos_entries[1][i]] = 1

label_loc = np.where(labelled_mask == 0) #### label_loc: observed entries 
train_label_masked = train_label_sub.copy()
train_label_masked[label_loc] = 0. #### weak label assignments

auc_score_test_gridsearch = []
auc_score_train_gridsearch = []
nrank = 10

for prob in [0.0001,0.001,0.01,0.1,0.2]: # 3
    for lambda0 in [0.001,0.005,0.01,0.05,0.1,0.5,1]: #7
        for lambda1 in [0.5,1,5,10,15,20,30]: #7 
            for lambda2 in [0.5,1,5,10,15,20,30]: # 7
                for lambda3 in [0.5,1,5,10,15,20,25,30]:#8
                    U,V,W,H,M,N = LogitCoupledMFInductiveSim(train_data,train_label_masked,fea_loc,prob,nrank,lambda0,lambda1,lambda2,lambda3)
                    WH = np.dot(W,H)
                    score_test = np.ndarray.flatten(np.dot(test_data,WH))
                    score_test = 1./(1. + np.exp(-1*score_test))
                    ground_truth_test = np.ndarray.flatten(test_label)
                    auc_score_test_logit = roc_auc_score(ground_truth_test,score_test)
                    print str(prob) + ' ' + str(lambda0) + ' ' + str(lambda1) + ' ' + str(lambda2) + ' ' + str(lambda3) + ' ' + str(auc_score_test_logit)
                    auc_score_test_gridsearch.append(auc_score_test_logit)

                    UV = np.dot(U,V)
                    MN = np.dot(M,N)
                    score_train = np.ndarray.flatten(np.dot(UV,WH)+MN)
                    score_train = 1./(1. + np.exp(-1*score_train))
                    ground_truth_train = np.ndarray.flatten(train_label)
                    auc_score_train_logit = roc_auc_score(ground_truth_train,score_train)
                    print str(prob) + ' ' + str(lambda0) + ' ' + str(lambda1) + ' ' + str(lambda2) + ' ' + str(lambda3) + ' ' + str(auc_score_train_logit)
                    auc_score_train_gridsearch.append(auc_score_train_logit)


parameters = []
for prob in [0.0001,0.001,0.01,0.1,0.2]: # 3
    for lambda0 in [0.001,0.005,0.01,0.05,0.1,0.5,1]: #7
        for lambda1 in [0.5,1,5,10,15,20,30]: #7 
            for lambda2 in [0.5,1,5,10,15,20,30]: # 7
                for lambda3 in [0.5,1,5,10,15,20,25,30]:#8
                    parameters.append((prob,lambda0,lambda1,lambda2,lambda3))

import pickle
with open('mediamill_inductive_grid_search_03.pickle','wb') as f:
       pickle.dump([parameters,auc_score_train_gridsearch,auc_score_test_gridsearch],f)