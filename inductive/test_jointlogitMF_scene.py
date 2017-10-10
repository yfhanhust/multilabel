import numpy as np 
import scipy as sp 
from sklearn.linear_model import Ridge 
import downhill
import theano
from sklearn.metrics import roc_auc_score

def LogitCoupledMFInductive(X,Y,fea_loc,lambda0,lambda1,lambda2,lambda3,lambda4,prob,nrank,trade_off):
    
    mask = np.ones(X.shape)
    mask[fea_loc] = 0.
    
    ######### label matrix factorization using reweighted logit loss 
    W = theano.shared(np.random.random((Y.shape[0],nrank)),name='W')
    H = theano.shared(np.random.random((nrank,Y.shape[1])),name='H')
    
    pos_loc_num = len(np.where(Y > 0)[0])
    neg_loc_num = len(np.where(Y < 1)[0])
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
    logit_neg = prob * theano.tensor.lt(tL,1) * theano.tensor.log(1. + theano.tensor.exp(WH))
    #logit_pos = (1.-prob) * theano.tensor.log(1. + theano.tensor.exp(-1.* theano.tensor.dot(W,H)))
    #logit_neg = prob * theano.tensor.log(1. + theano.tensor.exp(theano.tensor.dot(W,H)))
    logit_loss = logit_pos.mean() + factor * logit_neg.mean() + lambda0 * ((W * W).mean() + (H * H).mean())
    
    ########### feature matrix factorization
    tX = theano.tensor.matrix(name='X')
    feature_mask = theano.tensor.matrix('feature_mask')
    U = theano.shared(np.random.random((X.shape[0],nrank)),name='U')
    V = theano.shared(np.random.random((nrank,X.shape[1])),name='V')
    M = theano.shared(np.random.random((X.shape[1],Y.shape[1])),name='M')
    S = theano.shared(np.random.random((X.shape[0],X.shape[1])),name='S') 
    sloss = theano.tensor.sqr((tX - S) * feature_mask).mean()
    lrloss = theano.tensor.sqr((S - theano.tensor.dot(U,V))).mean()
    penaltyloss = (U * U).mean() + (V * V).mean()
    regloss = theano.tensor.sqr((theano.tensor.dot(S,M)-WH)).mean()
    penaltylossM = (M * M).mean()
    lse_loss = sloss + lambda1 * lrloss + lambda2 * penaltyloss + lambda3 * regloss + lambda4 * penaltylossM
    
    global_loss = lse_loss + trade_off * logit_loss 
    nbatch = Y.shape[0]
    
    downhill.minimize(
            loss= global_loss,
            params = [U,V,W,H,M,S],
            train = [X,Y,mask],
            inputs = [tX,tL,feature_mask],
            patience=0,
            algo='adagrad',
            batch_size= nbatch,
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.001)
                
    return U.get_value(),V.get_value(),W.get_value(),H.get_value(),M.get_value(),S.get_value()


def LogitMFInductive(X,Y,fea_loc,lambda0,prob,nrank,value):
    #M = theano.tensor.matrix('Feature')
    
    W = theano.shared(np.random.random((X.shape[1],nrank)),name='W')
    H = theano.shared(np.random.random((nrank,Y.shape[1])),name='H')
    
    
    Y_flipped = np.copy(Y)
    Y_flipped[np.where(Y < 1)] = -1
    X_masked = np.copy(X)
    X_masked[fea_loc] = value
    pos_loc_num = len(np.where(Y_flipped > 0)[0])
    neg_loc_num = len(np.where(Y_flipped < 0)[0])
    tL = theano.tensor.matrix(name='Y')
    tX = theano.tensor.matrix(name='X')
    #pos_mask = np.zeros(Y.shape)
    #pos_mask[np.where(Y_flipped > 0)] = 1.
    #neg_mask = np.zeros(Y.shape)
    #neg_mask[np.where(Y_flipped < 0)] = 1. 
    #pmask = theano.tensor.matrix('pmask')
    #nmask = theano.tensor.matrix('nmask')
    
    factor = float(pos_loc_num) / float(neg_loc_num)
    WH = theano.tensor.dot(W,H)
    XWH = theano.tensor.dot(tX,WH)
    logit_pos = (1.-prob) * theano.tensor.gt(tL,0) * theano.tensor.log(1. + theano.tensor.exp(-1.* XWH))
    logit_neg = prob * theano.tensor.lt(tL,0) * theano.tensor.log(1. + theano.tensor.exp(XWH))
    #logit_pos = (1.-prob) * theano.tensor.log(1. + theano.tensor.exp(-1.* theano.tensor.dot(W,H)))
    #logit_neg = prob * theano.tensor.log(1. + theano.tensor.exp(theano.tensor.dot(W,H)))
    logit_loss = logit_pos.mean() + factor * logit_neg.mean() + lambda0 * ((W * W).mean() + (H * H).mean()) 
    
    nbatch = Y.shape[0]
    
    downhill.minimize(
            loss= logit_loss,
            params = [W,H],
            train = [X_masked,Y_flipped],
            inputs = [tX,tL],
            patience=0,
            algo='adagrad',
            batch_size= nbatch,
            max_gradient_norm=1,
            learning_rate=0.05,
            min_improvement = 0.01)
            
    return W.get_value(), H.get_value()

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
            
    return W.get_value(), H.get_value()#
    
    
### Fast randomized SingularValue Thresholding for Low-rank Optimization
#### generate data
#### yeast: classes 14, data: 1500+917, dimensionality: 103
#### generate data
#### yeast: classes 14, data: 1500+917, dimensionality: 103
train_file = open('scene_train','r')
train_file_lines = train_file.readlines(100000000000000)
train_file.close()
train_fea = np.zeros((1211,294),dtype=float)
train_label = np.zeros((1211,6),dtype=int)
for k in range(1,len(train_file_lines)):
    data_segs = train_file_lines[k].split(' ')
    label_line = data_segs[0]
    labels = label_line.split(',')
    if (len(labels) == 0) or (labels[0] == ''):
        train_label[k-1,0] = 0
    else:
        for i in range(len(labels)):
            train_label[k-1,int(labels[i])-1] = 1

    for i in range(1,len(data_segs)-1):
        fea_pair = data_segs[i].split(':')
        fea_idx = int(fea_pair[0])
        fea_val = float(fea_pair[1])
        train_fea[k-1,fea_idx-1] = fea_val

#### yeast: classes 14, data: 917, dimensions: 103
test_file = open('scene_test','r')
test_file_lines = test_file.readlines(100000000000000)
test_file.close()
test_fea = np.zeros((1196,294),dtype=float)
test_label = np.zeros((1196,6),dtype=int)
for k in range(1,len(test_file_lines)):
    data_segs = test_file_lines[k].split(' ')
    label_line = data_segs[0]
    labels = label_line.split(',')
    if (len(labels) == 0) or (labels[0] == ''):
        test_label[k-1,0] = 0
    else:
        for i in range(len(labels)):
            test_label[k-1,int(labels[i])-1] = 1

    for i in range(1,len(data_segs)-1):
        fea_pair = data_segs[i].split(':')
        fea_idx = int(fea_pair[0])
        fea_val = float(fea_pair[1])
        test_fea[k-1,fea_idx-1] = fea_val

scene_data = np.concatenate((train_fea,test_fea))
scene_label = np.concatenate((train_label,test_label))


nsample = scene_data.shape[0]
ndim = scene_data.shape[1]
######### linear model ##############
fea_fraction = 0.6
label_fraction = 0.8
ind_scene_data = np.array(range(scene_data.shape[0]))
test_auc_score = []
nsample = scene_data.shape[0]
num_train = int(nsample * 0.8)
num_test = nsample - num_train
np.random.shuffle(ind_scene_data)
train_data = scene_data[ind_scene_data[0:num_train],:]
test_data = scene_data[ind_scene_data[num_train:],:]
train_label = scene_label[ind_scene_data[0:num_train],:]
test_label = scene_label[ind_scene_data[num_train:],:]

fea_mask = np.random.random(train_data.shape)
fea_loc = np.where(fea_mask < (1.-fea_fraction)) ### indexes of the observed entries 

pos_entries = np.where(train_label == 1)
pos_ind = np.array(range(len(pos_entries[0])))
np.random.shuffle(pos_ind)
labelled_ind = pos_ind[0:int(float(len(pos_ind))*(1-label_fraction))] # 20% of 1s are preserved 
labelled_mask = np.zeros(train_label.shape)
for i in labelled_ind:
    labelled_mask[pos_entries[0][i],pos_entries[1][i]] = 1

label_loc = np.where(labelled_mask == 1) #### label_loc: observed entries 
train_label_masked = train_label.copy()
train_label_masked[label_loc] = 0. #### weak label assignments  
##### test coupled matrix factorization
#lambda0 = 1e-2
prob = 0.2
#trade_off = 0.5
joint_para_list = []
joint_auc_list = []
for lambda0 in [0.0001,0.001,0.01,0.1,1,10]:
    for lambda1 in [0.0001,0.001,0.01,0.1,1,10]:
        for lambda2 in [0.0001,0.001,0.01,0.1,1,10]:
            for lambda3 in [0.0001,0.001,0.01,0.1,1,10]:
                for lambda4 in [0.0001,0.001,0.01,0.1,1,10]: ### 23328 parameter combinations 
                    for trade_off in [0.1,1,10]:
                        U,V,W,H,M,S = LogitCoupledMFInductive(train_data,train_label_masked,fea_loc,lambda0,lambda1,lambda2,lambda3,lambda4,prob,10,trade_off)
                        score_test = np.ndarray.flatten(np.dot(test_data,M))
                        ground_truth_test = np.ndarray.flatten(test_label)
                        auc_score_test_logit = roc_auc_score(ground_truth_test,score_test)
                        joint_auc_list.append(auc_score_test_logit)
                        joint_para_list.append((lambda0,lambda1,lambda2,lambda3,lambda4,trade_off))
                        print auc_score_test_logit


max_auc_score = np.max(np.array(joint_auc_list))
optimal_parameter = np.array(joint_para_list)[np.where(np.array(joint_auc_list) == max_auc_score)[0]]
lambda0 = optimal_parameter[0]
lambda1 = optimal_parameter[1]
lambda2 = optimal_parameter[2]
lambda3 = optimal_parameter[3]
lambda4 = optimal_parameter[4]
trade_off = optimal_parameter[5]
test_result_score_opt = []
for train_split in range(10):
    num_train = int(nsample * 0.8)
    num_test = nsample - num_train
    np.random.shuffle(ind_scene_data)
    train_data = scene_data[ind_scene_data[0:num_train],:]
    test_data = scene_data[ind_scene_data[num_train:],:]
    train_label = scene_label[ind_scene_data[0:num_train],:]
    test_label = scene_label[ind_scene_data[num_train:],:]
    for mask_split in range(10):
        fea_mask = np.random.random(train_data.shape)
        fea_loc = np.where(fea_mask < (1.-fea_fraction)) ### indexes of the observed entries 

        pos_entries = np.where(train_label == 1)
        pos_ind = np.array(range(len(pos_entries[0])))
        np.random.shuffle(pos_ind)
        labelled_ind = pos_ind[0:int(float(len(pos_ind))*(1-label_fraction))] # 20% of 1s are preserved 
        labelled_mask = np.zeros(train_label.shape)
        for i in labelled_ind:
            labelled_mask[pos_entries[0][i],pos_entries[1][i]] = 1

        label_loc = np.where(labelled_mask == 1) #### label_loc: observed entries 
        train_label_masked = train_label.copy()
        train_label_masked[label_loc] = 0. #### weak label assignments
    
    U,V,W,H,M,S = LogitCoupledMFInductive(train_data,train_label_masked,fea_loc,lambda0,lambda1,lambda2,lambda3,lambda4,prob,10,trade_off)
    score_test = np.ndarray.flatten(np.dot(test_data,M))
    ground_truth_test = np.ndarray.flatten(test_label)
    auc_score_test_logit = roc_auc_score(ground_truth_test,score_test)
    test_result_score_opt.append(auc_score_test_logit)
    print auc_score_test_logit