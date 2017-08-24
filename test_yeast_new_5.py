import numpy as np 
import scipy as sp 
from sklearn.linear_model import Ridge 
import downhill
import theano
from sklearn.metrics import roc_auc_score

def completionLR(X,kx,fea_loc,lambdaU):
    mask = np.ones(X.shape)
    mask[fea_loc] = 0.

    #### Theano and downhill
    U = theano.shared(np.random.random((X.shape[0],kx)),name='U')
    V = theano.shared(np.random.random((X.shape[1],kx)),name='V')

    X_symbolic = theano.tensor.matrix(name="X", dtype=X.dtype)
    reconstruction = theano.tensor.dot(U, V.T)
    difference = X_symbolic - reconstruction
    masked_difference = difference * mask
    err = theano.tensor.sqr(masked_difference)
    mse = err.mean()
    xloss = mse + lambdaU/2. * (U * U).mean() + lambdaU/2. * (V * V).mean()

    #### optimisation
    downhill.minimize(
            loss= xloss,
            train = [X],
            patience=0,
            algo='rmsprop',
            batch_size=X.shape[0],
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.001)

    return U.get_value(),V.get_value()
    
    
def baselinePU(Y,label_loc,alpha,vlambda,kx):

    #random_mat = np.random.random(Y.shape)
    #label_loc = np.where(random_mat < label_fraction) ## locate the masked entries in the label matrix
    #### print statistics
    #print np.where(Y[label_loc] > 0)[0].shape[0] / float(np.where(Y > 0)[0].shape[0]) ## the ratio of "1" entries being masked
    #print np.where(Y[label_loc] < 1)[0].shape[0] / float(np.where(Y < 1)[0].shape[0]) ## the ratio of "0" entries being masked
    W = theano.shared(np.random.random((Y.shape[0],kx)),name='W')
    H = theano.shared(np.random.random((Y.shape[1],kx)),name='H')

    labelmask = np.ones(Y.shape)
    #labelmask[np.where(Y > 0)] = 1
    labelmask[label_loc] = 0.
    label_mask = theano.tensor.matrix('label_mask')
    
    reconstruction = theano.tensor.dot(W, H.T)
    tY = theano.tensor.matrix(name="Y", dtype=Y.dtype)
    difference = theano.tensor.sqr((tY - reconstruction)) * (1 - alpha)
    positive_difference = theano.tensor.sqr((tY - reconstruction) * label_mask) * (2*alpha-1.)

    mse = difference.mean() + positive_difference.mean()
    loss = mse + (vlambda) * (W * W).mean() + (vlambda) * (H * H).mean()

    downhill.minimize(
            loss=loss,
            params = [W,H],
            inputs = [tY,label_mask],
            train=[Y,labelmask],
            patience=0,
            algo='rmsprop',
            batch_size=Y.shape[0],
            max_gradient_norm=1,
            learning_rate=0.06,
            min_improvement = 0.0001)

    return W.get_value(),H.get_value()

def completionPUV(X,Y,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx):

    #delta = 0.3
    ### masking out some entries from feature and label matrix

    mask = np.ones(X.shape)
    mask[fea_loc] = 0.
    labelmask = np.zeros(Y.shape)
    labelmask[np.where(Y > 0)] = 1
    #### Theano and downhill
    #### declare variables shared between symbolic and non-symbolic computing thread
    U = theano.shared(np.random.random((X.shape[0],kx)),name='U')
    V = theano.shared(np.random.random((X.shape[1],kx)),name='V')
    W = theano.shared(np.random.random((Y.shape[0],kx)),name='W')
    H = theano.shared(np.random.random((Y.shape[1],kx)),name='H')
    #### declare symbolic variables 
    feature_mask = theano.tensor.matrix('feature_mask')
    label_mask = theano.tensor.matrix('label_mask')
    #feature_mask = theano.shared(mask,name='mask')
    #label_mask = theano.shared(labelmask,name='labelmask')
    #### U,V,W and H randomly initialised 
    
    nsample = X.shape[0]
    #X_symbolic = theano.tensor.matrix(name="X", dtype=X.dtype)
    #tX = theano.shared(X.astype(theano.config.floatX),name="X")
    #tX = theano.shared(X,name="X")
    tX = theano.tensor.matrix('X') ### symbolic variable 
    difference = tX - theano.tensor.dot(U, V.T)
    masked_difference = difference * feature_mask
    err = theano.tensor.sqr(masked_difference)
    mse = err.mean()
    xloss = mse + lambda0 * ((U * U).mean() + (V * V).mean())

    #tY = theano.shared(Y.astype(theano.config.floatX),name="Y")
    #tY = theano.shared(Y,name="Y")
    tY = theano.tensor.matrix('Y') ### symbolic variable 
    Y_reconstruction = theano.tensor.dot(W, H.T)
    Ydifference = theano.tensor.sqr((tY - Y_reconstruction)) * (1 - alpha)
    positive_difference = theano.tensor.sqr((tY - Y_reconstruction) * label_mask) * (2*alpha-1.)
    Ymse = Ydifference.mean() + positive_difference.mean()
    global_loss = xloss + delta * Ymse + lambda1 * ((W * W).mean() + (H * H).mean()) + lambda2 * theano.tensor.sqr((U-W)).mean()

    #### optimisation
    downhill.minimize(
            loss= global_loss,
            params = [U,V,W,H],
            train = [X,Y,mask,labelmask],
            inputs = [tX,tY,feature_mask,label_mask],
            patience=0,
            algo='rmsprop',
            batch_size=nsample,
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.0001)
    
    return U.get_value(),V.get_value(),W.get_value(),H.get_value()


def completionPUVReg(X,Y,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx,ky):

    #delta = 0.3
    ### masking out some entries from feature and label matrix
    mask = np.ones(X.shape)
    mask[fea_loc] = 0.
    labelmask = np.ones(Y.shape)
    #labelmask[np.where(Y > 0)] = 1
    labelmask[label_loc] = 0
    #### Theano and downhill
    #### declare variables shared between symbolic and non-symbolic computing thread
    U = theano.shared(np.random.random((X.shape[0],kx)),name='U')
    V = theano.shared(np.random.random((X.shape[1],kx)),name='V')
    W = theano.shared(np.random.random((kx,ky)),name='W')
    H = theano.shared(np.random.random((Y.shape[1],ky)),name='H')
    #### declare symbolic variables 
    feature_mask = theano.tensor.matrix('feature_mask')
    label_mask = theano.tensor.matrix('label_mask')
    #feature_mask = theano.shared(mask,name='mask')
    #label_mask = theano.shared(labelmask,name='labelmask')
    #### U,V,W and H randomly initialised 
    nsample = X.shape[0]
    #X_symbolic = theano.tensor.matrix(name="X", dtype=X.dtype)
    #tX = theano.shared(X.astype(theano.config.floatX),name="X")
    #tX = theano.shared(X,name="X")
    tX = theano.tensor.matrix('X') ### symbolic variable
    UVT = theano.tensor.dot(U,V.T) 
    difference = tX - UVT
    masked_difference = difference * feature_mask
    err = theano.tensor.sqr(masked_difference)
    mse = err.mean()
    xloss = mse + lambda0 * (U * U).mean() + lambda1 * (V * V).mean()

    #tY = theano.shared(Y.astype(theano.config.floatX),name="Y")
    #tY = theano.shared(Y,name="Y")
    tY = theano.tensor.matrix('Y') ### symbolic variable 
    WHT = theano.tensor.dot(W, H.T)
    UWHT = theano.tensor.dot(U,WHT)
    #Y_reconstruction = theano.tensor.dot(U,WHT)
    Ydifference = theano.tensor.sqr((tY - UWHT)) * (1 - alpha)
    positive_difference = theano.tensor.sqr((tY - UWHT) * label_mask) * (2*alpha-1.)
    Ymse = Ydifference.mean() + positive_difference.mean()
    #UVTM = theano.tensor.dot(UVT,M) 
    global_loss = xloss + delta * Ymse + lambda2 * ((W * W).mean() + (H * H).mean()) 
    
    #### optimisation
    downhill.minimize(
            loss= global_loss,
            params = [U,V,W,H],
            train = [X,Y,mask,labelmask],
            inputs = [tX,tY,feature_mask,label_mask],
            patience=0,
            algo='rmsprop',
            batch_size=nsample,
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.001)
    
    return U.get_value(),V.get_value(),W.get_value(),H.get_value()

    
def completionPUVReg1(X,Y,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,lambda3,delta,kx):

    #delta = 0.3
    ### masking out some entries from feature and label matrix
    mask = np.ones(X.shape)
    mask[fea_loc] = 0.
    labelmask = np.ones(Y.shape)
    #labelmask[np.where(Y > 0)] = 1
    labelmask[label_loc] = 0
    #### Theano and downhill
    #### declare variables shared between symbolic and non-symbolic computing thread
    U = theano.shared(np.random.random((X.shape[0],kx)),name='U')
    V = theano.shared(np.random.random((X.shape[1],kx)),name='V')
    W = theano.shared(np.random.random((X.shape[0],kx)),name='W')
    H = theano.shared(np.random.random((Y.shape[1],kx)),name='H')
    M = theano.shared(np.random.random((kx,kx)),name='M')
    #### declare symbolic variables 
    feature_mask = theano.tensor.matrix('feature_mask')
    label_mask = theano.tensor.matrix('label_mask')
    #feature_mask = theano.shared(mask,name='mask')
    #label_mask = theano.shared(labelmask,name='labelmask')
    #### U,V,W and H randomly initialised 
    nsample = X.shape[0]
    #X_symbolic = theano.tensor.matrix(name="X", dtype=X.dtype)
    #tX = theano.shared(X.astype(theano.config.floatX),name="X")
    #tX = theano.shared(X,name="X")
    tX = theano.tensor.matrix('X') ### symbolic variable
    UVT = theano.tensor.dot(U,V.T) 
    difference = tX - UVT
    masked_difference = difference * feature_mask
    err = theano.tensor.sqr(masked_difference)
    mse = err.mean()
    xloss = mse + lambda0 * ((U * U).mean() + (V * V).mean())

    #tY = theano.shared(Y.astype(theano.config.floatX),name="Y")
    #tY = theano.shared(Y,name="Y")
    tY = theano.tensor.matrix('Y') ### symbolic variable 
    WHT = theano.tensor.dot(W, H.T)
    #Y_reconstruction = theano.tensor.dot(U,WHT)
    Ydifference = theano.tensor.sqr((tY - WHT)) * (1 - alpha)
    positive_difference = theano.tensor.sqr((tY - WHT) * label_mask) * (2*alpha-1.)
    Ymse = Ydifference.mean() + positive_difference.mean()
    UM = theano.tensor.dot(U,M) 
    global_loss = xloss + delta * Ymse + lambda1 * ((W * W).mean() + (H * H).mean()) + lambda2 * theano.tensor.sqr((UM - W)).mean() + lambda3 * (M * M).mean()
    
    #### optimisation
    downhill.minimize(
            loss= global_loss,
            params = [U,V,W,H,M],
            train = [X,Y,mask,labelmask],
            inputs = [tX,tY,feature_mask,label_mask],
            patience=0,
            algo='rmsprop',
            batch_size=nsample,
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.0001)
    
    return U.get_value(),V.get_value(),W.get_value(),H.get_value(),M.get_value()

    
def acc_label(Y,W,H,label_loc):
    Y_reconstructed = np.dot(W,H.T)
    ground_truth = Y[label_loc].tolist()
    reconstruction = Y_reconstructed[label_loc].tolist()
    auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
    return auc_score

def acc_feature(X,U,V,fea_loc):
    X_reconstruction = U.dot(V.T)
    return np.linalg.norm(X[fea_loc] - X_reconstruction[fea_loc])    
    


train_file = open('yeast_train.svm','r')
train_file_lines = train_file.readlines(100000000000000)
train_file.close()
train_fea = np.zeros((1500,103),dtype=float)
train_label = np.zeros((1500,14),dtype=int)
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
test_file = open('yeast_test.svm','r')
test_file_lines = test_file.readlines(100000000000000)
test_file.close()
test_fea = np.zeros((917,103),dtype=float)
test_label = np.zeros((917,14),dtype=int)
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

yeast_data = np.concatenate((train_fea,test_fea))
yeast_label = np.concatenate((train_label,test_label))
train_fea = yeast_data
train_label = yeast_label


kx = 10
alpha = (1. + 0.6)/2 ### insensitive parameter in PU matrix completion
fea_fraction = 0.6
label_fraction = 0.5#0.8

#### testing baseline PU 
pos_entries = np.where(train_label == 1)
pos_ind = np.array(range(len(pos_entries[0])))
baseline_acc_list = []
for lambda0 in [0.001,0.005,0.01,0.05,0.1,0.5,1.,1.5,5,10]:
    baseline_acc_oneround = []
    for iround in range(15):
        np.random.shuffle(pos_ind)
        labelled_ind = pos_ind[0:int(float(len(pos_ind))*(1-label_fraction))] # 30% of 1s are preserved 
        labelled_mask = np.zeros(train_label.shape)
        for i in labelled_ind:
            labelled_mask[pos_entries[0][i],pos_entries[1][i]] = 1

        label_loc = np.where(labelled_mask == 0)
        train_label_masked = train_label.copy()
        train_label_masked[label_loc] = 0. ### generate a positive-unlabelled matrix
        W_pu,H_pu = baselinePU(train_label_masked,label_loc,alpha,lambda0,kx)
        Y_reconstructed = np.dot(W_pu,H_pu.T)
        ground_truth = train_label[label_loc].tolist()
        reconstruction = Y_reconstructed[label_loc].tolist()
        auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
        baseline_acc_oneround.append(auc_score)
    
    baseline_acc_list.append(baseline_acc_oneround)

mean_acc_list = []
std_acc_list = []
for i in range(len(baseline_acc_list)):
    mean_acc_list.append(np.mean(np.array(baseline_acc_list[i])))
    std_acc_list.append(np.std(np.array(baseline_acc_list[i])))

#### testing baseline regression error 
baseline_reg_list = []
for lambda0 in [0.001,0.005,0.01,0.05,0.1,0.5,1.,1.5,5,10]:
    baseline_reg_oneround = []
    for iround in range(15):
        fea_mask = np.random.random(train_fea.shape)
        fea_loc = np.where(fea_mask < fea_fraction)
        U_lr, V_lr = completionLR(train_fea,kx,fea_loc,lambda0)
        reg_error = acc_feature(train_fea,U_lr,V_lr,fea_loc)
        baseline_reg_oneround.append(reg_error)
    
    baseline_reg_list.append(baseline_reg_oneround)

mean_reg_list = []
std_reg_list = []
for i in range(len(baseline_reg_list)):
    mean_reg_list.append(np.mean(np.array(baseline_reg_list[i])))
    std_reg_list.append(np.std(np.array(baseline_reg_list[i])))


import pickle 
result_file_name = 'yeast_baseline_test.pickle'
with open(result_file_name,'wb') as f:
    pickle.dump([baseline_acc_list,baseline_reg_list,mean_acc_list,std_acc_list,mean_reg_list,std_reg_list],f)


##### grid search 
#### grid search 
acc_list = []
para_list = []
###### testing our own method 
#kx = 5
#lambda0 = 0.01 ### U and V
#lambda1 = 0.1 ### W and H
#lambda2 = 0.1 ### regs for UM - W 
#lambda3 = 0.01#0.01 ### regs on M 
#delta = 0.1

#completionPUVReg1(X,Y,fea_loc,alpha,lambda0,lambda1,lambda2,lambda3,delta,kx)
#### grid search 
fea_mask = np.random.random(train_fea.shape)
fea_loc = np.where(fea_mask < fea_fraction)
np.random.shuffle(pos_ind)
labelled_ind = pos_ind[0:int(float(len(pos_ind))*(1-label_fraction))] # 30% of 1s are preserved 
labelled_mask = np.zeros(train_label.shape)
for i in labelled_ind:
    labelled_mask[pos_entries[0][i],pos_entries[1][i]] = 1

label_loc = np.where(labelled_mask == 0)
train_label_masked = train_label.copy()
train_label_masked[label_loc] = 0. ### generate a positive-unlabelled matrix

kx = 10 
alpha = (1. + 0.6)/2 ### insensitive parameter in PU matrix completion
acc_list = []
for lambda0 in [0.001,0.005,0.01,0.05,0.1,0.5]: #6
    for lambda1 in [0.0005,0.001,0.005,0.01,0.05,0.1,0.5]: #7
        for lambda2 in [0.001,0.005,0.01,0.05,0.1,0.5]: #6
            for delta in [0.005,0.01,0.05,0.1,0.5]: #5
                U,V,W,H = completionPUVReg(train_fea,train_label_masked,fea_loc,alpha,lambda0,lambda1,lambda2,delta,kx)
                Y_reconstructed = np.dot(U,np.dot(W,H.T))
                ground_truth = train_label[label_loc].tolist()
                reconstruction = Y_reconstructed[label_loc].tolist()
                auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
                print auc_score
                acc_list.append(auc_score)

para_list = []
for lambda0 in [0.001,0.005,0.01,0.05,0.1,0.5]: #4
    for lambda1 in [0.0005,0.001,0.005,0.01,0.05,0.1,0.5]: #4 
        for lambda2 in [0.001,0.005,0.01,0.05,0.1,0.5]: #3
            for delta in [0.005,0.01,0.05,0.1,0.5]: #2
                para_list.append((lambda0,lambda1,lambda2,delta))


ind = np.argsort(-1*np.array(acc_list))
print para_list[ind[0]]

