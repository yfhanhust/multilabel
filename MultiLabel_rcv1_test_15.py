import numpy as np
import scipy as sp
import downhill
import theano
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, accuracy_score, hamming_loss


def baselinePU(Y,label_loc,alpha,vlambda,kx):

    #random_mat = np.random.random(Y.shape)
    #label_loc = np.where(random_mat < label_fraction) ## locate the masked entries in the label matrix
    #### print statistics
    #print np.where(Y[label_loc] > 0)[0].shape[0] / float(np.where(Y > 0)[0].shape[0]) ## the ratio of "1" entries being masked
    #print np.where(Y[label_loc] < 1)[0].shape[0] / float(np.where(Y < 1)[0].shape[0]) ## the ratio of "0" entries being masked
    W = theano.shared(np.random.random((Y.shape[0],kx)),name='W')
    H = theano.shared(np.random.random((Y.shape[1],kx)),name='H')

    labelmask = np.ones(Y.shape)
    labelmask[label_loc] = 0
    Y_masked = Y.copy()
    Y_masked[label_loc] = 0

    reconstruction = theano.tensor.dot(W, H.T)
    X_symbolic = theano.tensor.matrix(name="Y_masked", dtype=Y_masked.dtype)
    difference = theano.tensor.sqr((X_symbolic - reconstruction)) * (1 - alpha)
    positive_difference = theano.tensor.sqr((X_symbolic - reconstruction) * labelmask) * (2*alpha-1.)

    mse = difference.mean() + positive_difference.mean()
    loss = mse + vlambda * (W * W).mean() + vlambda * (H * H).mean()

    downhill.minimize(
            loss=loss,
            train=[Y_masked],
            patience=0,
            algo='rmsprop',
            batch_size=Y_masked.shape[0],
            max_gradient_norm=1,
            learning_rate=0.06,
            min_improvement = 0.00001)

    return W.get_value(),H.get_value()
    
def acc_label(Y,W,H,label_loc):
    Y_reconstructed = np.dot(W,H.T)
    ground_truth = Y[label_loc].tolist()
    reconstruction = Y_reconstructed[label_loc].tolist()
    auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
    return auc_score

def acc_feature(X,U,V,fea_loc):
    X_reconstruction = U.dot(V.T)
    return np.linalg.norm(X[fea_loc] - X_reconstruction[fea_loc])

def completionLR(X,kx,fea_loc,lambdaU,lambdaV):
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
    xloss = mse + lambdaU * (U * U).mean() + lambdaV * (V * V).mean()

    #### optimisation
    downhill.minimize(
            loss= xloss,
            train = [X],
            patience=0,
            algo='rmsprop',
            batch_size=X.shape[0],
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.0001)

    return U.get_value(),V.get_value()

def completionPUV(X,Y,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx):

    #delta = 0.3
    ### masking out some entries from feature and label matrix

    mask = np.ones(X.shape)
    mask[fea_loc] = 0.
    labelmask = np.ones(Y.shape)
    labelmask[label_loc] = 0

    #### Theano and downhill
    U = theano.shared(np.random.random((X.shape[0],kx)),name='U')
    V = theano.shared(np.random.random((X.shape[1],kx)),name='V')
    W = theano.shared(np.random.random((Y.shape[0],kx)),name='W')
    H = theano.shared(np.random.random((Y.shape[1],kx)),name='H')

    X_symbolic = theano.tensor.matrix(name="X", dtype=X.dtype)
    reconstruction = theano.tensor.dot(U, V.T)
    difference = X_symbolic - reconstruction
    masked_difference = difference * mask
    err = theano.tensor.sqr(masked_difference)
    mse = err.mean()
    xloss = mse + lambda0 * ((U * U).mean() + (V * V).mean())

    Y_symbolic = theano.tensor.matrix(name="Y", dtype=Y.dtype)
    Y_reconstruction = theano.tensor.dot(U, H.T)
    Ydifference = theano.tensor.sqr((Y_symbolic - Y_reconstruction)) * (1 - alpha)
    positive_difference = theano.tensor.sqr((Y_symbolic - Y_reconstruction) * labelmask) * (2*alpha-1.)
    Ymse = Ydifference.mean() + positive_difference.mean()
    global_loss = xloss + delta * Ymse + lambda1 * ((W * W).mean() + (H * H).mean()) + lambda2 * theano.tensor.sqr((U-W)).mean()


    #### optimisation
    downhill.minimize(
            loss=global_loss,
            train = [X,Y],
            inputs = [X_symbolic,Y_symbolic],
            patience=0,
            algo='rmsprop',
            batch_size=Y.shape[0],
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.0001)

    return U.get_value(),V.get_value(),W.get_value(),H.get_value()

def TPAMI(X,Y,fea_loc_x,fea_loc_y,label_loc_x,label_loc_y,miu,lambda0,kx):
    ### X: feature matrix 
    ### Y: label matrix 
    ### fea_loc_x, fea_loc_y: masked entries in feature matrix 
    ### label_loc_x, label_loc_y: masked entries in label matrix 
    ### miu: regularisation parameter on matrix rank 
    ### lambda0: regularisation parameter on label reconstruction 
    ### kx: dimensionality of latent variables used for solving nuclear norm based regularisation 
    M = np.concatenate((Y,X),axis=1)
    M = M.T

    label_dim = Y.shape[1]
    fea_dim = X.shape[1]
    gamma = 15.
    featuremask = np.ones(M.shape)
    labelmask = np.ones(M.shape)
    for i in range(len(label_loc_x)):
        labelmask[label_loc_y[i],label_loc_x[i]] = 0.

    for i in range(len(fea_loc_x)):
        featuremask[fea_loc_y[i]+label_dim,fea_loc_x[i]] = 0.

    #### Theano and downhill
    U = theano.shared(np.random.random((M.shape[0],kx)),name='U')
    V = theano.shared(np.random.random((M.shape[1],kx)),name='V')

    #### feature loss
    M_symbolic = theano.tensor.matrix(name="M", dtype=M.dtype)
    reconstruction = theano.tensor.dot(U, V.T)
    difference = M_symbolic - reconstruction
    masked_difference = difference * featuremask
    err = theano.tensor.sqr(masked_difference)
    mse = err.mean()
    xloss = (1./float(len(fea_loc_x))) * mse + miu * ((U * U).mean() + (V * V).mean())
    #### label loss
    label_reconstruction_kernel = -1 * gamma * (2 * M - 1) * (reconstruction - M)
    label_reconstruction_difference = (1./gamma) * theano.tensor.log(1 + theano.tensor.exp(label_reconstruction_kernel)) * labelmask
    label_err = (1./float(len(label_loc_x))) * label_reconstruction_difference.mean()
    global_loss = xloss + lambda0 * label_err

    #### optimisation
    downhill.minimize(
            loss=global_loss,
            train = [M],
            inputs = [M_symbolic],
            patience=0,
            algo='rmsprop',
            batch_size= M.shape[0],
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.01)

    return U.get_value(),V.get_value()

#### generate data
#### yeast: classes 14, data: 1500+917, dimensionality: 103
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

### test
gd_reconstruction_error_list = []
gd_auc_score_list = []
reconstruction_error_list = []
auc_score_list = []

kx = 15
alpha = (1. + 0.5)/2
fea_fraction = 0.8
label_fraction = 0.8
#lambda0 = 10
#lambda1 = 1
#lambda2 = 0.1
#delta = 100

#fea_mask = np.random.random(yeast_data.shape)
#fea_loc = np.where(fea_mask < fea_fraction)
#random_mat = np.random.random(yeast_label.shape)
#label_loc = np.where(random_mat < label_fraction) ## locate the masked entries in the label matrix

#W_pu,H_pu = baselinePU(yeast_label,label_loc,alpha,lambda1,kx)
#print acc_label(yeast_label,W_pu,H_pu,label_loc)
                    
#U,V,W,H = completionPUV(yeast_data,yeast_label,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx) #(X,Y,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx)
#print acc_label(yeast_label,W,H,label_loc)

#print acc_feature(yeast_data,U,V,fea_loc)

#U_lr, V_lr = completionLR(yeast_data,kx,fea_loc,lambda0,lambda0)
#print acc_feature(yeast_data,U_lr,V_lr,fea_loc)
                    
for lambda0 in [10,1,0.1,0.01]:
    for lambda1 in [10,1,0.1,0.01]:
        for lambda2 in [10,1,0.1,0.01]:
            for delta in [100,50,10,1,0.1]:
                for iround in range(10): ### repeat for 10 times
                    fea_mask = np.random.random(yeast_data.shape)
                    fea_loc = np.where(fea_mask < fea_fraction)
                    random_mat = np.random.random(yeast_label.shape)
                    label_loc = np.where(random_mat < label_fraction) ## locate the masked entries in the label matrix
                    W_pu,H_pu = baselinePU(yeast_label,label_loc,alpha,lambda1,kx)
                    auc_score = acc_label(yeast_label,W_pu,H_pu,label_loc)
                    gd_auc_score_list.append(auc_score)
                    U,V,W,H = completionPUV(yeast_data,yeast_label,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx) #(X,Y,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx)
                    auc_score = acc_label(yeast_label,W,H,label_loc)
                    reconstruction_error = acc_feature(yeast_data,U,V,fea_loc)
                    auc_score_list.append(auc_score)
                    reconstruction_error_list.append(reconstruction_error)
                    U_lr, V_lr = completionLR(yeast_data,kx,fea_loc,lambda0,lambda0)
                    reconstruction_error = acc_feature(yeast_data,U_lr,V_lr,fea_loc)
                    gd_reconstruction_error_list.append(reconstruction_error)


parameters_setting = []
for lambda0 in [10,1,0.1,0.01]:
    for lambda1 in [10,1,0.1,0.01]:
        for lambda2 in [10,1,0.1,0.01]:
            for delta in [100,50,10,1,0.1]:
                parameters_setting.append((lambda0,lambda1,lambda2,delta))


import pickle
with open('results_yeast_kx_15.pickle','wb') as f:
    pickle.dump([gd_reconstruction_error_list,gd_auc_score_list,reconstruction_error_list,auc_score_list,parameters_setting],f)