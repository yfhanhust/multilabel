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
            min_improvement = 0.0005)

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
    labelmask[label_loc] = 0
    Y_masked = Y.copy()
    Y_masked[label_loc] = 0

    reconstruction = theano.tensor.dot(W, H.T)
    X_symbolic = theano.tensor.matrix(name="Y_masked", dtype=Y_masked.dtype)
    difference = theano.tensor.sqr((X_symbolic - reconstruction)) * (1 - alpha)
    positive_difference = theano.tensor.sqr((X_symbolic - reconstruction) * labelmask) * (2*alpha-1.)

    mse = difference.mean() + positive_difference.mean()
    loss = mse + (vlambda/2.) * (W * W).mean() + (vlambda/2.) * (H * H).mean()

    downhill.minimize(
            loss=loss,
            train=[Y_masked],
            patience=0,
            algo='rmsprop',
            batch_size=Y_masked.shape[0],
            max_gradient_norm=1,
            learning_rate=0.06,
            min_improvement = 0.001)

    return W.get_value(),H.get_value()


def solve_UV(X,W,fea_loc,lambda0,lambda_diff,kx):

    #delta = 0.3
    ### masking out some entries from feature and label matrix
    mask = np.ones(X.shape)
    mask[fea_loc] = 0.
    
    #### Theano and downhill
    #### declare variables shared between symbolic and non-symbolic computing thread
    U = theano.shared(np.random.random((X.shape[0],kx)),name='U')
    V = theano.shared(np.random.random((X.shape[1],kx)),name='V')
    #### declare symbolic variables 
    feature_mask = theano.tensor.matrix('feature_mask')
    #feature_mask = theano.shared(mask,name='mask')
    #label_mask = theano.shared(labelmask,name='labelmask')
    #### U,V,W and H randomly initialised     
    nsample = X.shape[0]
    #X_symbolic = theano.tensor.matrix(name="X", dtype=X.dtype)
    #tX = theano.shared(X.astype(theano.config.floatX),name="X")
    #tX = theano.shared(X,name="X")
    tX = theano.tensor.matrix('X') ### symbolic variable
    tW = theano.tensor.matrix('W') ### symbolic variable  
    difference = tX - theano.tensor.dot(U, V.T)
    masked_difference = difference * feature_mask
    err = theano.tensor.sqr(masked_difference)
    mse = err.mean()
    xloss = mse + (lambda0/2.) * ((U * U).mean() + (V * V).mean()) + lambda_diff * theano.tensor.sqr((U - tW)).mean() 

    #### optimisation
    downhill.minimize(
            loss= xloss,
            params = [U,V],
            train = [X,W,mask],
            inputs = [tX,tW,feature_mask],
            patience=0,
            algo='rmsprop',
            batch_size=nsample,
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.0005)
    
    return U.get_value(),V.get_value()


def solve_WH(Y,U,label_loc,alpha,delta,lambda1,lambda_diff,kx):

    #delta = 0.3
    ### masking out some entries from feature and label matrix
    labelmask = np.ones(Y.shape)
    labelmask[label_loc] = 0
    nsample = Y.shape[0]
    ndim  = Y.shape[1]
    #### Theano and downhill
    #### declare variables shared between symbolic and non-symbolic computing thread
    W = theano.shared(np.random.random((nsample,kx)),name='W')
    H = theano.shared(np.random.random((ndim,kx)),name='H')
    #### declare symbolic variables 
    label_mask = theano.tensor.matrix('label_mask')
    #feature_mask = theano.shared(mask,name='mask')
    #label_mask = theano.shared(labelmask,name='labelmask')
    #### U,V,W and H randomly initialised     
    #X_symbolic = theano.tensor.matrix(name="X", dtype=X.dtype)
    #tX = theano.shared(X.astype(theano.config.floatX),name="X")
    #tX = theano.shared(X,name="X")
    tY = theano.tensor.matrix('Y') ### symbolic variable
    tU = theano.tensor.matrix('U') ### symbolic variable   
    Y_reconstruction = theano.tensor.dot(W, H.T)
    Ydifference = theano.tensor.sqr((tY - Y_reconstruction)) * (1 - alpha)
    positive_difference = theano.tensor.sqr((tY - Y_reconstruction) * label_mask) * (2*alpha-1.)
    Ymse = Ydifference.mean() + positive_difference.mean()
    global_loss = delta * Ymse + (lambda1/2.) * ((W * W).mean() + (H * H).mean()) + lambda_diff * theano.tensor.sqr((tU-W)).mean()

    #### optimisation
    downhill.minimize(
            loss= global_loss,
            params = [W,H],
            train = [Y,U,labelmask],
            inputs = [tY,tU,label_mask],
            patience=0,
            algo='rmsprop',
            batch_size=nsample,
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.0005)
    
    return W.get_value(),H.get_value()


def completionPUV(X,Y,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx):

    #delta = 0.3
    ### masking out some entries from feature and label matrix

    mask = np.ones(X.shape)
    mask[fea_loc] = 0.
    labelmask = np.ones(Y.shape)
    labelmask[label_loc] = 0

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
            min_improvement = 0.0005)
    
    return U.get_value(),V.get_value(),W.get_value(),H.get_value()


def completionPUV1(X,Y,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx,kd):

    #delta = 0.3
    ### masking out some entries from feature and label matrix
    #kd is the dimension of side information

    mask = np.ones(X.shape)
    mask[fea_loc] = 0.
    labelmask = np.ones(Y.shape)
    labelmask[label_loc] = 0

    #### Theano and downhill
    #### declare variables shared between symbolic and non-symbolic computing thread
    U = theano.shared(np.random.random((kd,kx)),name='U')
    V = theano.shared(np.random.random((X.shape[1],kx)),name='V')
    W = theano.shared(np.random.random((kd,kx)),name='W')
    H = theano.shared(np.random.random((Y.shape[1],kx)),name='H')
    A = theano.shared(np.random.random((X.shape[0],kd)),name='A')
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
    difference = tX - theano.tensor.dot(A,theano.tensor.dot(U, V.T))
    masked_difference = difference * feature_mask
    err = theano.tensor.sqr(masked_difference)
    mse = err.mean()
    #xloss = mse + lambda0 * ((U * U).mean() + (V * V).mean())

    #tY = theano.shared(Y.astype(theano.config.floatX),name="Y")
    #tY = theano.shared(Y,name="Y")
    tY = theano.tensor.matrix('Y') ### symbolic variable 
    Y_reconstruction = theano.tensor.dot(A,theano.tensor.dot(W, H.T))
    Ydifference = theano.tensor.sqr((tY - Y_reconstruction)) * (1 - alpha)
    positive_difference = theano.tensor.sqr((tY - Y_reconstruction) * label_mask) * (2*alpha-1.)
    Ymse = Ydifference.mean() + positive_difference.mean()    
    global_loss = mse + delta * Ymse + lambda0 * ((U * U).mean() + (V * V).mean()) + lambda1 * ((W * W).mean() + (H * H).mean()) + lambda2 * ((A * A).mean())
    
    #### optimisation
    downhill.minimize(
            loss= global_loss,
            params = [U,V,W,H,A],
            train = [X,Y,mask,labelmask],
            inputs = [tX,tY,feature_mask,label_mask],
            patience=0,
            algo='rmsprop',
            batch_size=nsample,
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.0005)
    
    return U.get_value(),V.get_value(),W.get_value(),H.get_value(),A.get_value()
    
def acc_label(Y,W,H,label_loc):
    Y_reconstructed = np.dot(W,H.T)
    ground_truth = Y[label_loc].tolist()
    reconstruction = Y_reconstructed[label_loc].tolist()
    auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
    return auc_score

def acc_feature(X,U,V,fea_loc):
    X_reconstruction = U.dot(V.T)
    return np.linalg.norm(X[fea_loc] - X_reconstruction[fea_loc])    
    
def acc_label1(Y,A,W,H,label_loc):
    Y_reconstructed = np.dot(A,np.dot(W,H.T))
    ground_truth = Y[label_loc].tolist()
    reconstruction = Y_reconstructed[label_loc].tolist()
    auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
    return auc_score

def acc_feature1(X,A,U,V,fea_loc):
    X_reconstruction = np.dot(A,U.dot(V.T))
    return np.linalg.norm(X[fea_loc] - X_reconstruction[fea_loc])    
    
### Fast randomized SingularValue Thresholding for Low-rank Optimization
#### generate data
#### yeast: classes 14, data: 1500+917, dimensionality: 103
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



### test
gd_reconstruction_error_list = []
gd_auc_score_list = []
reconstruction_error_list = []
auc_score_list = []
parameter_list = []
###### for debug
###### fixed parameter 
kx = 10
kd = 300 
alpha = (1. + 0.5)/2 ### insensitive parameter in PU matrix completion
delta = 5.
lambda2 = 1000. ### regularisation on A 
fixed_parameter = [kx,kd,alpha,delta,lambda2]
for lambda0 in [0.01,0.05,0.1]: ### 3 
    for lambda1 in [0.01,0.05,0.1]: ### 3
        for delta in [1,5,10]: ### 3
            gd_reconstruction_one_round = []
            gd_acc_one_round = []
            reconstruction_one_round = []
            acc_one_round = []
            parameter_list.append([lambda0,lambda1,delta])
            for iround in range(10): ### 10 
                fea_fraction = 0.6
                label_fraction = 0.8
                fea_mask = np.random.random(train_fea.shape)
                fea_loc = np.where(fea_mask < fea_fraction)
                random_mat = np.random.random(train_label.shape)
                label_loc = np.where(random_mat < label_fraction) ## locate the masked entries in the label matrix

                #### baseline
                W_pu,H_pu = baselinePU(train_label,label_loc,alpha,lambda1,kx)
                U_lr, V_lr = completionLR(train_fea,kx,fea_loc,lambda0)
                #### comparison
                U_opt1,V_opt1,W_opt1,H_opt1,A_opt1 = completionPUV1(train_fea,train_label,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx,kd)
                #### evaluation metrics 
                lr_error = acc_feature(train_fea,U_lr,V_lr,fea_loc)
                pu_label = acc_label(train_label,W_pu,H_pu,label_loc)
                opt_error = acc_feature1(train_fea,A_opt1,U_opt1,V_opt1,fea_loc)
                opt_label = acc_label1(train_label,A_opt1,W_opt1,H_opt1,label_loc)
                #### store into list
                gd_reconstruction_one_round.append(lr_error)
                gd_acc_one_round.append(pu_label)
                reconstruction_one_round.append(opt_error)
                acc_one_round.append(opt_label)
                print 'reconstruction:  ' + str(lr_error) + ' ' + str(opt_error)
                print 'auc:  ' + str(pu_label) + ' ' + str(opt_label)
             
            gd_reconstruction_error_list.append(gd_reconstruction_one_round)
            gd_auc_score_list.append(gd_acc_one_round)
            reconstruction_error_list.append(reconstruction_one_round)
            auc_score_list.append(acc_one_round)

#lambda0 = 0.05
#lambda1 = 0.01
#lambda2 = 1000.
#delta = 5.
#U_opt1,V_opt1,W_opt1,H_opt1,A_opt1 = completionPUV1(train_fea,train_label,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx,kd)

#algo_label = acc_label(train_label,W,H,label_loc)
#algo_error = acc_feature(train_fea,U,V,fea_loc)
#lr_error = acc_feature(train_fea,U_lr,V_lr,fea_loc)
#pu_label = acc_label(train_label,W_pu,H_pu,label_loc)
#opt_error = acc_feature1(train_fea,A_opt1,U_opt1,V_opt1,fea_loc)
#opt_label = acc_label1(train_label,A_opt1,W_opt1,H_opt1,label_loc)
import pickle
import numpy as np 

result_file_name = 'mediamill_result_coembedding.pickle'
#with open(result_file_name,'wb') as f:
#    pickle.dump([gd_reconstruction_error_list,gd_auc_score_list,reconstruction_error_list,auc_score_list,parameter_list,fixed_parameter],f)

with open(result_file_name,'rb') as f:
    [gd_reconstruction_error_list,gd_auc_score_list,reconstruction_error_list,auc_score_list,parameter_list,fixed_parameter] = pickle.load(f)
    
#### mean and variance 
gd_reconstruction_error_mean = []
gd_reconstruction_error_var = []
gd_auc_score_mean = []
gd_auc_score_var = []
reconstruction_error_mean = []
reconstruction_error_var = []
auc_score_mean = []
auc_score_var = []


for i in range(len(parameter_list)):
    gd_reconstruction_error_mean.append(np.mean(np.array(gd_reconstruction_error_list[i])))
    gd_reconstruction_error_var.append(np.std(np.array(gd_reconstruction_error_list[i])))
    gd_auc_score_mean.append(np.mean(np.array(gd_auc_score_list[i])))
    gd_auc_score_var.append(np.std(np.array(gd_auc_score_list[i])))
    reconstruction_error_mean.append(np.mean(np.array(reconstruction_error_list[i])))
    reconstruction_error_var.append(np.std(np.array(reconstruction_error_list[i])))
    auc_score_mean.append(np.mean(np.array(auc_score_list[i])))
    auc_score_var.append(np.std(np.array(auc_score_list[i])))

import matplotlib.pyplot as plt 
plt.plot(range(len(gd_reconstruction_error_mean)),gd_reconstruction_error_mean,'r',label='Baseline',linewidth=5)
plt.plot(range(len(reconstruction_error_mean)),reconstruction_error_mean,'b',label='The proposed method',linewidth=5)
plt.xlabel('Different Parameter Setting')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()

#### mean and variance 
gd_reconstruction_error_mean = []
gd_reconstruction_error_var = []
gd_auc_score_mean = []
gd_auc_score_var = []
reconstruction_error_mean = []
reconstruction_error_var = []
auc_score_mean = []
auc_score_var = []


for i in range(len(parameter_list)):
    gd_reconstruction_error_mean.append(np.mean(np.array(gd_reconstruction_error_list[i])))
    gd_reconstruction_error_var.append(np.std(np.array(gd_reconstruction_error_list[i])))
    gd_auc_score_mean.append(np.mean(np.array(gd_auc_score_list[i])))
    gd_auc_score_var.append(np.std(np.array(gd_auc_score_list[i])))
    reconstruction_error_mean.append(np.mean(np.array(reconstruction_error_list[i])))
    reconstruction_error_var.append(np.std(np.array(reconstruction_error_list[i])))
    auc_score_mean.append(np.mean(np.array(auc_score_list[i])))
    auc_score_var.append(np.std(np.array(auc_score_list[i])))

import matplotlib.pyplot as plt 
plt.plot(range(len(gd_auc_score_mean)),gd_auc_score_mean,'r',label='Baseline',linewidth=5)
plt.plot(range(len(auc_score_mean)),auc_score_mean,'b',label='The proposed method',linewidth=5)
plt.xlabel('Different Parameter Setting')
plt.ylabel('Area-Under-Curve Score')
plt.legend()
plt.show()