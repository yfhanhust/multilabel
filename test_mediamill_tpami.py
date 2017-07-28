import numpy as np 
import scipy as sp 
import downhill
import theano
from sklearn.metrics import roc_auc_score

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

def CoEmbed(X,Y,fea_loc,label_loc,kx,alpha,beta,gamma):
    nsample = X.shape[0]
    ndim = X.shape[1]
    nlabel = Y.shape[1]
    
    mask = np.ones(X.shape)
    mask[fea_loc] = 0.
    labelmask = np.ones(Y.shape)
    labelmask[label_loc] = 0
    
    #### generate dirty side information matrix 
    X_corrupted = np.multiply(X,mask)
    #### declare variables shared between symbolic and non-symbolic computing thread
    Z = theano.shared(np.random.random((nsample,kx)),name='Z')
    B = theano.shared(np.random.random((nlabel,kx)),name='B')
    W = theano.shared(np.random.random((ndim,kx)),name='W')
    #b = theano.shared(np.random.random((nlabel,1)),name='b')
    #### 
    #### declare symbolic variables 
    feature_mask = theano.tensor.matrix('feature_mask')
    label_mask = theano.tensor.matrix('label_mask')
    
    tX = theano.tensor.matrix('X') ### symbolic variable 
    tY = theano.tensor.matrix('Y') ### symbolic variable 
    XWB = theano.tensor.dot(tX,theano.tensor.dot(W, B.T)) #+ theano.tensor.dot(tX,theano.tensor.dot(np.ones((nsample,1)), b.T))
    Xdifference = theano.tensor.sqr(label_mask * (XWB-tY))
    global_loss = Xdifference.mean() + alpha/2*(W*W).mean() + gamma/2*(B*B).mean()
    #ZB = theano.tensor.dot(Z,B.T)
    #Xdifference = theano.tensor.sqr((XWB - ZB))
    #Ydifference = theano.tensor.sqr(label_mask * (tY - ZB))
    #global_loss = Xdifference.mean() + beta * Ydifference.mean() + alpha/2 * (W * W).mean() + gamma/2 * ((Z * Z).mean() + (B * B).mean())
    #### optimisation
    downhill.minimize(
            loss= global_loss,
            params = [B,W,b],
            train = [X_corrupted,Y,labelmask],
            inputs = [tX,tY,label_mask],
            patience=0,
            algo='rmsprop',
            batch_size=nsample,
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.0005)
    
    return Z.get_value(),B.get_value(),W.get_value()
    
def TPAMI(X,Y,fea_loc_x,fea_loc_y,label_loc_x,label_loc_y,miu,gamma,lambda0,kx):
    ### X: feature matrix 
    ### Y: label matrix 
    ### fea_loc_x, fea_loc_y: masked entries in feature matrix 
    ### label_loc_x, label_loc_y: masked entries in label matrix 
    ### miu: regularisation parameter on matrix rank 
    ### lambda0: regularisation parameter on label reconstruction (delta)
    ### kx: dimensionality of latent variables used for solving nuclear norm based regularisation 
    M = np.concatenate((Y,X),axis=1)
    M = M.T

    label_dim = Y.shape[1]
    fea_dim = X.shape[1]
    #gamma = 15 # 15 
    featuremask = np.ones(M.shape)
    labelmask = np.ones(M.shape)
    for i in range(len(label_loc_x)):
        labelmask[label_loc_y[i],label_loc_x[i]] = 0.

    for i in range(len(fea_loc_x)):
        featuremask[fea_loc_y[i]+label_dim,fea_loc_x[i]] = 0.

    #### Theano and downhill
    U = theano.shared(np.random.random((M.shape[0],kx)),name='U')
    V = theano.shared(np.random.random((M.shape[1],kx)),name='V')
    feature_mask = theano.tensor.matrix(name="feature_mask")
    label_mask = theano.tensor.matrix(name = 'label_mask')
    #### feature loss
    M_symbolic = theano.tensor.matrix(name="M")
    reconstruction = theano.tensor.dot(U, V.T)
    difference = M_symbolic - reconstruction
    masked_difference = difference * feature_mask
    err = theano.tensor.sqr(masked_difference)
    xloss = (1./float(len(fea_loc_x))) * err.mean() + (miu /2.)  * ((U * U).mean() + (V * V).mean())
    #### label loss
    label_reconstruction_kernel = -1 * gamma * (2 * M_symbolic - 1) * (reconstruction - M_symbolic)
    label_reconstruction_difference = (1./gamma) * theano.tensor.log(1 + theano.tensor.exp(label_reconstruction_kernel)) * label_mask
    global_loss = xloss + lambda0 * (1./float(len(label_loc_x))) * label_reconstruction_difference.mean()

    #### optimisation
    downhill.minimize(
            loss=global_loss,
            params = [U,V],
            train = [M,featuremask,labelmask],
            inputs = [M_symbolic,feature_mask,label_mask],
            patience=0,
            algo='rmsprop',
            batch_size= M.shape[0],
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.0001)

    return U.get_value(),V.get_value()

    
def acc_label(Y,Z,B,label_loc):
    Y_reconstructed = np.dot(Z,B.T)
    ground_truth = Y[label_loc].tolist()
    reconstruction = Y_reconstructed[label_loc].tolist()
    auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
    return auc_score
 

def acc_pami(X,Y,U,V,fea_loc,label_loc):
    reconstructed_val = np.dot(U,V.T).T
    nsample = X.shape[0]
    nXdim = X.shape[1]
    nYdim = Y.shape[1]
    X_reconstruction = reconstructed_val[:,nYdim:]
    Y_reconstruction = reconstructed_val[:,:nYdim]
    auc_score = roc_auc_score(np.array(Y[label_loc].tolist()),np.array(Y_reconstruction[label_loc].tolist()))
    return auc_score, np.linalg.norm(X[fea_loc] - X_reconstruction[fea_loc])

  
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



fea_fraction = 0.6
label_fraction = 0.8


gamma = 15.
kx = 10
fixed_parameter = [gamma,kx]
tpami_auc_score = []
tpami_reconstruction_error = []
parameter_list = []
for lambda0 in [0.001,0.01,0.1,1,10,100]:
    for miu in [0.01,0.1,1,10,100]:
        parameter_list.append((lambda0,miu))
        for iround in range(10):
            fea_mask = np.random.random(train_fea.shape)
            fea_loc = np.where(fea_mask < fea_fraction)
            random_mat = np.random.random(train_label.shape)
            label_loc = np.where(random_mat < label_fraction) ## locate the masked entries in the label matrix
            single_round_error = []
            single_round_auc = []
            fea_loc_x = fea_loc[0]
            fea_loc_y = fea_loc[1]
            label_loc_x = label_loc[0]
            label_loc_y = label_loc[1]
            U_pami, V_pami = TPAMI(train_fea,train_label,fea_loc_x,fea_loc_y,label_loc_x,label_loc_y,miu,gamma,lambda0,kx)
            #print 'lambda0: ' + str(miu)
            auc_score, reconstruction_error = acc_pami(train_fea,train_label,U_pami,V_pami,fea_loc,label_loc)
            single_round_error.append(reconstruction_error)
            single_round_auc.append(auc_score)
            
        tpami_auc_score.append(single_round_auc)
        tpami_reconstruction_error.append(single_round_error)

import pickle

result_file_name = 'mediamill_result_tpami.pickle'
with open(result_file_name,'wb') as f:
    pickle.dump([tpami_auc_score,tpami_reconstruction_error,parameter_list,fixed_parameter],f)