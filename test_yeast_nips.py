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
    b = theano.shared(np.random.random((nlabel,1)),name='b')
    #### 
    #### declare symbolic variables 
    feature_mask = theano.tensor.matrix('feature_mask')
    label_mask = theano.tensor.matrix('label_mask')
    unit_vector = np.ones((nsample,1),dtype=float)
    tX = theano.tensor.matrix('X') ### symbolic variable 
    tY = theano.tensor.matrix('Y') ### symbolic variable 
    XWB = theano.tensor.dot(tX,theano.tensor.dot(W, B.T)) + theano.tensor.dot(unit_vector, b.T)
    ZB = theano.tensor.dot(Z,B.T)
    Xdifference = theano.tensor.sqr((XWB-ZB))
    Ydifference = theano.tensor.sqr(label_mask * (tY - ZB))
    global_loss = Xdifference.mean() + beta * Ydifference.mean() + alpha/2*(W*W).mean() + gamma/2*((B*B).mean() + (Z*Z).mean())
    #ZB = theano.tensor.dot(Z,B.T)
    #Xdifference = theano.tensor.sqr((XWB - ZB))
    #global_loss = Xdifference.mean() + beta * Ydifference.mean() + alpha/2 * (W * W).mean() + gamma/2 * ((Z * Z).mean() + (B * B).mean())
    #### optimisation
    downhill.minimize(
            loss= global_loss,
            params = [Z,B,W,b],
            train = [X_corrupted,Y,labelmask],
            inputs = [tX,tY,label_mask],
            patience=0,
            algo='rmsprop',
            batch_size=nsample,
            max_gradient_norm=1,
            learning_rate=0.1,
            min_improvement = 0.0005)
    
    return Z.get_value(),B.get_value(),W.get_value(),b.get_value()
    
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

def DirtyIMCv1(X,Y,fea_loc,label_loc,kx,ky,alpha,beta):
    ##### min_{M,N} l((XMY^{T}+N),R). s.t. |M|_{*} <= M and |N|_{*} <= N
    ##### min_{U_{M},V_{M},W_{N},H_{N}} l((U_{M}V_{M}^{T}Y^{T}+W_{N}H_{N}^{T}),R), s.t. |U_{M}V_{M}^{T}|_{*} <= M and |W_{N}H_{N}^{T}|_{*} <= 
    nsample = X.shape[0]
    ndim = X.shape[1]
    nlabel = Y.shape[1]
    
    mask = np.ones(Y.shape)
    mask[label_loc] = 0.
    featuremask = np.ones(X.shape)
    featuremask[fea_loc] = 0
    
    #### generate dirty side information matrix, generated from corrupted labels 
    X_corrupted = np.multiply(X,featuremask)
    #### eigendecomposition
    u,s,v = svd(X_corrupted)
    #ind = np.argsort(-1.*s)
    vl_reduced = u[:,:int(float(ndim)/4)]
    print vl_reduced.shape
    ndim_reduced = vl_reduced.shape[1]
    #Y_corrupted = Y.copy()
    #Y_corrupted = np.multiply(Y,0.)
    #### declare variables shared between symbolic and non-symbolic computing thread
    U = theano.shared(np.random.random((ndim_reduced,kx)),name='U')
    V = theano.shared(np.random.random((nlabel,kx)),name='V')
    #### UV^{T}\in R^{nsample * nlabel}
    W = theano.shared(np.random.random((nsample,ky)),name='W')
    H = theano.shared(np.random.random((nlabel,ky)),name='H') 
    #### WH^{T}\in R^{nsample * ndim}
    tX = theano.tensor.matrix('X') ### symbolic variable 
    tY = theano.tensor.matrix('Y') ### symbolic variable
    label_mask = theano.tensor.matrix('label_mask')
    UV = theano.tensor.dot(U, V.T)
    WH = theano.tensor.dot(W, H.T)
    XUV = theano.tensor.dot(tX,UV)
    Xdifference = theano.tensor.sqr(label_mask * (tY - XUV - WH)).mean()
    obj = Xdifference + alpha/2*((U*U).mean() + (V*V).mean()) + beta/2*((W*W).mean() + (H*H).mean())
    
    downhill.minimize(
            loss= obj,
            params = [U,V,W,H],
            train = [vl_reduced,Y,mask],
            inputs = [tX,tY,label_mask],
            patience=0,
            algo='rmsprop',
            batch_size=nsample,
            max_gradient_norm=1,
            learning_rate=0.05,
            min_improvement = 0.0005)
            
    return U.get_value(),V.get_value(),W.get_value(),H.get_value(),vl_reduced    
        
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

#fea_fraction = 0.6
#label_fraction = 0.8

auc_nips_list = []
parameter_list = []
for kx in [10,30,100]:
    for ky in [10,30,100]
        for alpha in [0.01,0.1,0.5,1,5]:
            for beta in [0.01,0.1,0.5,1,5]:
                parameter_list.append([alpha,beta,kx,ky])
                single_auc = []
                for iround in range(10):
                    fea_fraction = 0.6
                    label_fraction = 0.8
                    fea_mask = np.random.random(train_fea.shape)
                    fea_loc = np.where(fea_mask < fea_fraction)
                    random_mat = np.random.random(train_label.shape)
                    label_loc = np.where(random_mat < label_fraction) ## locate the masked entries in the label matrix
                    U,V,W,H,F = DirtyIMCv1(train_fea,train_label,fea_loc,label_loc,kx,ky,alpha,beta)
                    ### reconstruction error 
                    #### reconstruction 
                    XUV = np.dot(np.dot(F,U),V.T)
                    WH = np.dot(W,H.T)
                    Y_reconstruction = XUV + WH
                    reconstruction = Y_reconstruction[label_loc].tolist()
                    ground_truth = train_label[label_loc].tolist()
                    auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
                    single_auc.append(auc_score)
                
             auc_nips_list.append(single_auc)

import pickle
result_file_name = 'yeast_result_nips.pickle'
with open(result_file_name,'wb') as f:
    pickle.dump([auc_aaai_list,parameter_list,kx],f)

