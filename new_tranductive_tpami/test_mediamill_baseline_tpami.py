import numpy as np 
import scipy as sp 
from sklearn.linear_model import Ridge 
import downhill
import theano
from sklearn.metrics import roc_auc_score

from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd
def DirtyIMCv1(X,Y,fea_loc,label_loc,kx,ky,alpha,beta):
    ##### min_{M,N} l((XMY^{T}+N),R). s.t. |M|_{*} <= M and |N|_{*} <= N
    ##### min_{U_{M},V_{M},W_{N},H_{N}} l((U_{M}V_{M}^{T}Y^{T}+W_{N}H_{N}^{T}),R), s.t. |U_{M}V_{M}^{T}|_{*} <= M and |W_{N}H_{N}^{T}|_{*} <= 
    nsample = X.shape[0]
    ndim = X.shape[1]
    nlabel = Y.shape[1]

    mask = np.ones(Y.shape)
    mask[label_loc] = 0.
    featuremask = np.zeros(X.shape)
    featuremask[fea_loc] = 1
    unobserved_loc = np.where(featuremask == 0)
    #### generate dirty side information matrix, generated from corrupted labels 
    #X_corrupted = np.multiply(X,featuremask)
    X_corrupted = X.copy()
    X_corrupted[unobserved_loc] = 0.01
    #### eigendecomposition
    #u,s,v = svd(X_corrupted)    
    vl_reduced = TruncatedSVD(algorithm='randomized',n_components=30).fit_transform(X_corrupted)
    #ind = np.argsort(-1.*s)
    #vl_reduced = u[:,:int(float(ndim)/4)]
    #print vl_reduced.shape
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
    #labelmask[np.where(Y > 0)] = 1
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
            min_improvement = 0.001)

    return W.get_value(),H.get_value()


def completionLRV3(X,Y,kx,prob,fea_loc,lambda0,lambda1,lambda2):
    ##### fea_loc: indexes of the observed entries 
    mask = np.zeros(X.shape)
    mask[fea_loc] = 1
    nsample = X.shape[0]
    nfdim = X.shape[1]
    nlabel = Y.shape[1]
    #### Theano and downhill
    pos_loc_num = len(np.where(Y > 0)[0])
    neg_loc_num = len(np.where(Y < 1)[0])    
    factor = float(pos_loc_num) / float(neg_loc_num)
    U = theano.shared(np.random.random((nsample,kx)),name='U')
    V = theano.shared(np.random.random((nfdim,kx)),name='V')
    M = theano.shared(np.random.random((nfdim,nlabel)),name='M')
    #W = theano.shared(np.random.random((nfdim,kx)),name='W')
    #H = theano.shared(np.random.random((nlabel,kx)),name='H')
    #M = theano.shared(np.random.random((nsample,kx)),name='M')
    #N = theano.shared(np.random.random((nlabel,kx)),name='N')
    X_symbolic = theano.tensor.matrix(name="X",dtype=X.dtype)
    Y_symbolic = theano.tensor.matrix(name="Y",dtype=Y.dtype)
    fmask = theano.tensor.matrix(name='mask',dtype=mask.dtype)
    UV = theano.tensor.dot(U,V.T)
    lsq = theano.tensor.sqr(fmask * (X_symbolic - UV)).mean()
    
    #WH = theano.tensor.dot(W, H.T)
    #MN = theano.tensor.dot(M, N.T)
    #UWH = theano.tensor.dot(U,WH) + MN
    XM = theano.tensor.dot(UV,M) 
    logit_pos = (1.-prob) * theano.tensor.gt(Y_symbolic,0) * theano.tensor.log(1. + theano.tensor.exp(-1.* XM))
    logit_neg = prob * theano.tensor.lt(Y_symbolic,1) * theano.tensor.log(1. + theano.tensor.exp(XM))
    
    global_loss = lsq + lambda0 * (logit_pos.mean() + factor * logit_neg.mean()) + lambda1 * ((U*U).mean() + (V*V).mean()) + lambda2 * ((M*M).mean())#lambda2 * ((W*W).mean() + (H*H).mean()) 
    downhill.minimize(
            loss=global_loss,
            params = [U,V,M],
            inputs = [X_symbolic,Y_symbolic,fmask],
            train=[X,Y,mask],
            patience=0,
            algo='rmsprop',
            batch_size=nsample,
            max_gradient_norm=1,
            learning_rate= 0.05,#0.2,
            min_improvement = 0.01)
    
    return U.get_value(),V.get_value(),M.get_value()#W.get_value(),H.get_value()

def CoEmbed(X,Y,fea_loc,label_loc,kx,alpha,beta,gamma):
    nsample = X.shape[0]
    ndim = X.shape[1]
    nlabel = Y.shape[1]
    
    mask = np.zeros(X.shape)
    mask[fea_loc] = 1.
    labelmask = np.ones(Y.shape)
    labelmask[label_loc] = 0
    unobserved_loc = np.where(mask < 1)
    #### generate dirty side information matrix 
    #X_corrupted = np.multiply(X,mask)
    X_corrupted = X.copy()
    X_corrupted[unobserved_loc] = 0.1
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
    
    for i in range(fea_dim):
        featuremask[i+label_dim,:] = 1.

    for i in range(label_dim):
        labelmask[i,:] = 1.
    
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

with open('../inductive/nus_test.csv','rb') as f:
     feature_lines = f.readlines(100000000000000000)
     train_data = np.zeros((len(feature_lines)-1,128))
     train_label = np.zeros((len(feature_lines)-1,81))
     for k in range(1,len(feature_lines)):
         feature_segs = feature_lines[k].split(',')
         for i in range(1,129): ### 128-d feature vector
             train_data[k-1,i-1] = float(feature_segs[i])
         
         for i in range(129,len(feature_segs)):
             train_label[k-1,i-129] = int(feature_segs[i])

indexes = range(0,train_label.shape[0])
np.random.shuffle(np.array(indexes))

nus_data = train_data[indexes[0:5000],:]
nus_label = train_label[indexes[0:5000],:]

label_count = np.sum(nus_label,axis=0)
label_indexes = np.where(label_count > 0)[0]
nus_label = nus_label[:,label_indexes]

fraction = []

for i in range(nus_label.shape[1]):
    single_label = nus_label[:,i]
    fraction.append(np.where(single_label > 0)[0].shape[0] / float(len(single_label)))

sort_fraction = np.argsort(-1*np.array(fraction))
nus_label = nus_label[:,sort_fraction[0:6]]

gamma = 15.
kx = 30
tpami_auc_score = []
fea_fraction = 0.6
label_fraction = 0.8
fea_mask = np.random.random(nus_data.shape)
fea_loc = np.where(fea_mask < (1.-fea_fraction)) ### indexes of the observed entries 

pos_entries = np.where(nus_label == 1)
pos_ind = np.array(range(len(pos_entries[0])))
np.random.shuffle(pos_ind)
labelled_ind = pos_ind[0:int(float(len(pos_ind))*(1-label_fraction))] # 20% of 1s are preserved 
labelled_mask = np.zeros(nus_label.shape)
for i in labelled_ind:
    labelled_mask[pos_entries[0][i],pos_entries[1][i]] = 1

label_loc = np.where(labelled_mask == 0) #### label_loc: observed entries 
parameter_list = []
tpami_auc_score = []
for lambda0 in [0.001,0.01,0.1,1,10,100]:
    for miu in [0.01,0.1,1,5]:
        parameter_list.append((lambda0,miu))
        fea_loc_x = fea_loc[0]
        fea_loc_y = fea_loc[1]
        label_loc_x = label_loc[0]
        label_loc_y = label_loc[1]
        U_pami, V_pami = TPAMI(nus_data,nus_label,fea_loc_x,fea_loc_y,label_loc_x,label_loc_y,miu,gamma,lambda0,kx)
        reconstructed_val = np.dot(U_pami,V_pami.T).T
        nsample = nus_data.shape[0]
        nXdim = nus_data.shape[1]
        nYdim = nus_label.shape[1]
        X_reconstruction = reconstructed_val[:,nYdim:]
        Y_reconstruction = reconstructed_val[:,:nYdim]
        auc_score = roc_auc_score(np.array(nus_label[label_loc].tolist()),np.array(Y_reconstruction[label_loc].tolist()))
        print auc_score     
        tpami_auc_score.append(auc_score)
        #tpami_reconstruction_error.append(single_round_error)

'''
fea_fraction = 0.6
label_fraction = 0.8
fea_mask = np.random.random(nus_data.shape)
fea_loc = np.where(fea_mask < (1.-fea_fraction)) ### indexes of the observed entries 

pos_entries = np.where(nus_label == 1)
pos_ind = np.array(range(len(pos_entries[0])))
np.random.shuffle(pos_ind)
labelled_ind = pos_ind[0:int(float(len(pos_ind))*(1-label_fraction))] # 20% of 1s are preserved 
labelled_mask = np.zeros(nus_label.shape)
for i in labelled_ind:
    labelled_mask[pos_entries[0][i],pos_entries[1][i]] = 1

label_loc = np.where(labelled_mask == 0) #### label_loc: observed entries 
parameter_list = []
auc_aaai_list = []
kx = 30
for beta in [1,10,100,1000]:
    for alpha in [0.01,0.1,1,10]:
        for gamma in [0.1,1,10,100]:
            parameter_list.append([beta,alpha,gamma])
            Z,B,W,b= CoEmbed(nus_data,nus_label,fea_loc,label_loc,kx,alpha,beta,gamma)
            Y_reconstructed = np.dot(Z,B.T)
            ground_truth = nus_label[label_loc].tolist()
            reconstruction = Y_reconstructed[label_loc].tolist()
            auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
            print auc_score
            ### reconstruction error 
            #auc_score = acc_label(train_label,Z,B,label_loc)
            auc_aaai_list.append(auc_score)
'''
#In [7]: parameter_list[idx[0]]
#Out[7]: [100, 0.01, 100]

#In [8]: parameter_list[idx[1]]
#Out[8]: [100, 0.1, 100]

#In [9]: parameter_list[idx[2]]
#Out[9]: [100, 0.01, 10]
#beta = 100
#alpha = 0.01
#gamma = 100
#kx = 30
#fea_fraction = 0.6
#label_fraction = 0.3
#auc_aaai_list = []
#for iround in range(10):
#    fea_mask = np.random.random(nus_data.shape)
#    fea_loc = np.where(fea_mask < (1.-fea_fraction)) ### indexes of the observed entries 
#    pos_entries = np.where(nus_label == 1)
#    pos_ind = np.array(range(len(pos_entries[0])))
#    np.random.shuffle(pos_ind)
#    labelled_ind = pos_ind[0:int(float(len(pos_ind))*(1-label_fraction))] # 20% of 1s are preserved 
#    labelled_mask = np.zeros(nus_label.shape)
#    for i in labelled_ind:
#        labelled_mask[pos_entries[0][i],pos_entries[1][i]] = 1

#    label_loc = np.where(labelled_mask == 0) #### label_loc: observed entries  
#    Z,B,W,b= CoEmbed(nus_data,nus_label,fea_loc,label_loc,kx,alpha,beta,gamma)
#    Y_reconstructed = np.dot(Z,B.T)
#    ground_truth = nus_label[label_loc].tolist()
#    reconstruction = Y_reconstructed[label_loc].tolist()
#    auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
#    print auc_score
    ### reconstruction error 
    #auc_score = acc_label(train_label,Z,B,label_loc)
#    auc_aaai_list.append(auc_score)
#nus_label_masked = nus_label.copy()
#nus_label_masked[label_loc] = 0. #### weak label assignments
#for alpha in [0.01,0.1,0.5,1.,5,10]:
#    for beta in [0.01,0.1,0.5,1.,5,10]:
#        U,V,W,H,train_projection = DirtyIMCv1(nus_data,nus_label,fea_loc,label_loc,kx,ky,alpha,beta)
        #### inductive learning 
#        UV = np.dot(U,V.T)
#        XUV = np.dot(train_projection,UV)
#        WH = np.dot(W,H.T)
#        score_train = XUV + WH
        #score_test = np.ndarray.flatten(np.dot(test_data,UV))
        #ground_truth_test = np.ndarray.flatten(test_label)
#        reconstruction = score_train[label_loc].tolist()
#        ground_truth = train_label[label_loc].tolist()
#        auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
        #auc_score_test_logit = roc_auc_score(ground_truth_test,score_test)
        #test_auc_score.append(auc_score_test_logit)
#        print auc_score
#        dirtyimc.append(auc_score)

#parameter_list = []
#for alpha in [0.01,0.1,0.5,1.,5,10]:
#    for beta in [0.01,0.1,0.5,1.,5,10]:
#        parameter_list.append((alpha,beta))



'''
kx = 10
ky = 10
dirtyimc = []
alpha = 1
beta = 5
for iround in range(10):
    fea_mask = np.random.random(nus_data.shape)
    fea_loc = np.where(fea_mask < (1.-fea_fraction)) ### indexes of the observed entries 

    pos_entries = np.where(nus_label == 1)
    pos_ind = np.array(range(len(pos_entries[0])))
    np.random.shuffle(pos_ind)
    labelled_ind = pos_ind[0:int(float(len(pos_ind))*(1-label_fraction))] # 20% of 1s are preserved 
    labelled_mask = np.zeros(nus_label.shape)
    for i in labelled_ind:
        labelled_mask[pos_entries[0][i],pos_entries[1][i]] = 1

    label_loc = np.where(labelled_mask == 0) #### label_loc: observed entries 
    U,V,W,H,train_projection = DirtyIMCv1(nus_data,nus_label,fea_loc,label_loc,kx,ky,alpha,beta)    
    UV = np.dot(U,V.T)
    XUV = np.dot(train_projection,UV)
    WH = np.dot(W,H.T)
    score_train = XUV + WH
    #score_test = np.ndarray.flatten(np.dot(test_data,UV))
    #ground_truth_test = np.ndarray.flatten(test_label)
    reconstruction = score_train[label_loc].tolist()
    ground_truth = nus_label[label_loc].tolist()
    auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
    #auc_score_test_logit = roc_auc_score(ground_truth_test,score_test)
    #test_auc_score.append(auc_score_test_logit)
    print auc_score
    dirtyimc.append(auc_score)

print np.mean(dirtyimc)
print np.std(dirtyimc)
'''

#parameter_list = []
#for alpha in [0.01,0.1,0.5,1.,5,10]:
#    for beta in [0.01,0.1,0.5,1.,5,10]:
#        parameter_list.append((alpha,beta))


#for alpha in [0.01,0.1,0.5,1.,5,10]:
#    for beta in [0.01,0.1,0.5,1.,5,10]:
#        U,V,W,H,train_projection = DirtyIMCv1(nus_data,nus_label,fea_loc,label_loc,kx,ky,alpha,beta)
        #### inductive learning 
#        UV = np.dot(U,V.T)
#        XUV = np.dot(train_projection,UV)
#        WH = np.dot(W,H.T)
#        score_train = XUV + WH
        #score_test = np.ndarray.flatten(np.dot(test_data,UV))
        #ground_truth_test = np.ndarray.flatten(test_label)
#        reconstruction = score_train[label_loc].tolist()
#        ground_truth = train_label[label_loc].tolist()
#        auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
        #auc_score_test_logit = roc_auc_score(ground_truth_test,score_test)
        #test_auc_score.append(auc_score_test_logit)
#        print auc_score
#        dirtyimc.append(auc_score)

#parameter_list = []
#for alpha in [0.01,0.1,0.5,1.,5,10]:
#    for beta in [0.01,0.1,0.5,1.,5,10]:
#        parameter_list.append((alpha,beta))
'''
prob = 0.2
kx = 10
baseline_acc_list = []
for lambda0 in [0.001,0.005,0.01,0.05,0.1,0.5,1.,1.5]:
    baseline_acc_oneround = []
    for iround in range(15):
        np.random.shuffle(pos_ind)
        labelled_ind = pos_ind[0:int(float(len(pos_ind))*(1-label_fraction))] # 30% of 1s are preserved 
        labelled_mask = np.zeros(nus_label.shape)
        for i in labelled_ind:
            labelled_mask[pos_entries[0][i],pos_entries[1][i]] = 1

        label_loc = np.where(labelled_mask == 0)
        nus_label_masked = nus_label.copy()
        nus_label_masked[label_loc] = 0. ### generate a positive-unlabelled matrix
        W_pu,H_pu = baselinePU(nus_label_masked,label_loc,prob,lambda0,kx)
        Y_reconstructed = np.dot(W_pu,H_pu.T)
        ground_truth = train_label[label_loc].tolist()
        reconstruction = Y_reconstructed[label_loc].tolist()
        auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
        baseline_acc_oneround.append(auc_score)
    
    baseline_acc_list.append(baseline_acc_oneround)
'''


'''
prob = 0.2
nrank = 10
auc_logitlr_list = []
for lambda0 in [0.001,0.005,0.01,0.05,0.1,0.5,1]:
    for lambda1 in [0.001,0.005,0.01,0.05,0.1,0.5,1]:
        for lambda2 in [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,20,30]:
            #for lambda3 in [0.001,0.005,0.01,0.05,0.1,0.5,1]:
            U,V,M = completionLRV3(nus_data,nus_label_masked,nrank,prob,fea_loc,lambda0,lambda1,lambda2)
            XM = np.dot(np.dot(U,V.T),M)
            #UWH = np.dot(U,np.dot(W,H.T))
            ground_truth = nus_label[label_loc].tolist()
            reconstruction = XM[label_loc].tolist()
            print roc_auc_score(np.array(ground_truth),np.array(reconstruction))
            auc_logitlr_list.append(roc_auc_score(np.array(ground_truth),np.array(reconstruction)))


parameters = []
for lambda0 in [0.001,0.005,0.01,0.05,0.1,0.5,1]:
    for lambda1 in [0.001,0.005,0.01,0.05,0.1,0.5,1]:
        for lambda2 in [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,20,30]:
            #for lambda3 in [0.001,0.005,0.01,0.05,0.1,0.5,1]:
                parameters.append((lambda0,lambda1,lambda2))
'''
