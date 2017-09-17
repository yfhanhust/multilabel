import numpy as np 
import scipy as sp 
from sklearn.linear_model import Ridge 
import downhill
import theano
from sklearn.metrics import roc_auc_score

########## inductive training 
# L = |X-Xhat|^2 + lambda0 |Xhat|_star + delta * |Y-Yhat|^2 + delta * lambda1 |Yhat|_star + lambda2 |psi(UV)M - WH|^L1 + lambda3 |M|^2 

def completionPUVInductive(X,Y,ffproj,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,lambda3,delta,kx):

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
    W = theano.shared(np.random.random((Y.shape[0],kx)),name='W')
    H = theano.shared(np.random.random((Y.shape[1],kx)),name='H')
    ##### ffproj: R^{no_of_freq * X.shape[1]}
    no_of_freq = ffproj.shape[0]
    print ffproj.shape
    M = theano.shared(np.random.random((2*no_of_freq,Y.shape[1])),name='M')
    #### declare symbolic variables 
    feature_mask = theano.tensor.matrix('feature_mask')
    label_mask = theano.tensor.matrix('label_mask')
    #### U,V,W and H randomly initialised 
    nsample = X.shape[0]
    tX = theano.tensor.matrix('X') ### symbolic variable
    UVT = theano.tensor.dot(U,V.T) 
    difference = tX - UVT
    masked_difference = difference * feature_mask
    err = theano.tensor.sqr(masked_difference)
    mse = err.mean()
    xloss = mse + lambda0 * ((U * U).mean() + (V * V).mean())
    
    tY = theano.tensor.matrix('Y') ### symbolic variable 
    WHT = theano.tensor.dot(W, H.T)
    #Y_reconstruction = theano.tensor.dot(U,WHT)
    Ydifference = theano.tensor.sqr((tY - WHT)) * (1 - alpha)
    positive_difference = theano.tensor.sqr((tY - WHT) * label_mask) * (2*alpha-1.)
    Ymse = Ydifference.mean() + positive_difference.mean() 
    print 'tffproj'
    ########## |UV^TM-WH^T|_{L1}
    #tffproj = theano.tensor.matrix('ffproj') ### symbolic variable
    Cos_CoreMat = theano.tensor.cos(theano.tensor.dot(tX,ffproj.T))
    Sin_CoreMat = theano.tensor.sin(theano.tensor.dot(tX,ffproj.T))
    ##### combine cosine and sine core matrix 
    Psi = (1./np.sqrt(float(ffproj.shape[0]))) * theano.tensor.concatenate([Cos_CoreMat, Sin_CoreMat], axis=1)
    PsiM = theano.tensor.dot(Psi,M)
    #UM = theano.tensor.dot(U,M) 
    global_loss = xloss + delta * Ymse + lambda1 * ((W * W).mean() + (H * H).mean()) + lambda2 * theano.tensor.sqr((PsiM - WHT)).mean() + lambda3 * (M * M).mean()
    #global_loss = xloss + delta * Ymse + lambda1 * ((W * W).mean() + (H * H).mean())
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
            learning_rate=0.05,
            min_improvement = 0.001)
    
    return U.get_value(),V.get_value(),W.get_value(),H.get_value(),M.get_value()

def completionPUVInductiveLinear(X,Y,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,lambda3,delta,kx):

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
    W = theano.shared(np.random.random((Y.shape[0],kx)),name='W')
    H = theano.shared(np.random.random((Y.shape[1],kx)),name='H')
    ##### ffproj: R^{no_of_freq * X.shape[1]}
    #no_of_freq = ffproj.shape[0]
    #print ffproj.shape
    M = theano.shared(np.random.random((X.shape[1],Y.shape[1])),name='M')
    #### declare symbolic variables 
    feature_mask = theano.tensor.matrix('feature_mask')
    label_mask = theano.tensor.matrix('label_mask')
    #### U,V,W and H randomly initialised 
    nsample = X.shape[0]
    tX = theano.tensor.matrix('X') ### symbolic variable
    UVT = theano.tensor.dot(U,V.T) 
    difference = tX - UVT
    masked_difference = difference * feature_mask
    err = theano.tensor.sqr(masked_difference)
    mse = err.mean()
    xloss = mse + lambda0 * ((U * U).mean() + (V * V).mean())
    
    tY = theano.tensor.matrix('Y') ### symbolic variable 
    WHT = theano.tensor.dot(W, H.T)
    #Y_reconstruction = theano.tensor.dot(U,WHT)
    Ydifference = theano.tensor.sqr((tY - WHT)) * (1 - alpha)
    positive_difference = theano.tensor.sqr((tY - WHT) * label_mask) * (2*alpha-1.)
    Ymse = Ydifference.mean() + positive_difference.mean() 
    #print 'tffproj'
    ########## |UV^TM-WH^T|_{L1}
    #tffproj = theano.tensor.matrix('ffproj') ### symbolic variable
    #Cos_CoreMat = theano.tensor.cos(theano.tensor.dot(tX,ffproj.T))
    #Sin_CoreMat = theano.tensor.sin(theano.tensor.dot(tX,ffproj.T))
    ##### combine cosine and sine core matrix 
    #Psi = (1./np.sqrt(float(ffproj.shape[0]))) * theano.tensor.concatenate([Cos_CoreMat, Sin_CoreMat], axis=1)
    #PsiM = theano.tensor.dot(Psi,M)
    #UM = theano.tensor.dot(U,M) 
    XM = theano.tensor.dot(theano.tensor.dot(U,V.T),M)
    global_loss = xloss + delta * Ymse + lambda1 * ((W * W).mean() + (H * H).mean()) + lambda2 * theano.tensor.sqr((XM - WHT)).mean() + lambda3 * (M * M).mean()
    #global_loss = xloss + delta * Ymse + lambda1 * ((W * W).mean() + (H * H).mean())
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
            learning_rate=0.05,
            min_improvement = 0.001)
    
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
    
def test_func(X_test,ffproj,M):
    Cos_CoreMat = np.cos(np.dot(X_test,ffproj.T))
    Sin_CoreMat = np.sin(np.dot(X_test,ffproj.T))
    ##### combine cosine and sine core matrix 
    Psi = (1./np.sqrt(float(ffproj.shape[0]))) * np.concatenate([Cos_CoreMat, Sin_CoreMat], axis=1)
    PsiM = np.dot(Psi,M)
    return PsiM

train_file = open('../yeast_train.svm','r')
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
test_file = open('../yeast_test.svm','r')
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
#train_fea = yeast_data
#train_label = yeast_label

##### pproj generation 
##### p(u) = N(0,2gammaI)
#x, y = np.random.multivariate_normal(mean, cov, 5000).T
#### randomly divide into train and test set 
mean_vector = np.zeros(train_fea.shape[1])
cov_mat = np.eye(train_fea.shape[1])
ffproj = np.random.multivariate_normal(mean_vector,cov_mat,500)
alpha = 0.8
kx = 10
fea_fraction = 0.6
label_fraction = 0.8

ind_yeast_data = np.array(range(yeast_data.shape[0]))
train_auc_score = []
test_auc_score = []
nsample = yeast_data.shape[0]
num_train = int(nsample * 0.8)
num_test = nsample - num_train
for iround in range(10):
    np.random.shuffle(ind_yeast_data)
    train_data = yeast_data[ind_yeast_data[0:num_train],:]
    test_data = yeast_data[ind_yeast_data[num_train:],:]
    train_label = yeast_label[ind_yeast_data[0:num_train],:]
    test_label = yeast_label[ind_yeast_data[num_train:],:]
    for lambda0 in [0.001,0.01,0.1,1]:
        for lambda1 in [0.001,0.01,0.1,1]:
            for lambda2 in [0.001,0.01,0.1,1]:
                for lambda3 in [0.001,0.01,0.1,1]:
                    for delta in [0.05,0.1,0.5,1]:
                        fea_mask = np.random.random(train_fea.shape)
                        fea_loc = np.where(fea_mask < fea_fraction)
                        pos_entries = np.where(train_label == 1)
                        pos_ind = np.array(range(len(pos_entries[0])))
                        np.random.shuffle(pos_ind)
                        labelled_ind = pos_ind[0:int(float(len(pos_ind))*(1-label_fraction))] # 30% of 1s are preserved 
                        labelled_mask = np.zeros(train_label.shape)
                        for i in labelled_ind:
                            labelled_mask[pos_entries[0][i],pos_entries[1][i]] = 1

                        label_loc = np.where(labelled_mask == 0)
                        train_label_masked = train_label.copy()
                        train_label_masked[label_loc] = 0. ### generate a positive-unlabelled matrix
                        #U,V,W,H,M = completionPUVInductive(train_data,train_label_masked,ffproj,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,lambda3,delta,kx)
                        U,V,W,H,M = completionPUVInductiveLinear(train_data,train_label_masked,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,lambda3,delta,kx)
                        Y_reconstructed = np.dot(W,H.T)
                        ground_truth = train_label[label_loc].tolist()
                        reconstruction = Y_reconstructed[label_loc].tolist()
                        auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction)) #### train_auc_score
                        ##### test phase 
                        Y_test_reconstructed = np.dot(test_data,M)
                        ground_truth_test = test_label.tolist()
                        reconstruction_test = Y_test_reconstructed.tolist()
                        auc_score_test = roc_auc_score(np.array(ground_truth_test),np.array(reconstruction_test)) #### train_auc_score
                        print 'train auc: ' + str(auc_score)
                        print 'test auc: ' + str(auc_score_test)
                        train_auc_score.append(auc_score)
                        test_auc_score.append(auc_score_test)
