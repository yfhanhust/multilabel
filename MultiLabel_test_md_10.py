import numpy as np
import scipy as sp
import downhill
import theano
from sklearn.metrics import roc_auc_score
import tensorflow as tf 


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
    Y_reconstruction = theano.tensor.dot(W, H.T)
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

def completionPUV1(X,Y,fea_loc,label_loc,alpha,lambda0,delta,kx):

    #delta = 0.3
    ### masking out some entries from feature and label matrix
    mask = np.ones(X.shape)
    mask[fea_loc] = 0.
    labelmask = np.ones(Y.shape)
    labelmask[label_loc] = 0

    #### Theano and downhill
    U = theano.shared(np.random.random((X.shape[0],kx)),name='U')
    V = theano.shared(np.random.random((X.shape[1],kx)),name='V')
    #W = theano.shared(np.random.random((Y.shape[0],kx)),name='W')
    H = theano.shared(np.random.random((Y.shape[1],kx)),name='H')

    X_symbolic = theano.tensor.matrix(name="X", dtype=X.dtype)
    reconstruction = theano.tensor.dot(U, V.T)
    difference = X_symbolic - reconstruction
    masked_difference = difference * mask
    err = theano.tensor.sqr(masked_difference)
    mse = err.mean()
    xloss = mse + (lambda0/2.) * ((U * U).mean() + (V * V).mean() + (H * H).mean())

    Y_symbolic = theano.tensor.matrix(name="Y", dtype=Y.dtype)
    Y_reconstruction = theano.tensor.dot(U, H.T)
    Ydifference = theano.tensor.sqr((Y_symbolic - Y_reconstruction)) * (1 - alpha)
    positive_difference = theano.tensor.sqr((Y_symbolic - Y_reconstruction) * labelmask) * (2*alpha-1.)
    Ymse = Ydifference.mean() + positive_difference.mean()
    global_loss = xloss + delta * Ymse

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

    return U.get_value(),V.get_value(),H.get_value()


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
    

def completionPUVTF(X,Y,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx):

    #delta = 0.3
    ### masking out some entries from feature and label matrix

    mask = np.ones(X.shape)
    mask[fea_loc] = 0.
    labelmask = np.ones(Y.shape)
    labelmask[label_loc] = 0
    feature_mask = tf.Variable(mask,dtype=tf.float32)
    label_mask = tf.Variable(labelmask,dtype=tf.float32)
    #### trianing data index 

    #### tensorflow placeholder
    F = tf.Variable(X,dtype=tf.float32)
    L = tf.Variable(Y,dtype=tf.float32)
    
    #### tensorflow variable 
    U = tf.Variable(tf.zeros([X.shape[0],kx]),dtype=tf.float32)
    V = tf.Variable(tf.zeros([X.shape[1],kx]),dtype=tf.float32)
    W = tf.Variable(tf.zeros([Y.shape[0],kx]),dtype=tf.float32)
    H = tf.Variable(tf.zeros([Y.shape[1],kx]),dtype=tf.float32)
    #### Theano and downhill
    #U = theano.shared(np.random.random((X.shape[0],kx)),name='U')
    #V = theano.shared(np.random.random((X.shape[1],kx)),name='V')
    #W = theano.shared(np.random.random((Y.shape[0],kx)),name='W')
    #H = theano.shared(np.random.random((Y.shape[1],kx)),name='H')
    ##### tensorflow objective function 
    UV = tf.matmul(U,V,transpose_b=True)
    WH = tf.matmul(W,H,transpose_b=True)
    
    rF = tf.reduce_sum(tf.pow(tf.multiply(tf.subtract(UV,F),feature_mask),2))
    f_norm_loss = rF + tf.add(tf.multiply(lambda0, tf.nn.l2_loss(U)), tf.multiply(lambda0, tf.nn.l2_loss(V)))
    #X_symbolic = theano.tensor.matrix(name="X", dtype=X.dtype)
    #reconstruction = theano.tensor.dot(U, V.T)
    #difference = X_symbolic - reconstruction
    #masked_difference = difference * mask
    #err = theano.tensor.sqr(masked_difference)
    #mse = err.mean()
    #xloss = mse + lambda0 * ((U * U).mean() + (V * V).mean())
    LWH = tf.subtract(L,WH)
    L_difference = tf.reduce_sum(tf.multiply((1.-alpha),tf.pow(LWH,2))) + tf.reduce_sum(tf.multiply((2*alpha-1),tf.pow(tf.multiply(LWH,label_mask),2)))
    #positive_difference = theano.tensor.sqr((Y_symbolic - Y_reconstruction) * labelmask) * (2*alpha-1.)
    L_mse =  L_difference + tf.multiply(delta,L_difference) + tf.add(tf.multiply(lambda1, tf.nn.l2_loss(W)), tf.multiply(lambda1, tf.nn.l2_loss(H)))
    global_loss = f_norm_loss + L_mse + tf.multiply(lambda2, tf.reduce_sum(tf.pow(tf.subtract(U,W),2)))
    train_step = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(global_loss)
    init_op = tf.initialize_all_variables()
    steps = 2000
    with tf.Session() as sess:
         sess.run(init_op)
         for i in range(steps):
             sess.run(train_step)
             if i%100 == 0:
                 print("\nCost: %f" % sess.run(global_loss))
         learntU = sess.run(U)
         learntV = sess.run(V)
         learntW = sess.run(W)
         learntH = sess.run(H)
    #### optimisation
    #downhill.minimize(
    #        loss=global_loss,
    #        train = [X,Y],
    #        inputs = [X_symbolic,Y_symbolic],
    #        patience=0,
    #        algo='rmsprop',
    #        batch_size=Y.shape[0],
    #        max_gradient_norm=1,
    #        learning_rate=0.1,
    #        min_improvement = 0.0001)

    return learntU,learntV,learntW,learntH

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

###### for debug
kx = 10
alpha = (1. + 0.5)/2
fea_fraction = 0.8
label_fraction = 0.8
lambda0 = 0.1 ### regularisation on U,V and H
delta = 0.1 ### penalty trade-off between reconstruction of feature matrix and reconstruction of label matrix 

fea_mask = np.random.random(train_fea.shape)
fea_loc = np.where(fea_mask < fea_fraction)
random_mat = np.random.random(train_label.shape)
label_loc = np.where(random_mat < label_fraction) ## locate the masked entries in the label matrix

W_pu,H_pu = baselinePU(train_label,label_loc,alpha,lambda0,kx)
pu_label = acc_label(train_label,W_pu,H_pu,label_loc)
#completionPUV(X,Y,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx)             
lambda2 = 10.      
U,V,W,H = completionPUVTF(train_fea,train_label,fea_loc,label_loc,alpha,lambda0,lambda0,lambda2,delta,kx) #(X,Y,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx)
algo_label = acc_label(train_label,W,H,label_loc)
algo_error = acc_feature(train_fea,U,V,fea_loc)

#U,V,H = completionPUV1(train_fea,train_fea,fea_loc,label_loc,alpha,lambda0,delta,kx)
#algo_label = acc_label(train_label,U,H,label_loc)
#algo_error = acc_feature(train_fea,U,V,fea_loc)

U_lr, V_lr = completionLR(train_fea,kx,fea_loc,lambda0,lambda0)
lr_error = acc_feature(train_fea,U_lr,V_lr,fea_loc)

print 'baseline classification AUC: ' + str(pu_label)
print 'the proposed method AUC: ' + str(algo_label)
print 'baseline reconstruction error: ' + str(lr_error)
print 'the proposed method reconstruction error: ' + str(algo_error)
###### for debug

for lambda0 in [10,1,0.1,0.01]:
    for lambda1 in [10,1,0.1,0.01]:
        for lambda2 in [10,1,0.1,0.01]:
            for delta in [10,1,0.1]:
                for iround in range(10): ### repeat for 10 times
                    fea_mask = np.random.random(train_fea.shape)
                    fea_loc = np.where(fea_mask < fea_fraction)
                    random_mat = np.random.random(train_label.shape)
                    label_loc = np.where(random_mat < label_fraction) ## locate the masked entries in the label matrix
                    W_pu,H_pu = baselinePU(train_label,label_loc,alpha,lambda1,kx)
                    auc_score = acc_label(train_label,W_pu,H_pu,label_loc)
                    gd_auc_score_list.append(auc_score)
                    U,V,W,H = completionPUVVersion1(train_fea,train_label,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx) #(X,Y,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,delta,kx)
                    auc_score = acc_label(train_label,W,H,label_loc)
                    reconstruction_error = acc_feature(train_fea,U,V,fea_loc)
                    auc_score_list.append(auc_score)
                    reconstruction_error_list.append(reconstruction_error)
                    U_lr, V_lr = completionLR(train_fea,kx,fea_loc,lambda0,lambda0)
                    reconstruction_error = acc_feature(train_fea,U_lr,V_lr,fea_loc)
                    gd_reconstruction_error_list.append(reconstruction_error)


parameters_setting = []
for lambda0 in [10,1,0.1,0.01]:
    for lambda1 in [10,1,0.1,0.01]:
        for lambda2 in [10,1,0.1,0.01]:
            for delta in [10,1,0.1]:
                parameters_setting.append((lambda0,lambda1,lambda2,delta))


import pickle
with open('results_15.pickle','wb') as f:
    pickle.dump([gd_reconstruction_error_list,gd_auc_score_list,reconstruction_error_list,auc_score_list,parameters_setting],f)


#### non-negative matirx factorization 
import tensorflow as tf 
import numpy as np 
import pandas as pd

np.random.seed(0)
A_orig = np.array([[3, 4, 5, 2],
                   [4, 4, 3, 3],
                   [5, 5, 4, 4]], dtype=np.float32).T
A_orig_df = pd.DataFrame(A_orig)

A_df_masked = A_orig_df.copy()
A_df_masked.iloc[0,0]=np.NAN
np_mask = A_df_masked.notnull()

tf_mask = tf.Variable(np_mask.values)
A = tf.constant(A_df_masked.values)
shape = A_df_masked.values.shape

rank = 3 
temp_H = np.random.randn(rank, shape[1]).astype(np.float32)
temp_H = np.divide(temp_H, temp_H.max())

temp_W = np.random.randn(shape[0], rank).astype(np.float32)
temp_W = np.divide(temp_W, temp_W.max())

H = tf.Variable(temp_H)
W = tf.Variable(temp_W)
WH = tf.matmul(W,H)

### cost function
cost = tf.reduce_sum(tf.pow(tf.boolean_mask(A,tf_mask) - tf.boolean_mask(WH,tf_mask),2))
### learning rate 
lr = 0.001
steps = 1000
train_step = tf.train.AdagradOptimizer(learning_rate=lr).minimize(cost)
init = tf.global_variables_initializer()
### Ensuring non-negativity 
clip_W = W.assign(tf.maximum(tf.zeros_like(W),W))
clip_H = H.assign(tf.maximum(tf.zeros_like(H),H))
clip = tf.group(clip_W,clip_H)

steps = 5000
with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        sess.run(train_step)
        sess.run(clip) ## enforcing non-negativity 
        if i%100 == 0:
            print("\ncost: %f\n" % sess.run(cost))
    learnt_W = sess.run(W)
    learnt_H = sess.run(H)
    
