####### test positive unlabeled matrix using tensorflow 
import numpy as np 
import scipy as sp 
import tensorflow as tf 
from sklearn.metrics import roc_auc_score
import downhill
import theano

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
            learning_rate=0.05,
            min_improvement = 0.0001)

    return W.get_value(),H.get_value()


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

label_fraction = 0.8
##### 
pos_entries = np.where(train_label == 1)
pos_ind = np.array(range(len(pos_entries[0])))
np.random.shuffle(pos_ind)
labelled_ind = pos_ind[0:int(float(len(pos_ind))*(1-label_fraction))] # 20% of 1s are preserved 
labelled_mask = np.zeros(train_label.shape)
for i in labelled_ind:
    labelled_mask[pos_entries[0][i],pos_entries[1][i]] = 1

label_loc = np.where(labelled_mask == 1) #### label_loc: observed entries 
label_loc_x = label_loc[0]
label_loc_y = label_loc[1]
label_loc_test = np.where(labelled_mask == 0) #### label_loc_test: missing entries 
yeast_label_masked = yeast_label.copy()
yeast_label_masked[label_loc_test] = 0. #### weak label assignments

nrank = 10
lr = tf.constant(.1, name='learning_rate')
alpha = (1. + 0.6)/2
lambda0 = 0.1
alpha_weight = tf.constant((1.-alpha),name='alpha') ### 1 - alpha
beta_weight = tf.constant((2.*alpha-1),name='beta') ### 2* alpha -1 
lda = tf.constant(lambda0, name='lambda0')
global_step = tf.Variable(0, trainable=False)

U = tf.Variable(initial_value=tf.random_uniform([yeast_label.shape[0],nrank],0,1), name='users')
V = tf.Variable(initial_value=tf.random_uniform([nrank,yeast_label.shape[1]],0,1), name='items')
M = tf.matmul(U, V)
result_flatten = tf.reshape(M, [-1])
R = tf.gather(result_flatten, label_loc_x * yeast_label.shape[1] + label_loc_y, name='reconstructed_entries')
observed_labels = yeast_label[label_loc]

diff_op = tf.subtract(R, observed_labels, name='trainig_diff')
base_cost = tf.reduce_mean(tf.multiply(tf.pow(diff_op,2),beta_weight), name="sum_squared_error")

diff_op1 = tf.subtract(M,yeast_label_masked,name='training_diff_full')
base_cost1 = tf.reduce_mean(tf.multiply(tf.pow(diff_op1,2),alpha_weight), name="sum_squared_error_full")
full_base_cost = tf.add(base_cost,base_cost1)

##### regularization
U_frobenius_norm = tf.reduce_mean(tf.pow(U,2,name='use_fnorm'))
V_frobenius_norm = tf.reduce_mean(tf.pow(V,2,name='item_fnorm'))
norm_sums = tf.add(U_frobenius_norm, V_frobenius_norm, name='item_norm')
regularizer = tf.multiply(norm_sums, lda, 'regularizer')

cost = tf.add(full_base_cost, regularizer)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#training_step = tf.train.RMSPropOptimizer(lr,0.9,0.0,1e-10).minimize(cost)
#training_step = tf.train.AdagradOptimizer(lr).minimize(cost)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in xrange(1500):
    sess.run(training_step)
    
#### evaluate reconstruction accuracy 
ground_truth = train_label[label_loc_test].tolist()
U_mat = U.eval(sess)
V_mat = V.eval(sess)
yeast_label_reconstruction = np.dot(U_mat,V_mat)
reconstruction = yeast_label_reconstruction[label_loc_test].tolist()
auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
print auc_score


##### comparison 

W_pu,H_pu = baselinePU(yeast_label_masked,label_loc_test,alpha,lambda0,nrank)
Y_reconstructed = np.dot(W_pu,H_pu.T)
reconstruction = Y_reconstructed[label_loc_test].tolist()
auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction))
print auc_score