####### test positive unlabeled matrix using tensorflow 
import numpy as np 
import scipy as sp 
import tensorflow as tf 
from sklearn.metrics import roc_auc_score
        
def ComputePUV_Inductive_TF(X,Y,U_init,V_init,W_init,H_init,M_init,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,lambda3,delta,kx,lr,max_iter):  
    alpha_weight = tf.constant((1.-alpha),name='alpha',dtype=tf.float64) ### 1 - alpha
    beta_weight = tf.constant((2.*alpha-1),name='beta',dtype=tf.float64) ### 2* alpha -1 
    fea_loc_x = fea_loc[0]
    fea_loc_y = fea_loc[1]
    label_loc_x = label_loc[0]
    label_loc_y = label_loc[1]
    mask = np.zeros(X.shape)
    mask[fea_loc] = 1. ## observed entries 
    labelmask = np.zeros(Y.shape)
    labelmask[label_loc] = 1.
    
    U = tf.Variable(initial_value = U_init,name='U',dtype=tf.float64)
    V = tf.Variable(initial_value = V_init.T,name='V',dtype=tf.float64)
    S = tf.matmul(U, V)
    
    debug_rs = tf.subtract(S,X,name='debug_rs')
    sqr_fea = tf.reduce_mean(tf.pow(tf.multiply(debug_rs,mask),2),name='sqr_debug')
    
    #result_flatten_fea = tf.reshape(S, [-1])
    #RS = tf.gather(result_flatten_fea, fea_loc_x * X.shape[1] + fea_loc_y, name='reconstructed_fea')
    #observed_fea = X[fea_loc]
    #diff_op_fea = tf.subtract(RS, observed_fea, name='diff_fea')
    #sqr_fea = tf.reduce_mean(tf.pow(diff_op_fea,2), name="sqr_fea")

    U_frobenius_norm = tf.reduce_mean(tf.pow(U,2,name='U_frobenius'))
    V_frobenius_norm = tf.reduce_mean(tf.pow(V,2,name='V_frobenius'))
    norm_sums_fea = tf.add(U_frobenius_norm, V_frobenius_norm, name='norm_reg_fea')
    regularizer_fea = tf.multiply(norm_sums_fea, lambda0, 'regularizer_fea')

    W = tf.Variable(initial_value = W_init,name='W',dtype=tf.float64)
    H = tf.Variable(initial_value = H_init.T,name='H',dtype=tf.float64)
    M = tf.matmul(W, H)
    #result_flatten = tf.reshape(M, [-1])
    #R = tf.gather(result_flatten, label_loc_x * Y.shape[1] + label_loc_y, name='reconstructed_entries')
    #observed_labels = Y[label_loc]
    
    #diff_op_label1 = tf.subtract(R, observed_labels, name='diff_label1')
    #sqr_label_p1 = tf.reduce_mean(tf.multiply(tf.pow(diff_op_label1,2),beta_weight), name="sqr_label1")
    R = tf.subtract(M,Y,name='res_label')
    sqr_label_p1 = tf.reduce_mean(tf.multiply(tf.pow(tf.multiply(R,labelmask),2),beta_weight), name="sqr_label1")
    #diff_op_label2 = tf.subtract(M,Y,name='diff_label2')
    sqr_label_p2 = tf.reduce_mean(tf.multiply(tf.pow(R,2),alpha_weight), name="sqr_label2")
    sqr_label = tf.add(sqr_label_p1,sqr_label_p2)
    fused_cost = tf.add(sqr_fea,tf.multiply(sqr_label,delta))

    ##### regularization
    W_frobenius_norm = tf.reduce_mean(tf.pow(W,2,name='W_frobenius'))
    H_frobenius_norm = tf.reduce_mean(tf.pow(H,2,name='H_frobenius'))
    norm_sums_label = tf.add(W_frobenius_norm, H_frobenius_norm, name='norm_reg_label')
    regularizer_label = tf.multiply(norm_sums_label, lambda1, 'regularizer_label')

    cost = tf.add(tf.add(fused_cost, regularizer_fea),regularizer_label)
    #### linear prediction based regularization |UB-W| (L2 norm ? L1 norm)
    #B = tf.Variable(initial_value=tf.random_uniform([nrank,nrank],0,1), name='B')
    B = tf.Variable(initial_value=B_init,name='B',dtype=tf.float64)
    SB = tf.matmul(S,B)
    diff_ubw = tf.subtract(SB,M,name='diff_ubw')
    #### L2-norm 
    sqr_ubw = tf.reduce_mean(tf.multiply(tf.pow(diff_ubw,2),lambda2), name="sqr_label2")
    #### L1-norm
    #sqr_ubw = tf.reduce_mean(tf.multiply(tf.abs(diff_ubw),lambda2),name="sqr_label2")
    regularizer_ubw = tf.multiply(tf.reduce_mean(tf.pow(B,2,name='B_frobenius')),lambda3)
    cost_reg = tf.add(tf.add(cost,sqr_ubw),regularizer_ubw)

    global_step = tf.Variable(0, trainable=False)
    #learning_rate = tf.train.exponential_decay(lr, global_step, 20000, 0.996, staircase=True)
    #training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_reg)
    training_step = tf.train.RMSPropOptimizer(lr,0.9,0.0,1e-10).minimize(cost_reg)
    #training_step = tf.train.AdagradOptimizer(lr).minimize(cost_reg)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    print 'initial cost function value: ' + str(sess.run(cost_reg))
    
    for i in xrange(max_iter):
        sess.run(training_step)
    
    #print 'final cost function value: ' + str(sess.run(cost_reg))
    #sess.run(cost_reg)
    #sess.run(cost)
    return  U.eval(sess), V.eval(sess), W.eval(sess), H.eval(sess), B.eval(sess)
    
def CostFuncEval(X,Y,U_init,V_init,W_init,H_init,B_init,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,lambda3,delta,kx):
    alpha_weight = tf.constant((1.-alpha),name='alpha',dtype=tf.float64) ### 1 - alpha
    beta_weight = tf.constant((2.*alpha-1),name='beta',dtype=tf.float64) ### 2* alpha -1 
    fea_loc_x = fea_loc[0]
    fea_loc_y = fea_loc[1]
    label_loc_x = label_loc[0]
    label_loc_y = label_loc[1]
    mask = np.zeros(X.shape)
    mask[fea_loc] = 1. ## observed entries 
    labelmask = np.zeros(Y.shape)
    labelmask[label_loc] = 1.
    
    U = tf.constant(U_init,name='U',dtype=tf.float64)
    V = tf.constant(V_init,name='V',dtype=tf.float64)
    S = tf.matmul(U, V)
    
    debug_rs = tf.subtract(S,X,name='debug_rs')
    sqr_fea = tf.reduce_mean(tf.pow(tf.multiply(debug_rs,mask),2),name='sqr_debug')
    
    #result_flatten_fea = tf.reshape(S, [-1])
    #RS = tf.gather(result_flatten_fea, fea_loc_x * X.shape[1] + fea_loc_y, name='reconstructed_fea')
    #observed_fea = X[fea_loc]
    #diff_op_fea = tf.subtract(RS, observed_fea, name='diff_fea')
    #sqr_fea = tf.reduce_mean(tf.pow(diff_op_fea,2), name="sqr_fea")

    U_frobenius_norm = tf.reduce_mean(tf.pow(U,2,name='U_frobenius'))
    V_frobenius_norm = tf.reduce_mean(tf.pow(V,2,name='V_frobenius'))
    norm_sums_fea = tf.add(U_frobenius_norm, V_frobenius_norm, name='norm_reg_fea')
    regularizer_fea = tf.multiply(norm_sums_fea, lambda0, 'regularizer_fea')

    W = tf.constant(W_init,name='W',dtype=tf.float64)
    H = tf.constant(H_init,name='H',dtype=tf.float64)
    M = tf.matmul(W, H)
    #result_flatten = tf.reshape(M, [-1])
    #R = tf.gather(result_flatten, label_loc_x * Y.shape[1] + label_loc_y, name='reconstructed_entries')
    #observed_labels = Y[label_loc]
    
    #diff_op_label1 = tf.subtract(R, observed_labels, name='diff_label1')
    #sqr_label_p1 = tf.reduce_mean(tf.multiply(tf.pow(diff_op_label1,2),beta_weight), name="sqr_label1")
    R = tf.subtract(M,Y,name='res_label')
    sqr_label_p1 = tf.reduce_mean(tf.multiply(tf.pow(tf.multiply(R,labelmask),2),beta_weight), name="sqr_label1")
    #diff_op_label2 = tf.subtract(M,Y,name='diff_label2')
    sqr_label_p2 = tf.reduce_mean(tf.multiply(tf.pow(R,2),alpha_weight), name="sqr_label2")
    sqr_label = tf.add(sqr_label_p1,sqr_label_p2)
    fused_cost = tf.add(sqr_fea,tf.multiply(sqr_label,delta))

    ##### regularization
    W_frobenius_norm = tf.reduce_mean(tf.pow(W,2,name='W_frobenius'))
    H_frobenius_norm = tf.reduce_mean(tf.pow(H,2,name='H_frobenius'))
    norm_sums_label = tf.add(W_frobenius_norm, H_frobenius_norm, name='norm_reg_label')
    regularizer_label = tf.multiply(norm_sums_label, lambda1, 'regularizer_label')

    cost = tf.add(tf.add(fused_cost, regularizer_fea),regularizer_label)
    #### linear prediction based regularization |UB-W| (L2 norm ? L1 norm)
    #B = tf.Variable(initial_value=tf.random_uniform([nrank,nrank],0,1), name='B')
    B = tf.constant(B_init,name='B',dtype=tf.float64)
    SB = tf.matmul(S,B)
    diff_ubw = tf.subtract(SB,M,name='diff_ubw')
    #### L2-norm 
    sqr_ubw = tf.reduce_mean(tf.multiply(tf.pow(diff_ubw,2),lambda2), name="sqr_label2")
    #### L1-norm
    #sqr_ubw = tf.reduce_mean(tf.multiply(tf.abs(diff_ubw),lambda2),name="sqr_label2")
    regularizer_ubw = tf.multiply(tf.reduce_mean(tf.pow(B,2,name='B_frobenius')),lambda3)
    cost_reg = tf.add(tf.add(cost,sqr_ubw),regularizer_ubw)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    #print 'cost function value: ' + str(sess.run(cost_reg))
    return sess.run(cost_reg)


def ComputePUV_Inductive_NonLinearTF(X,Y,U_init,V_init,W_init,H_init,B_init,ff_proj,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,lambda3,delta,kx,lr,max_iter):  
    alpha_weight = tf.constant((1.-alpha),name='alpha',dtype=tf.float64) ### 1 - alpha
    beta_weight = tf.constant((2.*alpha-1),name='beta',dtype=tf.float64) ### 2* alpha -1 
    fea_loc_x = fea_loc[0]
    fea_loc_y = fea_loc[1]
    label_loc_x = label_loc[0]
    label_loc_y = label_loc[1]
    mask = np.zeros(X.shape)
    mask[fea_loc] = 1. ## observed entries 
    labelmask = np.zeros(Y.shape)
    labelmask[label_loc] = 1.
    
    U = tf.Variable(initial_value = U_init,name='U',dtype=tf.float64)
    V = tf.Variable(initial_value = V_init.T,name='V',dtype=tf.float64)
    S = tf.matmul(U, V)
    
    debug_rs = tf.subtract(S,X,name='debug_rs')
    sqr_fea = tf.reduce_mean(tf.pow(tf.multiply(debug_rs,mask),2),name='sqr_debug')
    
    #result_flatten_fea = tf.reshape(S, [-1])
    #RS = tf.gather(result_flatten_fea, fea_loc_x * X.shape[1] + fea_loc_y, name='reconstructed_fea')
    #observed_fea = X[fea_loc]
    #diff_op_fea = tf.subtract(RS, observed_fea, name='diff_fea')
    #sqr_fea = tf.reduce_mean(tf.pow(diff_op_fea,2), name="sqr_fea")

    U_frobenius_norm = tf.reduce_mean(tf.pow(U,2,name='U_frobenius'))
    V_frobenius_norm = tf.reduce_mean(tf.pow(V,2,name='V_frobenius'))
    norm_sums_fea = tf.add(U_frobenius_norm, V_frobenius_norm, name='norm_reg_fea')
    regularizer_fea = tf.multiply(norm_sums_fea, lambda0, 'regularizer_fea')

    W = tf.Variable(initial_value = W_init,name='W',dtype=tf.float64)
    H = tf.Variable(initial_value = H_init.T,name='H',dtype=tf.float64)
    M = tf.matmul(W, H)
    #result_flatten = tf.reshape(M, [-1])
    #R = tf.gather(result_flatten, label_loc_x * Y.shape[1] + label_loc_y, name='reconstructed_entries')
    #observed_labels = Y[label_loc]
    
    #diff_op_label1 = tf.subtract(R, observed_labels, name='diff_label1')
    #sqr_label_p1 = tf.reduce_mean(tf.multiply(tf.pow(diff_op_label1,2),beta_weight), name="sqr_label1")
    R = tf.subtract(M,Y,name='res_label')
    sqr_label_p1 = tf.reduce_mean(tf.multiply(tf.pow(tf.multiply(R,labelmask),2),beta_weight), name="sqr_label1")
    #diff_op_label2 = tf.subtract(M,Y,name='diff_label2')
    sqr_label_p2 = tf.reduce_mean(tf.multiply(tf.pow(R,2),alpha_weight), name="sqr_label2")
    sqr_label = tf.add(sqr_label_p1,sqr_label_p2)
    fused_cost = tf.add(sqr_fea,tf.multiply(sqr_label,delta))

    ##### regularization
    W_frobenius_norm = tf.reduce_mean(tf.pow(W,2,name='W_frobenius'))
    H_frobenius_norm = tf.reduce_mean(tf.pow(H,2,name='H_frobenius'))
    norm_sums_label = tf.add(W_frobenius_norm, H_frobenius_norm, name='norm_reg_label')
    regularizer_label = tf.multiply(norm_sums_label, lambda1, 'regularizer_label')

    cost = tf.add(tf.add(fused_cost, regularizer_fea),regularizer_label)
    
    B = tf.Variable(initial_value=B_init,name='B',dtype=tf.float64)
    #### linear prediction based regularization |UB-W| (L2 norm ? L1 norm) 
    Cos_CoreMat = tf.cos(tf.matmul(S,tf.transpose(ffproj)))
    Sin_CoreMat = tf.sin(tf.matmul(S,tf.transpose(ffproj)))
    ##### combine cosine and sine core matrix 
    Psi =  tf.multiply(tf.concat([Cos_CoreMat, Sin_CoreMat],1),(1./np.sqrt(float(ffproj.shape[0]))),name="Psi")
    PsiB = tf.matmul(Psi,B,name="PsiM")
    diff_ubw = tf.subtract(PsiB,M,name='diff_ubw')
    #### L2-norm 
    sqr_ubw = tf.reduce_mean(tf.multiply(tf.pow(diff_ubw,2),lambda2), name="sqr_label2")
    #### L1-norm
    #sqr_ubw = tf.reduce_mean(tf.multiply(tf.abs(diff_ubw),lambda2),name="sqr_label2")
    regularizer_ubw = tf.multiply(tf.reduce_mean(tf.pow(B,2,name='B_frobenius')),lambda3)
    cost_reg = tf.add(tf.add(cost,sqr_ubw),regularizer_ubw)

    global_step = tf.Variable(0, trainable=False)
    #learning_rate = tf.train.exponential_decay(lr, global_step, 20000, 0.996, staircase=True)
    #training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_reg)
    #training_step = tf.train.RMSPropOptimizer(lr,0.9,0.0,1e-10).minimize(cost_reg)
    training_step = tf.train.AdagradOptimizer(lr).minimize(cost_reg)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    print 'initial cost function value: ' + str(sess.run(cost_reg))
    
    for i in xrange(max_iter):
        sess.run(training_step)
    
    #print 'final cost function value: ' + str(sess.run(cost_reg))
    #sess.run(cost_reg)
    #sess.run(cost)
    return  U.eval(sess), V.eval(sess), W.eval(sess), H.eval(sess), B.eval(sess)
    
def ComputePUV_Inductive_NonLinearTF_CostEval(X,Y,U_init,V_init,W_init,H_init,B_init,ff_proj,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,lambda3,delta,kx):  
    alpha_weight = tf.constant((1.-alpha),name='alpha',dtype=tf.float64) ### 1 - alpha
    beta_weight = tf.constant((2.*alpha-1),name='beta',dtype=tf.float64) ### 2* alpha -1 
    fea_loc_x = fea_loc[0]
    fea_loc_y = fea_loc[1]
    label_loc_x = label_loc[0]
    label_loc_y = label_loc[1]
    mask = np.zeros(X.shape)
    mask[fea_loc] = 1. ## observed entries 
    labelmask = np.zeros(Y.shape)
    labelmask[label_loc] = 1.
    
    U = tf.Variable(initial_value = U_init,name='U',dtype=tf.float64)
    V = tf.Variable(initial_value = V_init.T,name='V',dtype=tf.float64)
    S = tf.matmul(U, V)
    
    debug_rs = tf.subtract(S,X,name='debug_rs')
    sqr_fea = tf.reduce_mean(tf.pow(tf.multiply(debug_rs,mask),2),name='sqr_debug')
    
    #result_flatten_fea = tf.reshape(S, [-1])
    #RS = tf.gather(result_flatten_fea, fea_loc_x * X.shape[1] + fea_loc_y, name='reconstructed_fea')
    #observed_fea = X[fea_loc]
    #diff_op_fea = tf.subtract(RS, observed_fea, name='diff_fea')
    #sqr_fea = tf.reduce_mean(tf.pow(diff_op_fea,2), name="sqr_fea")

    U_frobenius_norm = tf.reduce_mean(tf.pow(U,2,name='U_frobenius'))
    V_frobenius_norm = tf.reduce_mean(tf.pow(V,2,name='V_frobenius'))
    norm_sums_fea = tf.add(U_frobenius_norm, V_frobenius_norm, name='norm_reg_fea')
    regularizer_fea = tf.multiply(norm_sums_fea, lambda0, 'regularizer_fea')

    W = tf.Variable(initial_value = W_init,name='W',dtype=tf.float64)
    H = tf.Variable(initial_value = H_init.T,name='H',dtype=tf.float64)
    M = tf.matmul(W, H)
    #result_flatten = tf.reshape(M, [-1])
    #R = tf.gather(result_flatten, label_loc_x * Y.shape[1] + label_loc_y, name='reconstructed_entries')
    #observed_labels = Y[label_loc]
    
    #diff_op_label1 = tf.subtract(R, observed_labels, name='diff_label1')
    #sqr_label_p1 = tf.reduce_mean(tf.multiply(tf.pow(diff_op_label1,2),beta_weight), name="sqr_label1")
    R = tf.subtract(M,Y,name='res_label')
    sqr_label_p1 = tf.reduce_mean(tf.multiply(tf.pow(tf.multiply(R,labelmask),2),beta_weight), name="sqr_label1")
    #diff_op_label2 = tf.subtract(M,Y,name='diff_label2')
    sqr_label_p2 = tf.reduce_mean(tf.multiply(tf.pow(R,2),alpha_weight), name="sqr_label2")
    sqr_label = tf.add(sqr_label_p1,sqr_label_p2)
    fused_cost = tf.add(sqr_fea,tf.multiply(sqr_label,delta))

    ##### regularization
    W_frobenius_norm = tf.reduce_mean(tf.pow(W,2,name='W_frobenius'))
    H_frobenius_norm = tf.reduce_mean(tf.pow(H,2,name='H_frobenius'))
    norm_sums_label = tf.add(W_frobenius_norm, H_frobenius_norm, name='norm_reg_label')
    regularizer_label = tf.multiply(norm_sums_label, lambda1, 'regularizer_label')

    cost = tf.add(tf.add(fused_cost, regularizer_fea),regularizer_label)
    
    B = tf.Variable(initial_value=B_init,name='B',dtype=tf.float64)
    #### linear prediction based regularization |UB-W| (L2 norm ? L1 norm) 
    Cos_CoreMat = tf.cos(tf.matmul(S,tf.transpose(ffproj)))
    Sin_CoreMat = tf.sin(tf.matmul(S,tf.transpose(ffproj)))
    ##### combine cosine and sine core matrix 
    Psi =  tf.multiply(tf.concat([Cos_CoreMat, Sin_CoreMat],1),(1./np.sqrt(float(ffproj.shape[0]))),name="Psi")
    PsiB = tf.matmul(Psi,B,name="PsiM")
    diff_ubw = tf.subtract(PsiB,M,name='diff_ubw')
    #### L2-norm 
    sqr_ubw = tf.reduce_mean(tf.multiply(tf.pow(diff_ubw,2),lambda2), name="sqr_label2")
    #### L1-norm
    #sqr_ubw = tf.reduce_mean(tf.multiply(tf.abs(diff_ubw),lambda2),name="sqr_label2")
    regularizer_ubw = tf.multiply(tf.reduce_mean(tf.pow(B,2,name='B_frobenius')),lambda3)
    cost_reg = tf.add(tf.add(cost,sqr_ubw),regularizer_ubw)

    global_step = tf.Variable(0, trainable=False)
    #learning_rate = tf.train.exponential_decay(lr, global_step, 20000, 0.996, staircase=True)
    #training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_reg)
    #training_step = tf.train.RMSPropOptimizer(lr,0.9,0.0,1e-10).minimize(cost_reg)
    #training_step = tf.train.AdagradOptimizer(lr).minimize(cost_reg)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    #print 'initial cost function value: ' + str(sess.run(cost_reg))
    #print 'final cost function value: ' + str(sess.run(cost_reg))
    #sess.run(cost_reg)
    #sess.run(cost)
    return sess.run(cost_reg)

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
#train_fea = yeast_data
#train_label = yeast_label

##### pproj generation 
##### p(u) = N(0,2gammaI)
#x, y = np.random.multivariate_normal(mean, cov, 5000).T
#### randomly divide into train and test set 
#mean_vector = np.zeros(train_fea.shape[1])
#cov_mat = np.eye(train_fea.shape[1])
#ffproj = np.random.multivariate_normal(mean_vector,cov_mat,500)
alpha = (1.+0.6)/2.
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
                        ##### 
                        for nn_round in range(5):
                            single_train = []
                            single_test = []
                            fea_mask = np.random.random(train_data.shape)
                            fea_loc = np.where(fea_mask < (1.-fea_fraction)) ### indexes of the observed entries 
                            fea_loc_x = fea_loc[0]
                            fea_loc_y = fea_loc[1]#######
                            mask = np.zeros(train_data.shape)
                            mask[fea_loc] = 1.
                            fea_loc_test = np.where(mask < 1)

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
                            train_label_masked = train_label.copy()
                            train_label_masked[label_loc_test] = 0. #### weak label assignments
                            U_init = np.random.random((train_data.shape[0],nrank))
                            V_init = np.random.random((train_data.shape[1],nrank))
                            W_init = np.random.random((train_label.shape[0],nrank))
                            H_init = np.random.random((train_label.shape[1],nrank))
                            B_init = np.random.random((train_data.shape[1],train_label.shape[1]))
                            
                            U,V,W,H,B = ComputePUV_Inductive_TF(train_data,train_label_masked,U_init,V_init,W_init,H_init,B_init,fea_loc,label_loc,alpha,lambda0,lambda1,lambda2,lambda3,delta,nrank,0.06,2500)
                            ##### train auc score
                            Y_reconstructed = np.dot(W,H.T)
                            ground_truth = train_label[label_loc].tolist()
                            reconstruction = Y_reconstructed[label_loc].tolist()
                            auc_score = roc_auc_score(np.array(ground_truth),np.array(reconstruction)) #### train_auc_score
                            ##### test auc socre
                            Y_test_reconstructed = np.dot(test_data,B)
                            ground_truth_test = test_label.tolist()
                            reconstruction_test = Y_test_reconstructed.tolist()
                            auc_score_test = roc_auc_score(np.array(ground_truth_test),np.array(reconstruction_test)) #### train_auc_score
                            print 'train auc: ' + str(auc_score)
                            print 'test auc: ' + str(auc_score_test)
                            single_train.append(auc_score)
                            single_test.append(auc_score_test)
                            
                        train_auc_score.append(single_train)
                        test_auc_score.append(single_test)
                        
