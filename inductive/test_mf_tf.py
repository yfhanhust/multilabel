####### test matrix factorization using tensorflow 
import numpy as np 
import scipy as sp 
import tensorflow as tf 


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

fea_fraction = 0.6
fea_mask = np.random.random(yeast_data.shape)
fea_loc = np.where(fea_mask < (1.-fea_fraction)) ### indexes of the observed entries 
test_mat = yeast_data[fea_loc]
##### 
fea_loc_x = fea_loc[0]
fea_loc_y = fea_loc[1]

nrank = 10
U = tf.Variable(initial_value=tf.truncated_normal([yeast_data.shape[0],nrank]), name='users')
V = tf.Variable(initial_value=tf.truncated_normal([nrank,yeast_data.shape[1]]), name='items')
reconstructed_mat = tf.matmul(U, V)
result_flatten = tf.reshape(reconstructed_mat, [-1])
R = tf.gather(result_flatten, fea_loc_x * tf.shape(reconstructed_mat)[1] + fea_loc_y, name='reconstructed_entries')
diff_op = tf.subtract(R, test_mat, name='trainig_diff')
diff_op_squared = tf.abs(diff_op, name="squared_difference")
base_cost = tf.reduce_sum(diff_op_squared, name="sum_squared_error")
lda = tf.constant(.1, name='lambda')
norm_sums = tf.add(tf.reduce_sum(tf.abs(U, name='user_abs'), name='user_norm'), 
   tf.reduce_sum(tf.abs(V, name='item_abs'), name='item_norm'))
regularizer = tf.multiply(norm_sums, lda, 'regularizer')
cost = tf.add(base_cost, regularizer)
lr = tf.constant(.05, name='learning_rate')
global_step = tf.Variable(0, trainable=False)
optimizer = tf.train.RMSPropOptimizer(lr)
#learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_step = optimizer.minimize(cost, global_step=global_step)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
for i in xrange(500):
    sess.run(training_step)
    
#### evaluate reconstruction accuracy 
U_mat = U.eval(sess)
V_mat = V.eval(sess)
yeast_reconstruction = np.dot(U_mat,V_mat)
mask = np.zeros(yeast_data.shape)
mask[fea_loc] = 1.
fea_loc_test = np.where(mask <1)
print np.linalg.norm(yeast_data[fea_loc_test] - yeast_reconstruction[fea_loc_test])