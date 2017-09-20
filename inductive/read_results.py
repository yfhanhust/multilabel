import numpy as np 
import scipy as sp 
from sklearn.linear_model import Ridge 
import downhill
import theano
from sklearn.metrics import roc_auc_score

import pickle
import numpy as np 

result_file_name = 'I_pickle_yeast_l.dat'
# with open(result_file_name,'wb') as f:
#    pickle.dump([gd_reconstruction_error_list,gd_auc_score_list,reconstruction_error_list,auc_score_list,parameter_list,fixed_parameter],f)

with open(result_file_name,'rb') as f:
    [train_auc_score, test_auc_score] = pickle.load(f)

parameter_list=[]
for lambda0 in [0.001,0.01,0.1,1]:
    for lambda1 in [0.001,0.01,0.1,1]:
        for lambda2 in [0.001,0.01,0.1,1]:
            for lambda3 in [0.001,0.01,0.1,1]:
                for delta in [0.05,0.1,0.5,1]:
                    parameter_list.append([lambda0,lambda1,lambda2,lambda3,delta])

#### mean and variance 
train_auc_score_mean=[]
train_auc_score_var=[]
test_auc_score_mean=[]
test_auc_score_var=[]

print len(test_auc_score)

num_par=1024
train_auc_score=np.array(train_auc_score).reshape(10,num_par)
test_auc_score=np.array(test_auc_score).reshape(10,num_par)

# print (test_auc_score).shape[1]

for i in range((test_auc_score).shape[1]):
    train_auc_score_mean.append(np.mean(train_auc_score[:,i]))
    train_auc_score_var.append(np.std(train_auc_score[:,i]))
    test_auc_score_mean.append(np.mean(test_auc_score[:,i]))
    test_auc_score_var.append(np.std(test_auc_score[:,i]))

print "Maximum test auc: "+str(max(test_auc_score_mean))
print "Corresponding parameter setting: "+str(parameter_list[test_auc_score_mean.index(max(test_auc_score_mean))])

import matplotlib.pyplot as plt 
plt.errorbar(range(len(train_auc_score_mean)),train_auc_score_mean, yerr=train_auc_score_var,color='r',label='Train',linewidth=5, elinewidth=1)
plt.errorbar(range(len(test_auc_score_mean)),test_auc_score_mean,yerr=test_auc_score_var,color='b',label='Test',linewidth=5, elinewidth=1)
plt.xlabel('Yeast Different Parameter Setting')
plt.ylabel('Yeast Area-Under-Curve Score')
plt.legend()
plt.show()