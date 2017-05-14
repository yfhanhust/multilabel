import pickle 
import numpy as np 

with open('pickle.dat','rb') as f:
     results = pickle.load(f)

gd_reconstruction_error_list = results[0]
gd_auc_score_list = results[1]
reconstruction_error_list = results[2]
auc_score_list = results[3]

##### baseline 1: matrix factorisation based completion
gd_reconstruction_error = np.zeros((192,10),dtype=float)
##### baseline 2: PU matrix completion 
gd_auc_score = np.zeros((192,10),dtype=float)

##### Evalation metric of our proposed method
##### 192 different parameter settings, for each setting we run 10 times to obtain average and variance statistics. 
reconstruction_error = np.zeros((192,10),dtype=float)
auc_score = np.zeros((192,10),dtype=float)


for k in range(1,193):
    istart = (k-1)*10
    iend = k*10
    gd_reconstruction_error[(k-1),:] = gd_reconstruction_error_list[istart:iend]
    gd_auc_score[(k-1),:] = gd_auc_score_list[istart:iend]
    reconstruction_error[(k-1),:] = reconstruction_error_list[istart:iend]
    auc_score[(k-1),:] = auc_score_list[istart:iend]
    
mean_gd_reconstruction_error = np.mean(gd_reconstruction_error,axis=1)
mean_gd_auc_score = np.mean(gd_auc_score,axis=1)
mean_reconstruction_error = np.mean(reconstruction_error,axis=1)
mean_auc_score = np.mean(auc_score,axis=1)
    
import matplotlib.pyplot as plt
plt.plot(range(192),mean_gd_reconstruction_error)
plt.plot(range(192),mean_reconstruction_error)
plt.show()


plt.plot(range(192),mean_gd_auc_score,'b')
plt.plot(range(192),mean_auc_score,'r')
plt.show()

    
