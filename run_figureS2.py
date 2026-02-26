import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
import pandas as pd
from scipy.stats import chi2

import cal
import adl

n_sim = 2000   #number of simulations

# GLM objective
family = "binomial" #"gaussian"
penalty = "l1"

# for data generation
corst_x = "AR-1"    #"ind", "AR-1", "toeplitz"
rho_x = 0.5     #AR-1 correlation 

factor=1 # 1 * Sigma

beta01 = 1
beta02 = -1

folder_name = "figureS2"
os.mkdir(folder_name)

alpha = 0.05
beta_alt =[-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9]
n_list = [100,200,300,400,500]

for n in n_list:
    
    folder_name = "figureS2/n%s" % (n)
    os.mkdir(folder_name)
    for num in range(len(beta_alt)):
        
        rejected = np.zeros((n_sim,len(beta_alt)) , dtype=bool)
        for u in range(n_sim):
            
            beta_a=cal.betagenerator(500,6,beta01,beta02)
            beta_a[230] = beta_alt[num]  
            X,X_total,y=cal.datagenerator(n, 500, 1, corst_x, rho_x, beta_a,family,np.sqrt(factor),u)

            model=adl.on_ADL([230],family,beta_a,penalty)
            start_time = time.time()
            model.fit( X,y )
            print("---n%s, beta_a %s, %s-th simulation, %s seconds ---" % (n, beta_alt[num], u+1 ,(time.time() - start_time)) )
            wald_stat = (((model.beta_de-1)/model.tao)**2).item()
            
            p_value = 1 - chi2.cdf(wald_stat, df=1)
            rejected[u,num] = (p_value < alpha)
        np.save(os.path.join(folder_name, 'slice%s.npy'%(num)), rejected[:,num])
        
    for k in range(len(beta_alt)): 
        rejected[:,k] = np.load(os.path.join(folder_name, 'slice%s.npy'%(k)))
    np.save(os.path.join(folder_name, 'rejected.npy'), rejected)

all_yi = []

for i in n_list:
    yi = np.mean(np.load(f'figureS2/n{i}/rejected.npy'), axis=0)
    all_yi.append(yi)

plt.figure(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(n_list)))

for idx, (yi, sample_size) in enumerate(zip(all_yi, n_list)):
    plt.plot(beta_alt, yi, label=r'$n='+str(sample_size)+'$', color=colors[idx], linewidth=2)

plt.xticks(np.arange(-0.5, 1.0, 0.2),fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'$\beta_a$',fontsize=14)
plt.ylabel('Power',fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'figureS2/power_curve.png', dpi=600)
print("finished")