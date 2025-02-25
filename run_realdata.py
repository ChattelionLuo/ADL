import numpy as np
import multiprocessing
import os

from scipy import sparse
import adl_realdata

family = "binomial"
penalty = "l1"
X_total=sparse.load_npz('bigram_X.npz')
y_total=np.load('bigram_y.npy')

feature_list=[6795,7856,22608]

'''
('investment',) 6795
('schedule',) 7856
('per', 'cent') 22608
'''

def worker1(num):    #feature search worker

    print(f'worker {num} started')
    id=feature_list[num]
    
    model=adl_realdata.on_ADL([id],family,penalty)
    folder_name = "./realdata_result/subset%s" % (model.subset[0])
    os.makedirs(folder_name) # create the folder in the current directory

    model.fit(X_total,y_total,folder_name)

    np.save(os.path.join(folder_name, 'lb%s.npy'%model.subset[0]), model.lb_trajec)
    np.save(os.path.join(folder_name, 'ub%s.npy'%model.subset[0]), model.ub_trajec)
    np.save(os.path.join(folder_name, 'debeta%s.npy'%model.subset[0]), model.debeta_trajec)
    np.save(os.path.join(folder_name, 'tao%s.npy'%model.subset[0]), model.tao_trajec)
    np.save(os.path.join(folder_name, 'beta%s.npy'%model.subset[0]), model.beta_trajec)
    np.save(os.path.join(folder_name, 'pred.npy'), np.array(model.pred_err))

    print(f'worker {num} finised')

    return model

if __name__ == '__main__':

    processes=[]
    for i in range(len(feature_list)):
        p = multiprocessing.Process(target=worker1,args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("finished")
