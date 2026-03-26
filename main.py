'''
Program Description: implement DGCNN and tarin on subject data on DEAP
'''

from data_loader import get_loader
from sklearn.metrics import accuracy_score, f1_score
from config import get_config
from solver import Solver
from random import random

import os
import pickle
import numpy as np
import torch

if __name__=='__main__':
    random_name = str(random())
    random_seed = 123
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


    source_config=get_config(mode='source')
    target_config=get_config(mode='target')
    print(source_config.mode)
    print(source_config.dataset_dir)
    #todo: return target data and source data
    temp_n_files=3

    accs=[]
    test_acc_list=[]
    test_f1_list=[]

    for target_sub in range(1,33):
        print('target subject:', target_sub)
        source_config.target_sub=target_config.target_sub=str(target_sub)
        device = torch.device(f"cuda:{source_config.gpu_no}" if torch.cuda.is_available()  else "cpu")
        #todo:
        source_dataloader = get_loader(source_config, shuffle=True)
        target_dataloader = get_loader(target_config, shuffle=True)

        solver=Solver
        solver = solver(source_config,target_config, source_dataloader, target_dataloader, device, is_train=True)
        solver.build()
        solver.train()
        
        test_loss, test_true, test_pred = solver.eval(mode="target", to_print=True)

        test_acc=accuracy_score(test_true, test_pred)*100
        test_f1 = f1_score(test_true, test_pred, average = 'weighted')*100

        print('test acc:', test_acc)
        print('test F1:', test_f1)

        test_acc_list.append(test_acc)
        test_f1_list.append(test_f1)
        

    avg_cls_acc=np.round(np.mean(test_acc_list),3)    
    avg_f1 = np.round(np.mean(test_f1_list),3)

    print('test accs for all subjects')
    for i in range(len(test_acc_list)):
        print(f'Sub {i+1} test_acc:', test_acc_list[i])

    print(f'avg cls acc: {avg_cls_acc} / {np.round(np.std(test_acc_list),3)}')
    print(f'avg F1: {avg_f1} / {np.round(np.std(test_f1_list),3)}')


