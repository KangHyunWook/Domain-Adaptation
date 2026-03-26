from scipy.signal import butter, lfilter

import os
import numpy as np
import scipy.io as sio
import pickle

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)
    print('data saved to', path)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5*fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b,a,data)
    return y




class DEAP:
    def __init__(self,config):
        target_data=[]
        source_data=[]
        SAVE_DATA_PATH = config.save_path
        data_root_dir= config.dataset_dir
        target_sub=config.target_sub

        target_file_name=f'target_data_{target_sub}.pkl'        
        source_file_name=f'source_data_{target_sub}.pkl'
        target_sub_data_path=os.path.join(SAVE_DATA_PATH, target_file_name)
        source_sub_data_path=os.path.join(SAVE_DATA_PATH, source_file_name)
        
        try:
            target_data = load_pickle(target_sub_data_path)
            source_data=load_pickle(source_sub_data_path)

        except:
            files = sorted(os.listdir(config.dataset_dir))
            
            for file in files:
                splits = file.split('.')
                if len(splits)<2:
                    continue
                if splits[1]!='mat':
                    continue
                s_idx = file.find('s')
                
                current_subject = str(int(file[s_idx+1:s_idx+3]))
                
                print('current subject:', current_subject) 
                data_file_path=os.path.join(data_root_dir, file)
                
                data = sio.loadmat(data_file_path)['data'][:,:,384:]

                sr = 128
                #Save current subject's preprocessed data
                preprocessed_data=[]
                for trial in range(40):

                    temp_signal = np.empty([0,data.shape[2]]) # add each frequency band
                    for channel in range(32):
                        trial_signal = data[trial, channel]

                        theta = butter_bandpass_filter(trial_signal, 4, 8, sr)
                        alpha = butter_bandpass_filter(trial_signal, 8, 14, sr)
                        beta = butter_bandpass_filter(trial_signal, 14, 31, sr)
                        gamma = butter_bandpass_filter(trial_signal, 31, 45, sr)


                        temp_signal=np.vstack([temp_signal, theta])
                        temp_signal = np.vstack([temp_signal, alpha])
                        temp_signal = np.vstack([temp_signal, beta])    
                        temp_signal = np.vstack([temp_signal, gamma])
                    #slice temp_signal into chuncks of size of 128
                    temp_signal = temp_signal.reshape(32, 4, -1)
                                
                    
                    valence_labels = sio.loadmat(data_file_path)["labels"][:,0]>5
                    arousal_labels = sio.loadmat(data_file_path)["labels"][:,1]>5
                    trial_valence_label = int(valence_labels[trial])
                    trial_arousal_label = int(arousal_labels[trial])

                    start_idx=0                
                    #get valence and arousal labels
                    dim=trial_signal.shape[0]

                    start_idx=0
                    if current_subject==target_sub:
                        for i in range(dim//sr):
                            chunk = temp_signal[:,:, i*sr:(i+1)*sr]
                            target_data.append((chunk, (trial_valence_label, trial_arousal_label)))     
                        
                    else:
                        for i in range(dim//sr):
                            chunk = temp_signal[:,:, i*sr:(i+1)*sr]
                            source_data.append((chunk, (trial_valence_label, trial_arousal_label)))
                    
            #split into  train validation test    
       
            #save to pickle subject-wise preprocessed_data
            
            to_pickle(source_data, source_sub_data_path)
            to_pickle(target_data, target_sub_data_path)

        self.source_data = source_data
        self.target_data = target_data

    def get_data(self, mode):
        if mode=='source':
            # print('s:', self.source_data[0][0].shape)
            # exit()
            return self.source_data
        elif mode=='target':
            # print('t:', self.source_data[0][0].shape)
            # exit()
            return self.target_data
        else:
            print('wrong mode')
            exit(1)