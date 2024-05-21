import numpy as np
import os
from scipy.io import loadmat
from tqdm import tqdm
from random import sample


def load_npz(path, load, label_set, data_dict,sample_rate = 0.5):
    dataset = {label: [] for label in label_set}
    datalist = os.listdir(path)
    data, labels = [], []
    for label in tqdm(label_set):
        data_path = ''
        data_name = ''
        find = False
        fault_type = data_dict[load][label]
        for d_n in datalist:
            if (str(fault_type)+'.npz') in d_n:
                data_path = os.path.join(path,d_n)
                data_name = d_n
                find = True
                break
        if find == False:print('ERROR')    
        npz_data = np.load(data_path)
        sample_list = sample(range(len(npz_data)),int(sample_rate*len(npz_data)))
        for i in sample_list:
            data_name = data_name.split('.')[0]
            image_name = data_name+'_{'+str(i)+'}'
            complexdata = npz_data[image_name]
            real = np.real(complexdata)
            imag = np.imag(complexdata)
            data.append(np.expand_dims(np.concatenate((real,imag),axis=0),axis=0))
            labels.append(label)

        

    

    

    return np.array(data,dtype=np.float32), np.array(labels,dtype=np.float32)
