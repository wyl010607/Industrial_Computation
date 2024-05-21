import pandas as pd
from scipy.io import loadmat
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import pywt

source_data_path = 'D:\graduation_project\CaseWesternReserveUniversityData-master\CaseWesternReserveUniversityData-master'
data_list =  os.listdir(source_data_path)
data_path = 'D:\graduation_project\CWRUdata'
sampling_period = 1.0/12000
totalscal = 128
wavename = 'cmor1-1'
step = 512

#print(data_list)

for mat_name in data_list:
    mat_path = source_data_path +'\\'+ mat_name
    data_folder_name = os.path.splitext(mat_name)[0]
    data_folder_path = data_path +'\\'+ data_folder_name
    os.makedirs(data_folder_path,exist_ok=True)
    mat_data = loadmat(mat_path)
    variables = mat_data.keys()
    for var_name in variables:
        if '_DE_time' in var_name:
            data = mat_data[var_name].reshape(-1)
            
            fc = pywt.central_frequency(wavename)
            cparam = 2*fc*totalscal
            scales = cparam/np.arange(totalscal,0,-1)
            coefficients , frequencies = pywt.cwt(data,scales,wavename,sampling_period)
            amp = abs(coefficients)
            frequ_max = frequencies.max()

            num_columns = coefficients.shape[1]
            num_steps = num_columns // step
            array_dict = {}
            for i in range(num_steps):
                subset = coefficients[:,i*step:(i+1)*step]
                file_name = data_folder_name + '_{'+str(i)+'}'
               # file_path =data_folder_path+"\\"+file_name
               # np.save(file_path,subset)
                array_dict[file_name] = subset
            
            npz_file_name = data_folder_name + '.npz'
            npz_file_path = data_folder_path + '.npz'
            np.savez(npz_file_path,**array_dict)

            #t = np.linspace(0,sampling_period,data.size,endpoint=False) 
            #plt.contourf(t,frequencies,amp,cmap='jet')
            #plt.show()



            
            

