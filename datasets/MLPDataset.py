import numpy as np
import torch
import torch.nn as nn
import os

import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
from .abs import AbstractDataset


class MLPDataset(AbstractDataset):

    @staticmethod
    def clustering_operations(dataset_list):
        df_list = []
        for dataset in dataset_list:
            df_list.append(dataset.df)

        full_df = pd.concat(df_list)
        os_types = KMeans(n_clusters=6, random_state=1).fit_predict(full_df[['os1', 'os2']].values)
        full_df.insert(2, 'os_type', os_types)
        start = 0
        for i, df in enumerate(df_list):
            dataset = dataset_list[i]
            df_len = len(dataset.df)
            # print('loc', start, df_len)
            dataset.df = full_df.iloc[start: start + df_len]
            dataset.has_cluster_operations = True
            start += df_len

    @staticmethod
    def smooth(s, b):
        v = np.zeros(len(s)+1) #v_0 is already 0.
        bc = np.zeros(len(s)+1)
        for i in range(1, len(v)): #v_t = 0.95
            v[i] = (b * v[i-1] + (1-b) * s[i-1]) 
            bc[i] = 1 - b**i
        sm = v[1:] / bc[1:]
        return sm

    def __init__(self, indices, df,is_test):
        
        self.indices = indices
        self.df = df
        self.is_test = is_test
        
    def __len__(self):
        
        return len(self.indices)
    
    def __getitem__(self, idx):
        if not self.is_test:
          ind = self.indices[idx]
          X_ = self.df.iloc[ind, :].drop(['time','unit','rul']).copy().to_numpy()
          y_ = self.df.iloc[ind]['rul']
        else:
          n = self.indices[idx]
          U = self.df[self.df['unit'] == n].copy()
          X_ = U.reset_index().iloc[-1:].drop(['time','index','unit','rul'], axis = 1).copy().to_numpy()
          y_ = U['rul'].min()
        return X_, y_

    

    @staticmethod
    def getDataloader(data_path, sub_dataset):
        df_train = pd.read_csv(os.path.join(data_path, 'train_{:s}.txt'.format(sub_dataset)), sep=' ',header=None)
        df_test = pd.read_csv(os.path.join(data_path, 'test_{:s}.txt'.format(sub_dataset)), sep=' ', header=None)
        rul_test = pd.read_csv(os.path.join(data_path, 'RUL_{:s}.txt'.format(sub_dataset)), header=None)
        col_names = []

        col_names.append('unit')
        col_names.append('time')

        for i in range(1,4):
            col_names.append('os'+str(i))
        for i in range(1,22):
            col_names.append('s'+str(i))

        df_train = df_train.iloc[:,:-2].copy()
        df_train.columns = col_names
        df_test = df_test.iloc[:,:-2].copy()
        df_test.columns = col_names


        rul_list = []
        engine_numbers = max(df_train['unit'])
        for n in np.arange(1,engine_numbers+1):
            
            time_list = np.array(df_train[df_train['unit'] == n]['time'])
            length = len(time_list)
            rul = list(length - time_list)
            rul_list += rul
            
        df_train['rul'] = rul_list

        rul_list = []

        engine_numbers_test = max(df_test['unit'])
        print(engine_numbers_test)
        for n in np.arange(1,engine_numbers_test+1):
            
            time_list = np.array(df_test[df_test['unit'] == n]['time'])
            length = len(time_list)
            rul_val = rul_test.iloc[n-1].item()
            rul = list(length - time_list + rul_val)
            rul_list += rul

        df_test['rul'] = rul_list
 


        drop_cols1=[]
        
        if sub_dataset in ['FD001', 'FD003']:
            drop_cols1 = ['os3','s1','s5','s6','s10','s16','s18','s19']

        df_train = df_train.drop(drop_cols1, axis = 1)
        df_test = df_test.drop(drop_cols1, axis = 1)

        minmax_dict = {}

        for c in df_train.columns:
            if 's' in c:
                minmax_dict[c+'min'] = df_train[c].min()
                minmax_dict[c+'max']=  df_train[c].max()

               
        for c in df_train.columns:
            if 's' in c:
                df_train[c] = (df_train[c] - minmax_dict[c+'min']) / (minmax_dict[c+'max'] - minmax_dict[c+'min'])
        
        
        for c in df_test.columns:
            if 's' in c:
                df_test[c] = (df_test[c] - minmax_dict[c+'min']) / (minmax_dict[c+'max'] - minmax_dict[c+'min'])

        # minmax_dict['rul'+'min'] = df_train['rul'].min()
        # minmax_dict['rul'+'max']=  df_train['rul'].max() 
        # df_train['rul'] = (df_train['rul'] - minmax_dict['rul'+'min']) / (minmax_dict['rul'+'max'] - minmax_dict['rul'+'min'])
        # minmax_dict['rul1'+'min'] = df_test['rul'].min()
        # minmax_dict['rul1'+'max']=  df_test['rul'].max() 
        # df_test['rul'] = (df_test['rul'] - minmax_dict['rul1'+'min']) / (minmax_dict['rul1'+'max'] - minmax_dict['rul1'+'min'])
        # for df in [df_train, df_test]:
        #     display(df.head())

        for c in df_train.columns:
      
            if 's' in c:
                sm_list = []

                for n in np.arange(1,engine_numbers+1):
                    s = np.array(df_train[df_train['unit'] == n][c].copy())
                    sm = list(MLPDataset.smooth(s, 0.98))
                    sm_list += sm
                
                df_train[c+'_smoothed'] = sm_list
                
        for c in df_test.columns:
            
            if 's' in c:
                sm_list = []

                for n in np.arange(1,engine_numbers_test+1):
                    s = np.array(df_test[df_test['unit'] == n][c].copy())
                    sm = list(MLPDataset.smooth(s, 0.98))
                    sm_list += sm
                
                df_test[c+'_smoothed'] = sm_list
        for c in df_train.columns:
              if ('s' in c) and ('smoothed' not in c):
                df_train[c] = df_train[c+'_smoothed']
                df_train.drop(c+'_smoothed', axis = 1, inplace = True)
                
        for c in df_test.columns:
            if ('s' in c) and ('smoothed' not in c):
                df_test[c] = df_test[c+'_smoothed']
                df_test.drop(c+'_smoothed', axis = 1, inplace = True)
        
        # for df in [df_train, df_test]:
        #     print(df.head())
        #     print(df.tail())
 
  
        n_features = len([c for c in df_train.columns if 's' in c]) #plus one for time
        # print(n_features)
        window = 20
        np.random.seed(5)
        units = np.arange(1,engine_numbers+1) 
        train_units = list(np.random.choice(units, 80, replace = False))
        val_units = list(set(units) - set(train_units))

        train_data = df_train[df_train['unit'].isin(train_units)].copy()
        val_data = df_train[df_train['unit'].isin(val_units)].copy()

    

        train_indices = list(train_data[(train_data['rul'] >= 0) & (train_data['time'] > 10)].index)
        val_indices = list(val_data[(val_data['rul'] >= 0) & (val_data['time'] > 10)].index)

        units=np.arange(1,engine_numbers_test+1)
        train = MLPDataset(train_indices, df_train,False)
        #print(len(train))
        val = MLPDataset(val_indices, df_train,False)
        test = MLPDataset(units, df_test,True)
        dataset_list = [train, test,val]
        os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())
        # MLPDataset.clustering_operations(dataset_list)
        trainloader = DataLoader(train, batch_size = 64, shuffle = True)
        valloader = DataLoader(val, batch_size = len(val_indices), shuffle = True)
        testloader = DataLoader(test, batch_size = 1)

        dataiter = iter(valloader)
        x,y = next(dataiter)
        print(x.shape)
        print(y.shape)


        return trainloader,testloader,valloader,minmax_dict


    