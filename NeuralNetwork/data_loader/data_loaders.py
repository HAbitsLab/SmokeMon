from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
import sys, os
import datetime
from functools import partial
from multiprocessing import Pool, Lock

from skimage.util import random_noise

class SmokeMonDataLoader(BaseDataLoader):
    """
    SmokeMon DataLoader
    """
    def __init__(self, in_lab_dir, in_wild_dir, in_lab_test_id=None, in_wild_test_id=None , in_lab_droprate=0.5, in_wild_droprate=0.5, batch_size=64, shuffle=True, validation_split=0.15, num_workers=2, training=True):
        """
        in_lab_dir: directory containing in-lab data
        in_wild_dir: directory containing in-wild data
        in_wild_test_id: in-wild participant to leave out
        """
        training_trsfm = transforms.Compose([
            transforms.Lambda(lambda img: random_noise(img,clip=False,var=0.0004).astype('float32')),
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(0, translate=(0,0.1), fill=0), 
            transforms.ToTensor(),
            #transforms.Normalize(25, 6, inplace=True)
        ])
        trsfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            #transforms.Normalize(25, 6, inplace=True)
        ])
        #in_lab_list=['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
        # todo: add these to the config file
        in_lab_list=['P6']
        in_wild_list=['P0_filtered', 'P1_filtered', 'P2_filtered', 'P3_filtered', 'P4_filtered']
        #in_wild_list=[]
        print(f'Effective in-lab data drop rate is {in_lab_droprate}')
        print(f'Effective in-wild data drop rate is {in_wild_droprate}')
        #cleaning input
        if in_lab_test_id is not None and (in_lab_test_id < 0 or in_lab_test_id >= len(in_lab_list)):
            in_lab_test_id=None
        if in_wild_test_id is not None and (in_wild_test_id < 0 or in_wild_test_id >= len(in_wild_list)):
            in_wild_test_id=None
            
        if (training):
            transformation=training_trsfm
        else:
            transformation=trsfm
        self.dataset = SmokeMonDataset(in_lab_dir, in_wild_dir, in_lab_list, in_wild_list, in_lab_test_id, in_wild_test_id, in_lab_droprate=in_lab_droprate, in_wild_droprate=in_wild_droprate,  transform=transformation, training=training, num_workers=num_workers)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

def get_data(p, in_lab_dir):
    labels_ts = pd.read_csv(os.path.join(in_lab_dir, p, "CP_ts.csv"), index_col=0).reset_index(drop=True)
    mlx_data = pd.read_csv(os.path.join(in_lab_dir, p, "mlx.csv"), header=None, index_col=0).clip(upper=120, lower=0)
    index=pd.DataFrame(np.array(mlx_data.index//10000000), columns=['index#1'])
    index['index#2']=mlx_data.index%10000000
    mlx_data=mlx_data.reset_index(drop=True)

    if labels_ts.shape[0] > mlx_data.shape[0]:
        labels_ts = labels_ts[:mlx_data.shape[0]]

    labels_ts=labels_ts[['label']]

    #only for in-lab data, filter out 0-filled entries
    ValidDataIndex=[i for i in range(mlx_data.shape[0]) if mlx_data.iloc[i].sum()>0.1]

    #masking three edges
    mask_list=[i*32+32 for i in range(24)]+[i*32+31 for i in range(24)]+[i*32+30 for i in range(24)]+[i*32+1 for i in range(24)]+[i*32+2 for i in range(24)]+[i*32+3 for i in range(24)]+[*range(609,769)]
    mlx_data.drop(labels=mask_list, axis=1, inplace=True)
    dataDense=mlx_data.iloc[ValidDataIndex]
    dataDense=dataDense.div(dataDense.mean(axis=1), axis=0)    
    return labels_ts.iloc[ValidDataIndex], dataDense, index.iloc[ValidDataIndex]

def get_wild_data(p, in_wild_dir,):
    dataDense=pd.DataFrame()
    labelDense=pd.DataFrame()
    indexDense=pd.DataFrame()
    print(f'Loading from path of {p}')
    if p == "P0" or p == 'P0_filtered':
        data_path = os.path.join(in_wild_dir, p, "clean/17/17.csv")
        label_path = os.path.join(in_wild_dir, p, "clean/17/CP_ts.csv")
        data=pd.read_csv(data_path, header=None, index_col=0).clip(upper=120, lower=0)
        data=data.dropna(thresh=50).fillna(20).iloc[10:]
        index=pd.DataFrame(np.array(data.index//10000000), columns=['index#1'])
        index['index#2']=data.index%10000000
        data=data.reset_index(drop=True)
        labels_ts = pd.read_csv(label_path, index_col=0).iloc[10:].reset_index(drop=True)
        mean=data.mean(axis=1)
        data=data[mean<50]
        index=index[mean<50]
        labels_ts=labels_ts[mean<50]
        dataDense=pd.concat([dataDense, data],ignore_index=True)
        labelDense=pd.concat([labelDense, labels_ts],ignore_index=True)
        indexDense=pd.concat([indexDense, index],ignore_index=True)
        mask_list=[i*32+32 for i in range(24)]+[i*32+31 for i in range(24)]+[i*32+30 for i in range(24)]+[i*32+1 for i in range(24)]+[i*32+2 for i in range(24)]+[i*32+3 for i in range(24)]+[*range(609,769)]
        dataDense.drop(labels=mask_list, axis=1, inplace=True)
    else:      
        partial_path = os.path.join(in_wild_dir, p, "clean")
        data_dir_list= np.sort([i for i in os.listdir(partial_path) if not i.startswith('.')])
        for data_dir in data_dir_list:
            
            if not data_dir.startswith('.'):
                final_path = os.path.join(partial_path, data_dir)
                #print(f'Loading from path {final_path}')
                data_name = [i for i in os.listdir(final_path) if i.startswith('2021')][0]
                label_name = [i for i in os.listdir(final_path) if i.startswith('CP')][0]
                
                #Data inversion for all subjects not P0
                data=pd.read_csv(os.path.join(final_path, data_name), header=None, index_col=0).clip(upper=120, lower=0)
                data=data.dropna(thresh=50).fillna(20).iloc[10:, ::-1]
                index=pd.DataFrame(np.array(data.index//10000000), columns=['index#1'])
                index['index#2']=data.index%10000000
                data=data.reset_index(drop=True)
                labels_ts = pd.read_csv(os.path.join(final_path, label_name), index_col=0).iloc[10:].reset_index(drop=True)
                mean=data.mean(axis=1)
                data=data[mean<50]
                index=index[mean<50]
                labels_ts=labels_ts[mean<50]
                dataDense=pd.concat([dataDense, data],ignore_index=True)
                labelDense=pd.concat([labelDense, labels_ts],ignore_index=True)
                indexDense=pd.concat([indexDense, index],ignore_index=True)
                #masking three edges
                mask_list=[i*32+32 for i in range(24)]+[i*32+31 for i in range(24)]+[i*32+30 for i in range(24)]+[i*32+1 for i in range(24)]+[i*32+2 for i in range(24)]+[i*32+3 for i in range(24)]+[*range(1,161)]
                dataDense.drop(labels=mask_list, axis=1, inplace=True)
    assert(len(labelDense)==len(dataDense))
    assert(len(labelDense)==len(indexDense))
    dataDense=dataDense.div(dataDense.mean(axis=1), axis=0)
    return labelDense[['label']], dataDense, indexDense


#Caculate Moving Avergaes, along with other features
#return maxVal, (normalized)max, (normalized)maxAvg, vertical dist, 
#MvAvg vertical dist, horizontal dist, MvAvg horizontal dist
def FeatureCalc(data_entry, momentum=0.8):
    label, data, index=data_entry
    
    #std=data.std(axis=1)   
    #mean_alt=data.mean(axis=1)
    #data=data.subtract(mean, axis=0).div(std, axis=0).clip(upper=5, lower=-3)

    MvAvgMax=np.zeros(data.shape[0])
    MvAvgIndexVertical=np.zeros(data.shape[0])
    MvAvgIndexHorizontal=np.zeros(data.shape[0])

    IndexVertical=np.zeros(data.shape[0])
    IndexHorizontal=np.zeros(data.shape[0])
    deltaVertical=np.zeros(data.shape[0])
    deltaHorizontal=np.zeros(data.shape[0])
    deltaMax=np.zeros(data.shape[0])
    
    Max=data.max(axis=1)
    maxIndex=data.values.argmax(axis=1)
    for i in range(data.shape[0]):
        #For obvious ROI's
        threshold=1.3
        if(Max.iloc[i]>threshold):
            #from 1d maxIndex, calculate 2d maxIndex given h x w = (19, 26)
            IndexVertical[i]=maxIndex[i]//26
            horizontalIndex=maxIndex[i]%26
            if(horizontalIndex>=13):
                IndexHorizontal[i]=horizontalIndex-13
            else:
                IndexHorizontal[i]=13-horizontalIndex
        #Cigarette out of frame
        else:
            IndexVertical[i]=17+np.random.rand()*2
            IndexHorizontal[i]=11+np.random.rand()*2

        
        #calculate movingAvg
        if(i==0):
            MvAvgMax[i]=Max.iloc[i]
            MvAvgIndexVertical[i]=IndexVertical[i]
            MvAvgIndexHorizontal[i]=IndexHorizontal[i]
            deltaHorizontal[i]=0
            deltaVertical[i]=0
            deltaMax[i]=0
        else:
            MvAvgMax[i]=(1-momentum)*Max.iloc[i]+momentum*MvAvgMax[i-1]
            MvAvgIndexVertical[i]=(1-momentum)*IndexVertical[i]+momentum*MvAvgIndexVertical[i-1]
            MvAvgIndexHorizontal[i]=(1-momentum)*IndexHorizontal[i]+momentum*MvAvgIndexHorizontal[i-1]
            deltaHorizontal[i]=IndexHorizontal[i]-IndexHorizontal[i-1]
            deltaVertical[i]=IndexVertical[i]-IndexVertical[i-1]
            deltaMax[i]=Max.iloc[i]-Max.iloc[i-1]
         
        #normalize data
        #data=data.div(mean_alt, axis=0)
        
        #append series to dataframe
        #data['dmx']=deltaMax/120
        #data['didxV']=deltaVertical/20
        #data['didxH']=deltaHorizontal/14
        data['idxV']=IndexVertical/20
        data['idxH']=IndexHorizontal/14
        data['idxDist']=(IndexHorizontal**2+IndexVertical**2)**0.5/24
        data['max']=Max
        
    return label, data, index


#need some further tweaking
def balanceData(data_entry, training=True, droprate=0.5):
    """
    balance data if data is prepared for training
    return labels, data, count-positive, count-negative
    """
    labels, data, indexes=data_entry
    countNegative=0
    countPositive=0
    if training:

        #drop all data entries with max temperature lower than 40C
        #"Unintereting" data
        #index=data[(data['max']<=40)].index
        #data.drop(index, inplace=True)
        #labels.drop(index, inplace=True)
        #indexes.drop(index, inplace=True)
        
        #randomly drop negative cased to balance dataset
        labels['rand']=np.random.random(len(labels))
        index=labels[(labels['label']==0) & (labels['rand']<=droprate)].index
        data.drop(index, inplace=True)
        labels.drop(index, inplace=True)
        labels.drop(columns=['rand'], inplace=True)
        indexes.drop(index, inplace=True)

        countPos=len(labels[labels['label']>0.1])
        countNeg=len(labels[labels['label']<0.1])
        data=np.array(data).astype('float32')
        labels=np.array(labels).astype('float32')
        return labels, data, indexes, countPos, countNeg
    else:
        return np.array(labels).astype('float32') , np.array(data).astype('float32'), indexes, 0, 0

def combo_in_wild_get_data(p, in_wild_dir, droprate=0.7):
    """
    Wrapper for in-wild get-data
    """
    return balanceData(FeatureCalc(get_wild_data(p, in_wild_dir)), droprate=droprate)

#Data Loading & Formatting
class SmokeMonDataset(Dataset):
    def __init__(self, in_lab_dir, in_wild_dir, in_lab_list, in_wild_list, in_lab_test_id=None, in_wild_test_id=None, in_lab_droprate=0.5, in_wild_droprate=0.5, transform=None, training=True, num_workers=2):
        """
        Return a training/validating dataset without test
        p: index
        """
        self.transform=transform
        self.in_lab_dir=in_lab_dir
        self.in_wild_dir=in_wild_dir
        self.in_lab_list=in_lab_list
        self.in_wild_list=in_wild_list
        self.in_lab_test_id=in_lab_test_id
        self.in_wild_test_id=in_wild_test_id
        mlx_data=[]
        labels_ts=[]
        indexes_holder=[]
        totalPos=0
        totalNeg=0
        lock=Lock()
        if training:
            in_lab_train_list=self.in_lab_list if self.in_lab_test_id is None else list(set(self.in_lab_list)-set([self.in_lab_list[self.in_lab_test_id]]))
            in_wild_train_list=self.in_wild_list if self.in_wild_test_id is None else list(set(self.in_wild_list)-set([self.in_wild_list[self.in_wild_test_id]]))
            print(f'Using {in_lab_train_list} as in-lab training dataset')
            print(f'Using {in_wild_train_list} as in-wild training dataset')
            
            #single threaded version
            for p in in_wild_train_list:
                labels, data, indexes,countPos, countNeg=balanceData(FeatureCalc(get_wild_data(p, in_wild_dir)), droprate=in_wild_droprate)
                #print(f'Done processing {p} at time {datetime.datetime.now()}')
                totalPos+=countPos
                totalNeg+=countNeg
                mlx_data.append(data)
                labels_ts.append(labels)
                indexes_holder.append(indexes)
            
            #for multiprocessing
            #partial_combo_in_wild_get_data = partial(combo_in_wild_get_data, in_wild_dir=in_wild_dir, droprate=in_wild_droprate)
            #pool=Pool(num_workers)
            #def harvest_result(result):
            #    if(not result is None):
            #    
            #        labels, data, indexes, countPos, countNeg=result
            #        totalPos+=countPos
            #        totalNeg+=countNeg
            #        mlx_data.append(data)
            #        labels_ts.append(labels)
            #        indexes_holder.append(indexes)
                
            #for p in in_wild_train_list:
                #pool.apply_async(partial_combo_in_wild_get_data, args=(p, ), callback=harvest_result)    
            #pool.close()
                      
            for p in in_lab_train_list:
                labels, data, indexes, countPos, countNeg=balanceData(FeatureCalc(get_data(p, in_lab_dir)), droprate=in_lab_droprate)
                totalPos+=countPos
                totalNeg+=countNeg
                mlx_data.append(data)
                labels_ts.append(labels)
                indexes_holder.append(indexes)
            #pool.join()
            
            
            print(f'Total positive to negative ratio in training data is {totalPos} : {totalNeg}')
            self.mlx_data=np.concatenate(mlx_data, axis=0)
            self.labels_ts=np.concatenate(labels_ts, axis=0)
            self.indexes_holder=np.concatenate(indexes_holder, axis=0)
            #=mlx_data.reshape([mlx_data.shape[0],mlx_data.shape[1], 24, 32])
        else:
            if not self.in_lab_test_id is None:
                print(f'Using {self.in_lab_list[self.in_lab_test_id]} in Testing')
                labels, data, indexes, _, _=balanceData(FeatureCalc(get_data(self.in_lab_list[self.in_lab_test_id], in_lab_dir)), training=False)
                mlx_data.append(data)
                labels_ts.append(labels)
                indexes_holder.append(indexes)
            if not self.in_wild_test_id is None:
                print(f'Using {self.in_wild_list[self.in_wild_test_id]} in Testing')
                labels, data, indexes, _, _=balanceData(FeatureCalc(get_wild_data(self.in_wild_list[self.in_wild_test_id], in_wild_dir)), training=False)
                mlx_data.append(data)
                labels_ts.append(labels)
                indexes_holder.append(indexes)
            if len(mlx_data)>0:

                self.mlx_data=np.concatenate(mlx_data, axis=0)
                self.labels_ts=np.concatenate(labels_ts, axis=0)
                self.indexes_holder=np.concatenate(indexes_holder, axis=0)
            else:

                self.labels_ts=None
                self.mlx_data=None
                self.indexes_holder=None
        

    def __len__(self):
        if self.mlx_data is not None:
            return len(self.mlx_data)
        else:
            return 0
    
    def __getitem__(self, idx):
        """
        data returned in the format of (data, label)
        """
        if torch.is_tensor(idx):
            idx=idx.tolist()
        
        data=self.mlx_data[idx]
        label=self.labels_ts[idx]
        index=self.indexes_holder[idx]
        if self.transform:
            data1=data[ :494].reshape(19,26)
            data2=data[ 494:]
            data1=self.transform(data1).reshape(-1)
            data=torch.cat((data1, torch.from_numpy(data2)), dim=0)
        return (data, label, index)
