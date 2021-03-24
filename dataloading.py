import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, dataset_dir ='../../../workdir/pellegrainv/3Tanks/capteurs_labels', augment = False, noise_ratio = 0.05, train = True, clean = True, trainset = None, device = 'cpu'):
        self.train = train
        self.clean = clean
        self.trainset = trainset
        self.device = device
        if not train:
            dataset_dir = dataset_dir+'_test'
        self.data = pd.read_csv(dataset_dir+'.csv')
        if clean:
            self.data = self.data[self.data["LABEL"] == 0].drop('TIME', axis = 1)
        else:
            self.data = self.data.drop('TIME', axis = 1)
        self.sensors = self.data.ID.unique()
        self.safe_sensors = [self.sensors[i] for i,x in enumerate([len(self.data[self.data["ID"] == i])>1 for i in self.sensors]) if x]
        if train and clean:
            self.safe_sensors.remove(687)
            self.safe_sensors.remove(692)
        self.nb_safe_sensors = len(self.safe_sensors)
        self.X = []
        self.y = []
        self.lengths = []
        self.augment = augment
        self.noise_ratio = noise_ratio
        self.labels = self.data[self.data["ID"] == self.safe_sensors[0]]["LABEL"].values[1:]
        for i in self.safe_sensors[1:]:
            self.labels = np.concatenate((self.labels, self.data[self.data["ID"] == i]["LABEL"].values[1:]), axis = 0)
        self.labels = torch.tensor(self.labels)
        for i in range(self.nb_safe_sensors):
            self.X.append(torch.FloatTensor(self.data[self.data["ID"] == self.safe_sensors[i]].drop(['ID', 'LABEL'], axis = 1).values[:-1]))
            self.y.append(torch.FloatTensor(self.data[self.data["ID"] == self.safe_sensors[i]].drop(['ID', 'LABEL', 'MAN_P1', 'MAN_P2', 'MAN_V1', 'MAN_V2', 'MAN_V3'], axis = 1).values[1:]))
            self.lengths.append(self.X[i].shape[0])
        self.mean = torch.cat((torch.tensor(self.data.mean(axis = 0).drop(['ID', 'LABEL', 'MAN_P1', 'MAN_P2', 'MAN_V1', 'MAN_V2', 'MAN_V3'])), torch.zeros(5)), 0)
        self.std = torch.cat((torch.tensor(self.data.std(axis = 0).drop(['ID', 'LABEL', 'MAN_P1', 'MAN_P2', 'MAN_V1', 'MAN_V2', 'MAN_V3'])), torch.ones(5)),0)
        
    def __len__(self):
        return len(self.lengths)
    def __getitem__(self,idx):
        torch.manual_seed(42)
        if self.augment:
            noiseSeq = torch.randn(self.X[idx].size()).squeeze()
            augmentedData = self.X[idx].clone()
            augmentedLabel = self.y[idx].clone()

            scaled_noiseSeq = self.noise_ratio * self.std.expand_as(self.X[idx]) * noiseSeq
            retX = self.X[idx] + scaled_noiseSeq
        else:
            retX = self.X[idx]
        if self.train and self.clean:
            return (((retX - self.mean.expand_as(retX)) / self.std.expand_as(retX)).float().to(self.device), ((self.y[idx] - self.mean.expand_as(retX)[:,0:4]) / self.std.expand_as(retX)[:,0:4]).float().to(self.device), (torch.tensor(np.array(self.safe_sensors[idx]))).float().to(self.device)) 
        return (((retX - self.trainset.mean.expand_as(retX)) / self.trainset.std.expand_as(retX)).float().to(self.device), ((self.y[idx] - self.trainset.mean.expand_as(retX)[:,0:4]) / self.trainset.std.expand_as(retX)[:,0:4]).float().to(self.device), torch.tensor(np.array(self.safe_sensors[idx])).float().to(self.device)) 