# loading dataset

import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, sampler


class ForestDataset(torch.utils.data.Dataset):

    '''Characterizes a dataset for PyTorch'''

    def __init__(self, path):
        '''Initialization'''
        # open dataset
        self.dset = h5py.File(path, 'r')
        self.ortho = self.dset['ortho']
        self.dsm = self.dset['dsm']
        self.dtm = self.dset['dtm']
        self.slope = self.dset['slope']
        self.ground_truth = self.dset['ground_truth']

        # set number of samples
        self.dataset_size = 39280

        ## TODO:
        # make means and stds load from hdf5
        self.means_tams = np.array([56.12055784563426, 62.130400134006976, 53.03228547781888, 119.50916281232037], dtype='float32')
        self.stds_tams = np.array([30.37628560708646, 30.152693706272483, 23.13718651792004, 49.301477498205074], dtype='float32')

        self.means_dsm = np.array([13.45]).astype(np.float32)
        self.stds_dsm = np.array([10.386633252098674]).astype(np.float32)

        self.means_dtm = np.array([1446.0]).astype(np.float32)
        self.stds_dtm = np.array([271.05322202384195]).astype(np.float32)

        self.means_slope = np.array([22.39]).astype(np.float32)
        self.stds_slope = np.array([11.69830556896441]).astype(np.float32)


    def __len__(self):
        '''Denotes the total number of samples'''
        return self.dataset_size


    def __getitem__(self, index):
        '''Generates one sample of data'''

        # depending on data change mean and std
        means = self.means_tams
        stds = self.stds_tams

        # Load data and get label
        X_ortho = (torch.tensor(self.ortho[index], \
            dtype=torch.float32).permute(2, 0, 1) - \
            means[:, np.newaxis, np.newaxis]) / stds[:, np.newaxis, np.newaxis]
        X_dsm = (torch.tensor(self.dsm[index], \
            dtype=torch.float32).permute(2, 0, 1) - self.means_dsm) / self.stds_dsm
        X_dtm = (torch.tensor(self.dtm[index], \
            dtype=torch.float32).permute(2, 0, 1) - self.means_dtm) / self.stds_dtm
        X_slope = (torch.tensor(self.slope[index], \
            dtype=torch.float32).permute(2, 0, 1) - self.means_slope) / self.stds_slope

        X = torch.cat((X_ortho, X_dsm, X_dtm, X_slope),0)
        y = torch.tensor(self.ground_truth[index][:,:,0], dtype=torch.torch.int64)

        return X, y #torch.from_numpy(y).permute(2, 0, 1)


    def close(self):
        ''' closes the hdf5 file'''
        self.dset.close()


    def get_sampler(self, split={'train':0.8, 'val':0.1, 'test':0.1}, shuffle_dataset=True, random_seed=0):
        # create indices
        indices = list(range(self.dataset_size))
        # shuffle dataset
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        # split validation
        split_val = int(np.floor(split['val'] * self.dataset_size))
        val_indices = indices[:split_val]

        split_test = int(np.floor(split['test'] * self.dataset_size))
        test_indices = indices[split_val:split_val+split_test]

        if sum(split.values()) != 1.0:
            split_train = int(np.floor(split['train'] * self.dataset_size))
            train_indices = indices[split_val+split_test:split_val+split_test+split_train]
        else:
            train_indices = indices[split_val+split_test:]

        train_sampler = sampler.SubsetRandomSampler(train_indices)
        valid_sampler = sampler.SubsetRandomSampler(val_indices)
        test_sampler = sampler.SubsetRandomSampler(test_indices)

        return train_sampler, valid_sampler, test_sampler


    def show_item(self, index):
        '''shows the data'''
        #plt.imshow(np.array(self.ground_truth[index]))

        fig = plt.figure(figsize=(20,20))

        dic_data = {'RGB' : [np.array(self.ortho[index][:,:,:3]), [0.1, 0.3, 0.5, 0.7]], \
        'CIR' : [np.array(np.roll(self.ortho[index], 1, axis=2)[:,:,:3]), [0.1, 0.3, 0.5, 0.7]], \
        'DSM' : [np.array(self.dsm[index].astype('f')), [10, 20, 30]], \
        'DTM' : [np.array(self.dtm[index].astype('f')), [10, 20, 30]], \
        'Slope' : [np.array(self.slope[index].astype('f')), [10, 20, 30]], \
        'Ground Truth' : [np.array(self.ground_truth[index].astype('f')), [0, 1, 2, 3, 4]]}

        for i, key in enumerate(dic_data):
            ax = fig.add_subplot(2, 3, i+1)
            imgplot = plt.imshow(dic_data[key][0])
            ax.set_title(key)
            plt.colorbar(ticks=dic_data[key][1], orientation='horizontal')
            plt.axis('off')
