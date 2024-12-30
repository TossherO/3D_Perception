import numpy as np
import torch
import mmengine
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):

    def __init__(self, dataset_path, dataset_name, split, class_balance=False, translation=True, rotation=True):
        self.translation = translation
        self.rotation = rotation
        data_path = dataset_path + dataset_name + '_traj_' + split + '.pkl'
        self.data_list = mmengine.load(data_path)
        if class_balance:
            self.data_list = self.balance_data()

    def collate_fn(self, data_batch):
        batch_size = len(data_batch)
        obs_len = data_batch[0]['ob'].shape[0]
        obs = []
        futures = []
        neis_ = []
        num_neis = []
        self_labels = []
        nei_labels_ = []
        refs = []
        rot_mats = []
        for data in data_batch:
            traj = np.concatenate((data['ob'], data['future']), axis=0)
            nei = data['nei']
            if self.translation:
                ref = traj[obs_len-1:obs_len]
                traj = traj - ref
                if nei.shape[0] != 0:
                    nei = nei - ref
                if self.rotation:
                    angle = np.arctan2(traj[0][1], traj[0][0])
                    rot_mat = np.array([[np.cos(angle), -np.sin(angle)], 
                                        [np.sin(angle), np.cos(angle)]])
                    traj = np.dot(traj, rot_mat)
                    if nei.shape[0] != 0:
                        nei = np.dot(nei, rot_mat)
                else:
                    rot_mat = np.array([[1, 0], [0, 1]])
            else:
                ref = np.array([0, 0])
                rot_mat = np.array([[1, 0], [0, 1]])
            
            obs.append(traj[:obs_len])
            futures.append(traj[obs_len:])
            neis_.append(nei)
            num_neis.append(nei.shape[0])
            self_labels.append(data['label'])
            nei_labels_.append(data['nei_label'])
            refs.append(ref.flatten())
            rot_mats.append(rot_mat)

        max_num_nei = max(num_neis)
        if max_num_nei == 0:
            max_num_nei = 1
        nei_masks = torch.zeros(batch_size, max_num_nei, dtype=torch.bool)
        neis = torch.zeros(batch_size, max_num_nei, obs_len, 2)
        nei_labels = torch.zeros(batch_size, max_num_nei, dtype=torch.int32) - 1
        for i in range(batch_size):
            nei_masks[i, :num_neis[i]] = True
            neis[i, :num_neis[i]] = torch.tensor(neis_[i])
            nei_labels[i, :num_neis[i]] = torch.tensor(nei_labels_[i])
        
        obs = torch.tensor(np.stack(obs, axis=0), dtype=torch.float32)
        futures = torch.tensor(np.stack(futures, axis=0), dtype=torch.float32)
        self_labels = torch.tensor(self_labels, dtype=torch.int32)
        refs = torch.tensor(np.stack(refs, axis=0), dtype=torch.float32)
        rot_mats = torch.tensor(np.stack(rot_mats, axis=0), dtype=torch.float32)
        return obs, futures, neis, nei_masks, self_labels, nei_labels, refs, rot_mats

    def balance_data(self):
        class_data_dict = {}
        for data in self.data_list:
            label = data['label']
            if label not in class_data_dict:
                class_data_dict[label] = []
            class_data_dict[label].append(data)
        max_num = max([len(class_data_dict[label]) for label in class_data_dict])
        new_data_list = []
        for label in class_data_dict:
            class_data = class_data_dict[label]
            num = len(class_data)
            if num < max_num:
                new_data_list += class_data * (max_num // num)
            new_data_list += class_data[:max_num % num]
        return new_data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]