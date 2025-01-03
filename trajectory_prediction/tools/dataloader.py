from typing import Optional, Sequence, List

import os, sys
import torch
import numpy as np

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

import pickle

class Dataloader(torch.utils.data.Dataset):

    class FixedNumberBatchSampler(torch.utils.data.sampler.BatchSampler):
        def __init__(self, n_batches, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_batches = n_batches
            self.sampler_iter = None #iter(self.sampler)
        def __iter__(self):
            # same with BatchSampler, but StopIteration every n batches
            counter = 0
            batch = []
            while True:
                if counter >= self.n_batches:
                    break
                if self.sampler_iter is None: 
                    self.sampler_iter = iter(self.sampler)
                try:
                    idx = next(self.sampler_iter)
                except StopIteration:
                    self.sampler_iter = None
                    if self.drop_last: batch = []
                    continue
                batch.append(idx)
                if len(batch) == self.batch_size:
                    counter += 1
                    yield batch
                    batch = []

    def __init__(self, 
        files: List[str], ob_horizon: int, pred_horizon: int,
        batch_size: int, drop_last: bool=False, shuffle: bool=False, batches_per_epoch=None, 
        frameskip: int=1, inclusive_groups: Optional[Sequence]=None,
        batch_first: bool=False, seed: Optional[int]=None,
        device: Optional[torch.device]=None,
        flip: bool=False, rotate: bool=False, scale: bool=False,
        dataset_name='sdd', dataset_type='train'
    ):
        super().__init__()
        self.ob_horizon = ob_horizon
        self.pred_horizon = pred_horizon
        self.horizon = self.ob_horizon+self.pred_horizon
        self.frameskip = int(frameskip) if frameskip and int(frameskip) > 1 else 1
        self.batch_first = batch_first
        self.flip = flip
        self.rotate = rotate
        self.scale = scale
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu") 
        else:
            self.device = device

        if inclusive_groups is None:
            inclusive_groups = [[] for _ in range(len(files))]
        assert(len(inclusive_groups) == len(files))

        print(" Scanning files...")
        files_ = []
        for path, incl_g in zip(files, inclusive_groups):
            if os.path.isdir(path):
                files_.extend([(os.path.join(root, f), incl_g) \
                    for root, _, fs in os.walk(path) \
                    for f in fs if f.endswith(".txt")])
            elif os.path.exists(path):
                files_.append((path, incl_g))
        data_files = sorted(files_, key=lambda _: _[0])

        data = []
        
        done = 0
        # too large of max_workers will cause the problem of memory usage
        max_workers = min(len(data_files), torch.get_num_threads(), 20)
        with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"), max_workers=max_workers) as p:
            futures = [p.submit(self.__class__.load, self, f, incl_g) for f, incl_g in data_files]
            for fut in as_completed(futures):
                done += 1
                sys.stdout.write("\r\033[K Loading data files...{}/{}".format(
                    done, len(data_files)
                ))
            for fut in futures:
                item = fut.result()
                if item is not None:
                    data.extend(item)
                sys.stdout.write("\r\033[K Loading data files...{}/{} ".format(
                    done, len(data_files)
                ))
        self.data = np.array(data, dtype=object)
        print("\n   {} trajectories loaded.".format(len(self.data)))
        save_file = './data/'
        if not os.path.exists(save_file):
            os.makedirs(save_file)
        save_path_file = save_file + dataset_name + '_' + dataset_type +  '.pkl'
        f = open(save_path_file, 'wb')
        pickle.dump(data, f)
        f.close()
        del data
        self.rng = np.random.RandomState()
        if seed: self.rng.seed(seed)

        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(self)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(self)
        if batches_per_epoch is None:
            self.batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
            self.batches_per_epoch = len(self.batch_sampler)
        else:
            self.batch_sampler = self.__class__.FixedNumberBatchSampler(batches_per_epoch, sampler, batch_size, drop_last)
            self.batches_per_epoch = batches_per_epoch

    def collate_fn(self, batch):
        X, Y, NEIGHBOR = [], [], []
        for item in batch:
            hist, future, neighbor = item[0], item[1], item[2]

            hist_shape = hist.shape
            neighbor_shape = neighbor.shape
            hist = np.reshape(hist, (-1, 2))
            neighbor = np.reshape(neighbor, (-1, 2))
            if self.flip:
                if self.rng.randint(2):
                    hist[..., 1] *= -1
                    future[..., 1] *= -1
                    neighbor[..., 1] *= -1
                if self.rng.randint(2):
                    hist[..., 0] *= -1
                    future[..., 0] *= -1
                    neighbor[..., 0] *= -1
            if self.rotate:
                rot = self.rng.random() * (np.pi+np.pi) 
                s, c = np.sin(rot), np.cos(rot)
                r = np.asarray([
                    [c, -s],
                    [s,  c]
                ])
                hist = (r @ np.expand_dims(hist, -1)).squeeze(-1)
                future = (r @ np.expand_dims(future, -1)).squeeze(-1)
                neighbor = (r @ np.expand_dims(neighbor, -1)).squeeze(-1)
            if self.scale:
                s = self.rng.randn()*0.05 + 1 # N(1, 0.05)
                hist = s * hist
                future = s * future
                neighbor = s * neighbor
            hist = np.reshape(hist, hist_shape)
            neighbor = np.reshape(neighbor, neighbor_shape)

            X.append(hist)
            Y.append(future)
            NEIGHBOR.append(neighbor)
        
        n_neighbors = [n.shape[1] for n in NEIGHBOR]
        max_neighbors = max(n_neighbors) 
        if max_neighbors != min(n_neighbors):
            NEIGHBOR = [
                np.pad(neighbor, ((0, 0), (0, max_neighbors-n), (0, 0)), 
                "constant", constant_values=1e9)
                for neighbor, n in zip(NEIGHBOR, n_neighbors)
            ]
        stack_dim = 0 if self.batch_first else 1
        x = np.stack(X, stack_dim)
        y = np.stack(Y, stack_dim)
        neighbor = np.stack(NEIGHBOR, stack_dim)

        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        neighbor = torch.tensor(neighbor, dtype=torch.float32, device=self.device)
        return x, y, neighbor

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def load(self, filename, inclusive_groups):
        if os.path.isdir(filename): return None
        with open(filename, "r") as record:
            data = self.load_traj(record)
        data = self.extend(data, self.frameskip)
        time = np.sort(list(data.keys()))
        valid_horizon = self.ob_horizon + self.pred_horizon

        traj = []
        tracks = {}
        neighbors = {}
        count = 0
        for tid in range(len(time)):
            t_pred = time[tid]
            t_ob = time[tid - self.pred_horizon] if tid >= self.pred_horizon else None
            
            for idx, item in data[t_pred].items():
                count += 1
                if idx not in tracks:
                    tracks[idx] = [[1e9, 1e9]] * (valid_horizon - 1)
                else:
                    tracks[idx].pop(0)
                tracks[idx].append(item[:2])
            tracks_ids = list(tracks.keys())
            for idx in tracks_ids:
                if idx not in data[t_pred]:
                    tracks.pop(idx)
            
            if t_ob is None: continue
            for idx, item in data[t_ob].items():
                if idx not in neighbors:
                    neighbors[idx] = [[1e9, 1e9]] * (self.ob_horizon - 1)
                else:
                    neighbors[idx].pop(0)
                neighbors[idx].append(item[:2])
            neighbors_ids = list(neighbors.keys())
            for idx in neighbors_ids:
                if idx not in data[t_ob]:
                    if neighbors[idx][0][0] < 1e8:
                        neighbors[idx].pop(0)
                        neighbors[idx].append([1e9, 1e9])
                    else:
                        neighbors.pop(idx)

            for idx, item in tracks.items():
                min_obs_len = 2
                if item[self.ob_horizon - min_obs_len][0] < 1e8:
                    step_range = [0, 0]
                    for i in range(1, self.ob_horizon + self.pred_horizon):
                        if item[i-1][0] < 1e8:
                            if abs(item[i][0] - item[i-1][0]) > step_range[0]:
                                step_range[0] = abs(item[i][0] - item[i-1][0])
                            if abs(item[i][1] - item[i-1][1]) > step_range[1]:
                                step_range[1] = abs(item[i][1] - item[i-1][1])
                    if step_range[0] > 2 or step_range[1] > 2:
                        continue
                    hist = np.array(item[:self.ob_horizon], dtype=np.float32)
                    for i in range(self.ob_horizon-min_obs_len, -1, -1):
                        if item[i][0] > 1e8:
                            hist[i] = hist[i+1]
                    future = np.array(item[self.ob_horizon:], dtype=np.float32)
                    neighbor = np.array([neighbors[i] for i in neighbors if i != idx], dtype=np.float32)
                    traj.append((hist, future, neighbor))

        print('\n', 'instance_num:', count, 'len_traj:', len(traj))
        return traj
                
    def extend(self, data, frameskip):
        time = np.sort(list(data.keys()))
        dts = np.unique(time[1:] - time[:-1])
        dt = dts.min()
        if np.any(dts % dt != 0):
            raise ValueError("Inconsistent frame interval:", dts)
        i = 0
        while i < len(time)-1:
            if time[i+1] - time[i] != dt:
                time = np.insert(time, i+1, time[i]+dt)
            i += 1
        # ignore those only appearing at one frame
        for tid, t in enumerate(time):
            removed = []
            if t not in data: data[t] = {}
            for idx in data[t].keys():
                t0 = time[tid-frameskip] if tid >= frameskip else None
                t1 = time[tid+frameskip] if tid+frameskip < len(time) else None
                if (t0 is None or t0 not in data or idx not in data[t0]) and \
                (t1 is None or t1 not in data or idx not in data[t1]):
                    removed.append(idx)
            for idx in removed:
                data[t].pop(idx)
        return data

    def load_traj(self, file):
        data = {}
        for row in file.readlines():
            item = row.split()
            if not item: continue
            t = int(float(item[0]))
            idx = int(float(item[1]))
            x = float(item[2])
            y = float(item[3])
            group = item[4].split("/") if len(item) > 4 else None
            if t not in data:
                data[t] = {}
            data[t][idx] = [x, y, group]
        return data


if __name__ == '__main__':

    dataset_list = ['']