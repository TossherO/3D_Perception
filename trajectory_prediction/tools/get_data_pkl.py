import os
import importlib
import torch
import numpy as np
import pickle
from sklearn.cluster import KMeans

from dataloader import Dataloader
from utils import seed, get_rng_state

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs='+', default=[])
parser.add_argument("--test", nargs='+', default=[])
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)


# python get_data_pkl.py --train data/eth/train --test data/eth/test --config config/eth.py

if __name__ == "__main__":
    settings = parser.parse_args()
    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if settings.device is None:
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
    settings.device = torch.device(settings.device)
    
    seed(settings.seed)
    init_rng_state = get_rng_state(settings.device)
    rng_state = init_rng_state

    ###############################################################################
    #####                                                                    ######
    ##### prepare datasets                                                   ######
    #####                                                                    ######
    ###############################################################################
    kwargs = dict(
            batch_first=False, frameskip=settings.frameskip,
            ob_horizon=config.OB_HORIZON, pred_horizon=config.PRED_HORIZON,
            device=settings.device, seed=settings.seed)
    train_data, test_data = None, None
    if settings.test:
        print(settings.test)
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.test))]
        else:
            inclusive = None
        test_dataset = Dataloader(
            settings.test, **kwargs, inclusive_groups=inclusive,
            batch_size=config.batch_size, shuffle=False, dataset_type='test', dataset_name=str(settings.test[0]).split('/')[1]
        )


    if settings.train:
        print(settings.train)
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.train))]
        else:
            inclusive = None
        train_dataset = Dataloader(
            settings.train, **kwargs, inclusive_groups=inclusive, 
            flip=True, rotate=True, scale=True,
            batch_size=config.batch_size, shuffle=True, batches_per_epoch=config.EPOCH_BATCHES, dataset_type='train',
            dataset_name=str(settings.train[0]).split('/')[1]
        )

        # create motion modes
        print('Creating motion modes...')
        train_info_path = './data/' + str(settings.train[0]).split('/')[1] + '_train.pkl'
        f = open(train_info_path, 'rb')
        train_info = pickle.load(f)
        f.close()

        obs_len = config.OB_HORIZON
        pred_len = config.PRED_HORIZON
        results = []
        for data in train_info:
            traj = np.concatenate([np.array(data[0]), np.array(data[1])], axis=0)
            traj = traj - traj[obs_len - 1]
            ref = traj[0]
            angle = np.arctan2(ref[1], ref[0])
            rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
            traj = np.dot(traj, rot_mat.T)
            results.append(traj)
        results = np.array(results)
        cluster_data = results[:, obs_len:].reshape(results.shape[0], -1)
        clf = KMeans(n_clusters=config.n_clusters, random_state=1).fit(cluster_data)
        motion_modes = clf.cluster_centers_.reshape(config.n_clusters, -1, 2)

        if not os.path.exists('./data'):
            os.makedirs('./data')
        save_path_file = './data/' + str(settings.train[0]).split('/')[1] + '_motion_modes.pkl'
        f = open(save_path_file, 'wb')
        pickle.dump(motion_modes, f)
        f.close()
        print('Finish creating motion modes')