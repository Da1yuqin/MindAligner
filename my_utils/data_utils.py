import json
import os
import random
from itertools import cycle

import braceexpand
import h5py
import torch
import webdataset as wds
from torch.utils.data import DataLoader, Dataset


def load_nsd_images(data_path):
    """
    Args:
        data_path (str): The path to the directory containing the HDF5 file.
    """
    file = h5py.File(os.path.join(data_path, 'coco_images_224_float16.hdf5'), 'r')
    return file['images']


def get_train_dataloaders(subj_ids=[1, 2], num_sessions=1, return_num_train=True, 
                          data_path=".../your_data_path"):
    train_dls = []
    num_train = []
    voxel_dataset = {}

    for subj_id in subj_ids:
        train_url = f"{data_path}/wds/subj0{subj_id}/train/{{0..{num_sessions-1}}}.tar"
        train_url = list(braceexpand.braceexpand(train_url))

        train_data = wds.WebDataset(train_url, resampled=False, nodesplitter=lambda x: x) \
            .decode("torch") \
            .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy") \
            .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
        
        num_data = sum(1 for _ in train_data)
        train_dl = DataLoader(train_data, batch_size=num_data, shuffle=False, drop_last=False, pin_memory=True)
        train_dls.append(train_dl)
        num_train.append(num_data)

        file = h5py.File(f'{data_path}/betas_all_subj0{subj_id}_fp32_renorm.hdf5', 'r')
        betas = file['betas'][:]
        betas = torch.tensor(betas).to("cpu").to(torch.float16)
        voxel_dataset[f'subj0{subj_id}'] = betas

    if return_num_train:
        return train_dls, voxel_dataset, num_train
    return train_dls, voxel_dataset


def get_test_dataloader(data_path, subj, new_test):
    def get_num_test_samples(subj, new_test):
        if not new_test:  # using old test set from before full dataset released (used in original MindEye paper)
            return {
                3: 2113,
                4: 1985,
                6: 2113,
                8: 1985
            }.get(subj, 2770)
        else:  # using larger test set from after full dataset released
            return {
                3: 2371,
                4: 2188,
                6: 2371,
                8: 2188
            }.get(subj, 3000)
    
    num_test = get_num_test_samples(subj, new_test)
    
    base_path = f"{data_path}/wds/subj0{subj}"
    test_type = "new_test" if new_test else "test"
    test_url = f"{base_path}/{test_type}/0.tar"
    print(f"test_url: {test_url}")

    test_data = wds.WebDataset(test_url, resampled=False, nodesplitter=lambda x: x) \
        .decode("torch") \
        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy") \
        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])

    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
    print(f"loaded test dataloader for subj{subj}!\n")
    return test_dl, num_test


# create paired indices based on categories from two JSON files
class PairedIndexDataset(Dataset):
    def __init__(self, json_file1, json_file2, seed=None):
        super(PairedIndexDataset, self).__init__()
        
        # set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
        
        with open(json_file1, 'r') as f:
            self.data1 = json.load(f)
        with open(json_file2, 'r') as f:
            self.data2 = json.load(f)
        
        self.categories = list(set(self.data1.keys()).intersection(self.data2.keys()))
        self.pairs = []

        for category in self.categories:
            indices1 = self.data1.get(category, [])
            indices2 = self.data2.get(category, [])

            if not indices1 or not indices2:
                continue
            
            for img_id1 in indices1:
                for img_id2 in indices2:
                    self.pairs.append({
                        'category': category,
                        'img_id1': img_id1,
                        'img_id2': img_id2,
                    })

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]


def custom_collate(batch):
    return {
        'categories': [item['category'] for item in batch],
        'img_ids1': [item['img_id1'] for item in batch],
        'img_ids2': [item['img_id2'] for item in batch],
    }


# example usage
if __name__ == "__main__":
    json_path = "../sim_dataset/v2subj1257"
    json_file1 = f"{json_path}/category_image_idx_subj1.json"
    json_file2 = f"{json_path}/category_image_idx_subj2.json"

    dataset = PairedIndexDataset(json_file1, json_file2, seed=42)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    infinite_dataloader = cycle(dataloader)

    try:
        for i, batch in enumerate(infinite_dataloader):
            print(f"batch {i + 1}:")
            for j in range(len(batch['categories'])):
                print(f"  category: {batch['categories'][j]}, img_id1: {batch['img_ids1'][j]}, img_id2: {batch['img_ids2'][j]}")
            if i > 10:
                break
    except KeyboardInterrupt:
        print("Data loading interrupted.")
