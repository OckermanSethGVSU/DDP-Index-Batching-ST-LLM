import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
import torch
import pickle
from torch.utils.data import Dataset 


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
    

class IndexDataset(Dataset):
    def __init__(self,x, data, lazy=False):
         self.x = x 
         self.data = data 
         self.lazy = lazy 
        
        
    def __len__(self):
        # Return the number of samples
        return self.x.shape[0]

    def __getitem__(self, idx):
        idx = self.x[idx]
        if self.lazy:
            y_start = idx + 12
            y2 = y_start + 12
            x=  torch.from_numpy(self.data[idx:y_start,...].compute())
            y = torch.from_numpy(self.data[y_start:y2,...].compute())
            return x,y


        y_start = idx + 12
        return self.data[idx:y_start,...], self.data[y_start:y_start + 12,...]
    

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def index_load_dataset(h5_filename, h5_key, add_time_in_day=True, add_day_in_week=True, lazy=False):
        
        
        df = pd.read_hdf(h5_filename, h5_key)
        
        if 'bay' in h5_filename:
            pass
        else:
            df.index.freq='5min'  # Manually assign the index frequency
        
        df.index.freq = df.index.inferred_freq

        x_offsets = np.sort(
            # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
            np.concatenate((np.arange(-11, 1, 1),))
        )
        # Predict the next one hour
        y_offsets = np.sort(np.arange(1, 13, 1))

        num_samples, num_nodes = df.shape
        
        data = np.expand_dims(df.values, axis=-1)
        
       
        
        data_list = [data[:]]
    
      
        if add_time_in_day:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        
        if add_day_in_week:
            day_of_week = np.array(df.index.dayofweek)[:, np.newaxis]
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 1))
            day_in_week[...,0] = day_of_week
            data_list.append(day_in_week)

        data = np.concatenate(data_list, axis=-1)

        x, y = [], []
        # t is the index of the last observation.
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
        x_i = np.arange(x_offsets[0] + max_t)

        concat = False
        
        bin_len = round(x_i.shape[0] * 0.7) + 11
        if concat:
            ascending = np.arange(1, 12)
            descending = np.arange(11, 0, -1)
            remaining_elements = bin_len - (len(ascending) + len(descending))
            
            repeat_12 = np.full(remaining_elements, 12)
            
            # Step 3: Concatenate the sequences
            bin = np.concatenate((ascending, repeat_12, descending))

        else: 
            # Step 1: pre-alloc
            bin = np.empty(bin_len, dtype=int)
            
            # Step 2: Fill in the ascending part [1, 2, ..., 12]
            bin[:12] = np.arange(1, 13)
            # Step 3: Calculate how many times to repeat 12
            remaining_elements = bin_len - 24
            bin[12:13+remaining_elements] = 12
            
            # Step 4: Fill in the descending part [11, 10, ..., 1]
            bin[13+remaining_elements:] = np.arange(11, 0, -1)
        

        num_samples = x_i.shape[0]
        num_test = round(num_samples * 0.2)
        num_train = round(num_samples * 0.7)
        num_val = num_samples - num_test - num_train

        

        x_train = x_i[:num_train]
        x_val = x_i[num_train: num_train + num_val]
        x_test = x_i[-num_test:]

        cutoff = bin.shape[0]
        
        total_entries = x_train.shape[0] * 12 * num_nodes
    
        mean = (data[: cutoff,..., 0].sum(axis=1) * bin).sum() / total_entries
        std = np.sqrt(  ((np.square(data[: cutoff, ..., 0] - mean)).sum(axis=1) * bin ).sum() / total_entries )
        

        # data[..., 0] = (data[..., 0] - mean) / std
        
        return data, mean, std, x_train, x_val, x_test


def h5_load_dataset(h5file, batch_size, add_time_in_day=True, add_day_in_week=True):
    
    df = pd.read_hdf(h5file)
    num_samples, num_nodes = df.shape
    
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
       

      
        # Convert the day of the week to a NumPy array and reshape for broadcasting
        day_of_week = np.array(df.index.dayofweek)[:, np.newaxis]
        

        # Assign the day of the week directly to the array
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 1))

        # Assign the day of the week directly to the array
        day_in_week[...,0] = day_of_week
        data_list.append(day_in_week)

        

    data = np.concatenate(data_list, axis=-1)
    
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]
    
    
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]
    data = {}
    

    data['x_train'] = x_train
    data['y_train'] = y_train
    data['x_val'] = x_val
    data['y_val'] = y_val
    data['x_test'] = x_test
    data['y_test'] = y_test



    scaler = StandardScaler(
        mean=data["x_train"][..., 0].mean(), std=data["x_train"][..., 0].std()
    )
    

    for category in ["train", "val", "test"]:
        data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])
    print("Perform shuffle on the dataset")
    # random_train = torch.arange(int(data["x_train"].shape[0]))
    # random_train = torch.randperm(random_train.size(0))
    # data["x_train"] = data["x_train"][random_train, ...]
    # data["y_train"] = data["y_train"][random_train, ...]

    # random_val = torch.arange(int(data["x_val"].shape[0]))
    # random_val = torch.randperm(random_val.size(0))
    # data["x_val"] = data["x_val"][random_val, ...]
    # data["y_val"] = data["y_val"][random_val, ...]

    # random_test = torch.arange(int(data['x_test'].shape[0]))
    # random_test = torch.randperm(random_test.size(0))
    # data['x_test'] =  data['x_test'][random_test,...]
    # data['y_test'] =  data['y_test'][random_test,...]

    data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)
    data["val_loader"] = DataLoader(data["x_val"], data["y_val"], batch_size)
    data["test_loader"] = DataLoader(data["x_test"], data["y_test"], batch_size)
    data["scaler"] = scaler

    return data


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):

    
    data = {}
    for category in ["train", "val", "test"]:
        cat_data = np.load(os.path.join(dataset_dir, category + ".npz"))
        data["x_" + category] = cat_data["x"]
        data["y_" + category] = cat_data["y"]
    scaler = StandardScaler(
        mean=data["x_train"][..., 0].mean(), std=data["x_train"][..., 0].std()
    )
    # Data format
    # print("x_train.shape: ", data["x_train"].shape)
    # print("x_train ", np.unique(data["x_train"][...,2]))
    # print("x_train ", data["x_train"][...,2])
    for category in ["train", "val", "test"]:
        data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])
    print("Perform shuffle on the dataset")
    random_train = torch.arange(int(data["x_train"].shape[0]))
    random_train = torch.randperm(random_train.size(0))
    data["x_train"] = data["x_train"][random_train, ...]
    data["y_train"] = data["y_train"][random_train, ...]

    random_val = torch.arange(int(data["x_val"].shape[0]))
    random_val = torch.randperm(random_val.size(0))
    data["x_val"] = data["x_val"][random_val, ...]
    data["y_val"] = data["y_val"][random_val, ...]

    # random_test = torch.arange(int(data['x_test'].shape[0]))
    # random_test = torch.randperm(random_test.size(0))
    # data['x_test'] =  data['x_test'][random_test,...]
    # data['y_test'] =  data['y_test'][random_test,...]

    data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)
    data["val_loader"] = DataLoader(data["x_val"], data["y_val"], valid_batch_size)
    data["test_loader"] = DataLoader(data["x_test"], data["y_test"], test_batch_size)
    data["scaler"] = scaler

    return data

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))


def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def WMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    loss = torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))
    return loss

def metric(pred, real):
    mae = MAE_torch(pred, real, 0).item()
    mape = MAPE_torch(pred, real,0).item()
    wmape = WMAPE_torch(pred, real, 0).item()
    rmse = RMSE_torch(pred, real, 0).item()
    return mae, mape, rmse, wmape


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise e
    return pickle_data