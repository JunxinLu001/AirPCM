import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Union, List
import metpy.calc as mpcalc
from metpy.units import units


class Dataset_Mydata(Dataset):
    def __init__(self, root_path, flag='train', seq_len=24, pred_len=24,
                 freq='3h', scale=True, embed=0,
                 normalized_col: Union[str, List[int]] = 'default'):
        self.freq = freq
        if normalized_col == 'default':
            self.normalized_col = np.arange(0, 6)
        else:
            self.normalized_col = normalized_col

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window_size = seq_len + pred_len
        self.scale = scale
        self.embed = embed
        if scale:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.stations = pd.read_csv(os.path.join(self.root_path, "stations.csv"))

        self.metero_idx = ['t2m', 'sp', 'tp', 'u10', 'v10']
        self.air_idx = ['pm25', 'pm10', 'no2', 'co', 'o3', 'so2']
        self.__process_raw_data__()
        self.border = self.__read_data__()
        self.times_pred = []
        self.times_input = []

    def __process_raw_data__(self):
        train_set = []
        time_set = []
        city_list = self.stations['station'].tolist()
        for station in city_list:
            station_df = pd.read_csv(os.path.join(self.root_path, "stations", station.lower() + ".csv"))
            station_df['time'] = pd.to_datetime(station_df['datetime'])
            times = station_df['time']
            station_df = station_df.set_index('time')
            if self.freq == '3h':
                station_df = station_df.iloc[::3]
                times = times[::3]
            air = station_df[self.air_idx].values
            feature = station_df[self.metero_idx].values
            u = feature[:, -2] * units.meter / units.second  # m/s
            v = feature[:, -1] * units.meter / units.second  # m/s
            speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude  # km/h  计算风速
            direc = mpcalc.wind_direction(u, v)._magnitude  # 计算风向
            feature[:, -2] = speed  # 先速度后方向
            feature[:, -1] = direc
            train_set.append(np.concatenate([air, feature], axis=-1))
            time_set.append(times)
        train_set = np.stack(train_set, axis=0)
        train_set = np.transpose(train_set, (1, 0, 2))
        time_set = np.stack(time_set, axis=0)
        self.raw_data = train_set  # T x N x D   [11688, 184, 6]
        self.raw_time = time_set[0]

    def __read_data__(self):
        border1s = [0, int(len(self.raw_data) * 0.5), int(len(self.raw_data) * 0.75)]
        border2s = [int(len(self.raw_data) * 0.5), int(len(self.raw_data) * 0.75), len(self.raw_data)]
        self.train_border = (border1s[0], border2s[0])
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.data = self.raw_data[border1: border2]  # 得到划分的数据
        self.time = self.raw_time[border1: border2]
        if self.scale:  # 数据进行scale操作
            train_set = self.raw_data[self.train_border[0]: self.train_border[1], :, :]
            T, N, D = self.data.shape  # 5844，
            self.scaler.fit(train_set.reshape(-1, D)[:, self.normalized_col])
            self.data = self.data.reshape(-1, D)
            self.data[:, self.normalized_col] = self.scaler.transform(self.data[:, self.normalized_col])
            self.data = self.data.reshape(T, N, D)  # [11688, 184, 6]
        return border1, border2

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        x_start = idx
        x_end = x_start + self.seq_len
        y_start = idx + self.seq_len
        y_end = y_start + self.pred_len

        seq_x = self.data[x_start: x_end]
        seq_y = self.data[y_start: y_end]

        self.times_pred.append(self.time[y_start: y_end])
        self.times_input.append(self.time[x_start: x_end])
        return seq_x, seq_y

    def inverse_transform(self, data):
        assert self.scale is True
        pm25_mean = self.scaler.mean_[0]
        pm25_std = self.scaler.scale_[0]
        return (data * pm25_std) + pm25_mean
