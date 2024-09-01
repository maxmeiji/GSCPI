from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

class GSCPI_Dataset:
    def __init__(
        self,
        seq_len: Optional[int] = 32,
        forecast_horizon: Optional[int] = 6,
        data_split: str = "train",
        data_stride_len: int = 1,
        random_seed: int = 42,
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Length of the prediction sequence.
        data_split : str
            Split of the dataset, 'train' or 'test'.
        data_stride_len : int
            Stride length when generating consecutive
            time series windows.
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', or  'imputation'.
        random_seed : int
            Random seed for reproducibility.
        """

        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = "./data/gscpi_data.csv"
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.random_seed = random_seed
        self.length_timeseries_original = 0
        # Read data
        self._read_data()

    def _get_borders(self):
        total_length = self.length_timeseries_original
        n_train = int(total_length*0.6)
        n_val = int(total_length*0.2)
        n_test = int(total_length*0.2)
        train_end = n_train
        val_end = n_train + n_val
        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        train = slice(0, train_end)
        valid = slice(train_end, val_end)
        test = slice(test_start, test_end)

        return train, valid, test

    def _read_data(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(self.full_file_path_and_name)
        self.length_timeseries_original = df.shape[0] - df.shape[0] % 10

        # transfer date and drop
        df_date = pd.to_datetime(df['Date']).astype(np.int64) / 1e9
        df.drop(columns=["Date"], inplace=True)

        # data interporlation
        df = df.infer_objects(copy=False).interpolate(method="cubic")
        data_splits = self._get_borders()
        train_data = df[data_splits[0]]
        self.scaler.fit(train_data.values)
        df = self.scaler.transform(df.values)

        # add date information back
        df = np.column_stack((df_date, df))

        if self.data_split == "train":
            self.data = df[data_splits[0], :]
        elif self.data_split == "val":
            self.data = df[data_splits[1], :]
        else:
            self.data = df[data_splits[2], :]

        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        pred_end = seq_end + self.forecast_horizon

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = seq_end - self.forecast_horizon
            seq_start = seq_end - self.seq_len

        timeseries = self.data[seq_start:seq_end, 1:].T
        forecast = self.data[seq_end:pred_end, 1:].T
        date = self.data[seq_start:seq_end, 0].T 

        return timeseries, forecast, date


    def __len__(self):
        return (
            self.length_timeseries - self.seq_len - self.forecast_horizon
        ) // self.data_stride_len + 1

