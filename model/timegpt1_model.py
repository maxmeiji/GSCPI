import pandas as pd
from nixtla import NixtlaClient
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings("ignore")

# 
class TimeGPT1:
    def __init__(self, context_len, horizon_len, device='cpu'):

        # Initialize the TimeGPT-1 model
        # Define the api through https://docs.nixtla.io/docs/getting-started-timegpt_quickstart
        
        self.tgpt1 = NixtlaClient(api_key = '')
        self.tgpt1.validate_api_key()    
        self.pred_len = horizon_len

    def forward(self, batch):
        # The model of TimeGPT-1 only accept the input foramat of dataframe with date
        past = batch[0]
        future = batch[1]
        date = batch [2]
        batch_forecasts = []

        for i in range(past.shape[0]):
            date_instance = date[i]
            past_instance = past[i].numpy().reshape(-1, 1)

            # date transformation
            date_series = pd.Series(date_instance.numpy().flatten())
            datetime_series = pd.to_datetime(date_series, unit='s')
            string_dates = datetime_series.dt.strftime('%Y-%m-%d').values.reshape(-1, 1)

            #cpmbine to dataframe
            combined_np = np.hstack((string_dates, past_instance))  # Shape: (seq_len, features_num+1(date))
            df = pd.DataFrame(combined_np, columns=['Date', 'Value'])

            # model inference
            timegpt_fcst_df = self.tgpt1.forecast(df=df, h=self.pred_len, freq='M', time_col='Date', target_col='Value')
            batch_forecasts.append(timegpt_fcst_df['TimeGPT'].values)

        batch_forecasts = np.array(batch_forecasts).reshape(past.shape[0], 1, -1)
        return batch_forecasts
