import numpy as np
import torch
import torch.cuda.amp
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import timesfm


class TimesFM:
    def __init__(self, context_len, horizon_len, device='cpu'):
        # Initialize the TimesFm model
        self.tfm = timesfm.TimesFm(
            context_len=context_len,
            horizon_len=horizon_len,
            input_patch_len=32,  # Keeping these fixed as per your example
            output_patch_len=128, # Keeping these fixed as per your example
            num_layers=20,
            model_dims=1280,
            backend=device
        )
        self.tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

    def forward(self, batch):
        past_batch = batch[0]
        batch_forecasts = []

        for past in past_batch:
            # Forecast using the TimesFm model
            _, forecasts = self.tfm.forecast(list(past), [1] * past.shape[0])  # [1]: monthly frequency
            forecasts = forecasts[:, :, 5]  # (channels, pred_len)
            batch_forecasts.append(forecasts)  # (batch size, channels, pred_len)

        batch_forecasts = np.array(batch_forecasts)
        return batch_forecasts
    
