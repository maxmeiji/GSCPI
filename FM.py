import numpy as np
import torch
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_provider.data_loader import GSCPI_Dataset

import matplotlib.pyplot as plt
import argparse
from model.tfm_model import TimesFM
from model.timegpt1_model import TimeGPT1
import utils.utils as util

def inference(test_loader, model, model_name):

    mae_losses = []
    mse_losses = []   
    
    # for output demo
    plot = 1

    for batch in tqdm(test_loader):
        past_batch = batch[0] # (batch size, channels, seq_len)
        actuals_batch = batch[1] # (batch_size, channels, pred_len)

        # For different foundation models setting, we pass through the batch dorectly
        batch_forecasts = model.forward(batch)
        
        # For univairate setting 
        actuals = actuals_batch.numpy()
        past_batch = past_batch.numpy()
        batch_forecasts = batch_forecasts[:, -1, :]
        actuals = actuals[:, -1, :]
        pasts = past_batch[:, -1, :]
        combined_actuals = np.concatenate((pasts, actuals), axis=1)
        
        # MAE Calculation
        mae_loss = np.abs(batch_forecasts - actuals).mean()
        mae_losses.append(mae_loss)

        # MSE Calculation
        mse_loss = np.square(batch_forecasts - actuals).mean()
        mse_losses.append(mse_loss)


        if plot == 1:
            # Plotting the first 5 instances in the batch
            util.plot(combined_actuals, pasts, batch_forecasts, model_name)
            plot = 0

    return np.mean(mae_losses), np.mean(mse_losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimesFM Model Training")

    # Add arguments
    parser.add_argument('--model_name', type=str, required=True, help='Model to use.')
    parser.add_argument('--root_path', type=str, required=True, help='Root path of the dataset.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file.')
    parser.add_argument('--model_id', type=str, required=True, help='Identifier for the model.')
    parser.add_argument('--data', type=str, required=True, help='Dataset class name.')
    parser.add_argument('--data_type', type=str, required=True, help='train / val / test')
    parser.add_argument('--seq_len', type=int, required=True, help='Sequence length for the input.')
    parser.add_argument('--pred_len', type=int, required=True, help='Prediction length (forecast horizon).')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training.')
    parser.add_argument('--device', type=str, required=True, help='cpu/gpu of timesfm')
    args = parser.parse_args()
    
    # Load data 
    if args.data == "GSCPI_Dataset":
        test_dataset = GSCPI_Dataset(data_split=args.data_type, random_seed=13, seq_len = args.seq_len, forecast_horizon=args.pred_len)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        NotImplementedError(f"Dataset {args.data} is not implemented. Please define the custom dataloader in ./data_provider")
    print(f'Shape of test dataloader: {len(test_loader)}')
    
    # Define model
    if args.model_name =='TimesFM':
        model = TimesFM(args.seq_len, args.pred_len, args.device)
    elif args.model_name == 'TimeGPT1':
        model = TimeGPT1(args.seq_len, args.pred_len, args.device)
    else:
        raise NotImplementedError("The  method is not implemented yet.")
    
    # Inference
    mae, mse = inference(test_loader, model, args.model_name)

    # Result
    print(f'Model use: {args.model_name}')
    print(f'Dataset: {args.data}_{args.data_type}')
    print(f'Inference result')
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
