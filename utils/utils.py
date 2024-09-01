import matplotlib.pyplot as plt
import numpy as np
import os

def plot(combined_actuals, pasts, batch_forecasts, model_name='tfm', max_plots=5):
    """
    Plots and saves the forecasted series versus the actual series.

    Parameters:
    - combined_actuals (np.ndarray): The actual series data.
    - pasts (np.ndarray): The past data used for forecasting.
    - batch_forecasts (np.ndarray): The forecasted series data.
    - model_name: identify the save path
    - max_plots (int): Maximum number of plots to generate and save.
    """
    save_dir = f'./pic/{model_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_plots = min(max_plots, batch_forecasts.shape[0])

    for i in range(num_plots):
        plt.figure(figsize=(10, 5))
        
        # Plot actual series
        plt.plot(combined_actuals[i], label='Actual Series', color='blue')

        # Plot forecasted series
        plt.plot(range(pasts.shape[1], pasts.shape[1] + batch_forecasts.shape[1]), 
                 batch_forecasts[i], label='Forecasted Series', color='orange')
        
        plt.title(f'TimesFM - Forecast vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        # Save the plot
        plt.savefig(f'{save_dir}/forecast_vs_actual_batch_series_{i+1}.png')
