import numpy as np
import os
import json
import pandas as pd

from Helper import smooth
from Helper import LearningCurvePlot


def plot_hyperparameter(data, title, filename, smoothing_window):
    # Initialize plot
    plot = LearningCurvePlot(title)
    
    # Plot
    for _, row in data.iterrows():
        
        # Add smoothing if required
        if smoothing_window:
            row['smoothed_returns'] = smooth(row['mean_returns'], smoothing_window, poly=1)
        
        x = row['timesteps']
        y = row['smoothed_returns']
        std = row['std_returns'] 
            
        plot.add_shaded_region(x, y - std, y + std, alpha=0.2)
        plot.add_curve(x, y, f"{row["algorithm"]} {row["label"]}")
    
    # Save
    plot.save(filename)
    
def plot_experiments(data_folder = 'data',smoothing_window=None, save_dir = ''):
    data = load_data(data_folder)
    
    # Create directory to save to if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
                
    def get_label(x):
        params = np.array(["bootstrapping", "baseline substraction"])[np.array([x["bootstrap"], x["baseline_substraction"]])]
        if len(params) > 0:
            return f"({' + '.join(params)})"
        else:
            return ""
        
    for name, group in data.groupby("name"):
        # group and compute mean of runs
        config_df = group.groupby(["algorithm", "bootstrap", "baseline_substraction"]).agg({'returns': 'mean', 'timesteps': 'first'}).reset_index()
        
        # Extract required metrics
        config_df['std_returns'] = config_df['returns'].transform(lambda x: np.std(x, axis=1))
        config_df['mean_returns'] = config_df['returns'].transform(lambda x: np.mean(x,axis=1))
        config_df["label"] = config_df.apply(get_label, axis=1)
            
        # plot and save
        plot_hyperparameter(config_df, name, f"{os.path.join(save_dir,name)}.pdf", smoothing_window = smoothing_window)
        print(f"Saved {name}.pdf")

def load_data(data_dir):
    # Initialize an empty list to store dataframe rows
    df_data = []
    
    for root, dirs, files in os.walk(data_dir):
        dir_list = root.split(os.sep)
        
        # Extract required data from files
        if files:
            config_dict = dict(run=int(dir_list[1][-1]), algorithm=dir_list[2])
            config_series = pd.Series(config_dict)
            for file in files:
                # Load data from saved json
                if file.endswith(".json") and "meta" not in file:
                    json_data = dict()
                    with open(os.path.join(root,file), 'r') as f:
                        json_data = json.load(f)
                        json_data.update(dict(name=os.path.splitext(file)[0]))
                        
                        # Convert timesteps and results to seperate columns
                        results = json_data.pop("results")
                        json_data["timesteps"] = np.array(results)[:,0]
                        json_data["returns"] = np.array(results)[:,1:]
                        
                    # Add row to list for dataframe
                    data_series = pd.concat([config_series, pd.Series(json_data)])
                    df_data.append(data_series)
                    
    # Convert and return dataframe
    return pd.DataFrame(df_data).sort_values(by=['run'])

if __name__ == '__main__':
    plot_experiments(data_folder = 'data_reinf_xav_init', smoothing_window=9, save_dir="results1")