import numpy as np
import os
import json
import pandas as pd

from Helper import smooth
from Helper import LearningCurvePlot


# func to plot
def plot_hyperparameter(data, title, filename, param_names=None, param_keys=None):
    plot = LearningCurvePlot(title)
    for _, row in data.iterrows():
        
        result = row['mean_results'].T
        x = result[0]
        y = result[1]
            
        plot.add_curve(x, y, f"{row["algorithm"]} {row["label"]}")
    plot.save(filename)
    
# code to plot experiments
def plot_experiments(data_folder = 'data',smoothing_window=None, save_dir = ''):
    data = load_data(data_folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
        
    print(data)
    
    def results_to_means(x):
        return x.apply(lambda x: np.column_stack([np.array(x)[:, 0], np.mean(np.array(x)[:, 1:], axis=1)])).mean()
        
    def get_label(x):
        params = np.array(["boootstrapping", "baseline substraction"])[np.array([x["bootstrap"], x["baseline_substraction"]])]
        if len(params) > 0:
            return f"({' + '.join(params)})"
        else:
            return ""
    
    data = data[data['num_timesteps'] != 101]   # TODO: remove
    # result_means = data.groupby("name").apply(
    #     lambda group: group.groupby(["algorithm", "bootstrap", "baseline_substraction"])['results'].agg(results_to_means)
    # )
    for name, group in data.groupby("name"):
        config_df = group.groupby(["algorithm", "bootstrap", "baseline_substraction"])['results'].agg(results_to_means).rename("mean_results").reset_index()
        
        if smoothing_window is not None:
                # additional smoothing
                config_df["mean_results"] = config_df['mean_results'].apply(lambda x: np.column_stack([np.array(x)[:,0],smooth(np.array(x)[:,1], smoothing_window, poly=1)]))

                
        config_df["label"] = config_df.apply(get_label, axis=1)
        print(config_df)
            
        plot_hyperparameter(config_df, name, f"{os.path.join(save_dir,name)}.pdf")

def load_data(data_dir):
    df_data = []
    for root, dirs, files in os.walk(data_dir):
        dir_list = root.split(os.sep)
        if files:
            config_dict = dict(run=int(dir_list[1][-1]), algorithm=dir_list[2])
            config_series = pd.Series(config_dict)
            for file in files:
                if file.endswith(".json") and "meta" not in file:
                    json_data = dict()
                    with open(os.path.join(root,file), 'r') as f:
                        json_data = json.load(f)
                        json_data.update(dict(name=os.path.splitext(file)[0]))
                            
                    data_series = pd.concat([config_series, pd.Series(json_data)])
                    df_data.append(data_series)
    return pd.DataFrame(df_data).sort_values(by=['run'])

if __name__ == '__main__':
    plot_experiments(data_folder = 'data', smoothing_window=9, save_dir="results")