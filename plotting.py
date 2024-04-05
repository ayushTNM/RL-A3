import numpy as np
import os
import json
import re

from Helper import smooth
from Helper import LearningCurvePlot


# func to plot

def plot_hyperparameter(data, names, title, filename, param_names, param_keys):
    plot = LearningCurvePlot(title)
    for name in names:
        if name in data:
            experiment = data[name]
            result = experiment['results'].T
            x = result[0]
            y = result[1]

            if "policy" in param_names:
                label = ', '.join(f"{pn} : {list(experiment[pk].values())[0]}" for pn,pk in zip(param_names,param_keys))
                label = label.replace("egreedy","ϵ-greedy").replace("ann_", "ann. ")
            else:
                label = ', '.join(f"{pn} : {experiment[pk]}" for pn,pk in zip(param_names,param_keys))
                
            plot.add_curve(x, y, label)
        else:
            print(f"{name} not in data. Skipping..")
    plot.save(filename)

# code to plot experiments
def plot_experiments(smoothing_window=None):
    data = load_data("data.json")

    for value in data.values():
        if 'results' in value:
            value['results'] = np.array(value['results'])
            if smoothing_window is not None:
                # additional smoothing
                value['results'][:,1] = smooth(value['results'][:,1], smoothing_window, poly=1)

    configs = {"DQN Learning rate" : ["DQN_rb_tn", 'DQN_lr_0.005', 'DQN_lr_0.01'],
               "DQN exploration" : ['DQN_rb_tn', 'DQN_as_softmax', 'DQN_as_egreedy', 'DQN_as_ann_softmax'],
               "DQN Configuration": ['DQN', 'DQN_rb', 'DQN_tn', 'DQN_rb_tn']}
    
    param_names_dict = [{"learning_rate": "learning rate"},
                        {"action_selection_kwargs": "policy"}, 
                        {"target_net_delay": "target net","replay_buffer_size": "replay buffer"}]
    
    for index, (title, names) in enumerate(configs.items()):
        param_keys = list(param_names_dict[index].keys())
        param_names = list(param_names_dict[index].values())

        plot_hyperparameter(data, names, title, f"{title}.pdf", param_names, param_keys)

def load_data(data_path):
    if os.path.exists(data_path):
        # Load the JSON file
        with open(data_path, 'r') as file:
            data = json.load(file)
        print("JSON file loaded successfully.")
    else:
        data = {}
        print("JSON file does not exist.")
    print(data.keys())
    return data

if __name__ == '__main__':
    plot_experiments(smoothing_window=9)