import numpy as np
# import time
import os
import json

from torch import save as torch_save

from policy_search import policy_search, playout
# def average_over_repetitions(n_repetitions, n_timesteps, learning_rate, gamma, action_selection_kwargs, replay_buffer_size=1000, target_net_delay=100, eval_interval=500):#TODO remove args

#     returns_over_repetitions = []
#     now = time.time()
#     # progress_bar_desc = f"{f'{epsilon=}' if policy ==
#     #                        'egreedy' else f'{temp=}'}|lr={learning_rate}"
#     for rep in range(n_repetitions):  # Loop over repetitions
#         print(f"Repetition {rep+1}/{n_repetitions}:")
#         returns, timesteps = agent(n_timesteps, learning_rate, gamma, action_selection_kwargs,
#                                  replay_buffer_size=replay_buffer_size, target_net_delay=target_net_delay, eval_interval=eval_interval)

#         returns_over_repetitions.append(returns)
        
#     print('Running one setting takes {} minutes'.format((time.time()-now)/60))
#     # average over repetitions
#     learning_curve = np.mean(np.array(returns_over_repetitions), axis=0)
#     return learning_curve, timesteps


def experiment(experiment_comment=None, overwrite=False):
    # Settings
    # Experiment
    n_repetitions = 5
    
    n_timesteps = 1_01
    eval_interval = n_timesteps//100
    
    hp = dict(num_timesteps = n_timesteps, eval_interval = eval_interval, n = 30, pol_lr=1e-3, val_lr=1e-3, gamma=0.92, entropy_coef=0.01)
    
    # For every run set the hps that must overwrite the standards hps, also set a name for the run
    runs_kwargs = [
                # Reinforce
                dict(env_name="LunarLander-v2", bootstrap=False, baseline_substraction=False),
                dict(env_name="Pendulum-v1", bootstrap=False, baseline_substraction=False),
                dict(env_name="Pendulum-discrete", bootstrap=False, baseline_substraction=False),
                
                # Actor-Critic
                dict(env_name="LunarLander-v2", bootstrap=True, baseline_substraction=True),
                dict(env_name="LunarLander-v2", bootstrap=False, baseline_substraction=True),
                dict(env_name="LunarLander-v2", bootstrap=True, baseline_substraction=False),
                
                dict(env_name="Pendulum-v1", bootstrap=True, baseline_substraction=True),
                dict(env_name="Pendulum-v1", bootstrap=False, baseline_substraction=True),
                dict(env_name="Pendulum-v1", bootstrap=True, baseline_substraction=False),
                
                dict(env_name="Pendulum-discrete", bootstrap=True, baseline_substraction=True),
                dict(env_name="Pendulum-discrete", bootstrap=False, baseline_substraction=True),
                dict(env_name="Pendulum-discrete", bootstrap=True, baseline_substraction=False)
                ]
    
    for rep in range(1, n_repetitions+1):
    
        print(f"Default params: {hp}")
        
        
        # Define the path to the JSON file
        for params in runs_kwargs:
            if params['bootstrap'] == False and params['baseline_substraction'] == False:   # reinforce
                alg = 'reinforce'
                data_dir = f"data/run_{rep}/{alg}"
            else:                                               # actor-critic
                alg = 'actor-critic'
                config_name = " and ".join([key for key, value in params.items() if value == True])
                data_dir = f"data/run_{rep}/{alg}/{config_name}"


            # Check if the file exists
            data_path = os.path.join(data_dir, "data.json")
            if os.path.exists(data_path):
                # Load the JSON file
                with open(data_path, 'r') as file:
                    data = json.load(file)
                print(f"JSON file for run {rep} loaded successfully.")
            else:
                os.makedirs(data_dir, exist_ok=True)
                data = {}
                print(f"JSON file for run {rep} does not exist. Creating new one")


            new_params = {**hp, **params}
            name = new_params.pop("env_name", None)

            print(f"Config: {name}")
            print(f"Adjusted params: {params}")
            if name in data and not overwrite:
                print(f'Configuration already found, skipping..')
                continue
            
            
            pol_net, val_net, returns, timesteps = policy_search(name, **new_params)

            data.update({name:{**new_params,"results":np.column_stack([timesteps, returns]).tolist()}})
            
            with open(data_path, "w") as file:
                json.dump(data, file, indent=2)
            torch_save(pol_net.state_dict(), os.path.join(data_dir,f'{name}_policy_weights.pth'))
            if alg == "actor-critic":
                torch_save(val_net.state_dict(), os.path.join(data_dir,f'{name}_value_weights.pth'))
            playout(name, pol_net, file_prefix=f"{name}_{alg}", file_folder=data_dir)


if __name__ == '__main__':
    experiment()