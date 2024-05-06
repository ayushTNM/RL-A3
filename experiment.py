import numpy as np
import argparse
import os
import json

from torch import save as torch_save

from policy_search import policy_search, playout

# Function for running experiments
def experiment():
    
    # default (hyper)parameters
    n_timesteps = 1_000_001
    eval_interval = n_timesteps//100
    hp = dict(num_timesteps = n_timesteps, eval_interval = eval_interval, n = 30, pol_lr=1e-3, val_lr=1e-3, gamma=0.92, entropy_coef=0.01)
    
    # For every run set the hps that must overwrite the standards hps, also set a name for the run
    runs_kwargs = [
                # Reinforce
                dict(env_name="LunarLander-v2", bootstrap=False, baseline_substraction=False, pol_init = "xavier", val_init = "xavier"),
                dict(env_name="Pendulum-v1", bootstrap=False, baseline_substraction=False),
                dict(env_name="Pendulum-discrete", bootstrap=False, baseline_substraction=False, pol_init = "xavier", val_init = "xavier"),
                
                # Actor-Critic
                dict(env_name="LunarLander-v2", bootstrap=True, baseline_substraction=True, pol_init = "xavier", val_init = "xavier"),
                dict(env_name="LunarLander-v2", bootstrap=False, baseline_substraction=True, pol_init = "xavier", val_init = "xavier"),
                dict(env_name="LunarLander-v2", bootstrap=True, baseline_substraction=False, pol_init = "xavier", val_init = "xavier"),
                
                dict(env_name="Pendulum-v1", bootstrap=True, baseline_substraction=True),
                dict(env_name="Pendulum-v1", bootstrap=False, baseline_substraction=True),
                dict(env_name="Pendulum-v1", bootstrap=True, baseline_substraction=False),
                
                dict(env_name="Pendulum-discrete", bootstrap=True, baseline_substraction=True, pol_init = "xavier", val_init = "xavier"),
                dict(env_name="Pendulum-discrete", bootstrap=False, baseline_substraction=True, pol_init = "xavier", val_init = "xavier"),
                dict(env_name="Pendulum-discrete", bootstrap=True, baseline_substraction=False, pol_init = "xavier", val_init = "xavier")
                ]
    
    # parse args
    parser = argparse.ArgumentParser(description="Run and store experiment data.")
    parser.add_argument("--run","-r",dest="run_nr", type=int, help="Run number", required=True)
    parser.add_argument("--config-index", "-c", type=int, dest="conf_idx", metavar=f"[0-{len(runs_kwargs)}]", 
                        help=f"Index of run configuration in run_kwargs list", required=True,choices=range(len(runs_kwargs)))
    args = parser.parse_args()
    
    print(f"Default params: {hp}")
    
    rep = args.run_nr
    params = runs_kwargs[args.conf_idx]
    
    # Define the path to the JSON file
    if params['bootstrap'] == False and params['baseline_substraction'] == False:   # reinforce
        alg = 'reinforce'
        data_dir = f"data/run_{rep}/{alg}"
    else:                                               # actor-critic
        alg = 'actor-critic'
        config_name = " and ".join([key for key, value in params.items() if value == True])
        data_dir = f"data/run_{rep}/{alg}/{config_name}"
        
    # Make directories for different configs
    os.makedirs(data_dir, exist_ok=True)

    new_params = {**hp, **params}
    name = new_params.pop("env_name", None)
    
    # Run experiment
    pol_net, val_net, returns, timesteps = policy_search(name, **new_params)

    # Store data in json file
    with open(os.path.join(data_dir,f'{name}.json'), "w") as file:
        json.dump({**new_params,"results":np.column_stack([timesteps, returns]).tolist()}, file, indent=2)
        
    # Save models
    torch_save(pol_net.state_dict(), os.path.join(data_dir,f'{name}_policy_weights.pth'))
    if alg == "actor-critic":
        torch_save(val_net.state_dict(), os.path.join(data_dir,f'{name}_value_weights.pth'))
        
    # Save video of trained playout
    playout(name, pol_net, file_prefix=f"{name}_{alg}", file_folder=data_dir)


if __name__ == '__main__':
    experiment()
