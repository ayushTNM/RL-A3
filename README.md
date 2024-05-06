# RL-A3

## Running a Single Experiment

### Prerequisites
- Install required libraries (`requirements_3.12.2.txt`)

### Running the Experiment
Run the experiment script by executing the following command:
```
python experiment.py -r <run_nr> -c <conf_idx>
```
- run_nr &rarr; index of run to save to (e.g., -r 0 &rarr; data/run_0)
- conf_idx &rarr; index of configuration to run from run_kwargs array in experiment.py (e.g., -c 0 &rarr; [1st element in run_kwargs](experiment.py#L21))