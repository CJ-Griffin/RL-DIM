# RL

```commandline
% pip install gym - NOPE! - do from source

git submodule add https://github.com/openai/gym
cd gym
pip install -e .
cd ..
pip install pyglet
conda install -c pytorch pytorch torchvision
pip install pygame
pip install tensorboard
```

For now, I'm assuming fully observable, i.e. that `observation = state`

```commandline
git clone https://github.com/CJ-Griffin/RL.git
sudo apt-get upgrade
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
sudo apt-get install curl
```

got to https://www.anaconda.com/products/distribution#linux to find the correct install, copy the URL and run:
```commandline
curl https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh --output conda.sh
bash conda.sh
```
Restart terminal

```commandline
conda update -n base conda
conda update anaconda
conda install pytorch torchvision torchaudio -c pytorch
conda install -c conda-forge neptune-client emoji tqdm matplotlib array2gif
pip install xxhash enquiries torchviz
```
## Running experiments
To run experiments, use the following command structure:
```commandline
python experiment_manager.py -e [experiment set] [additional arguments]
```
Where `[experiment set]` is the name of the experiment you wish to run. Accepted experiment names correspond to keys in `SKEIN_DICT`, which is defined in `skein_definitions.py` and in the files in the 'more_skein_definitions' folder.

### Additional arguments
* -p: When used, if `is_parallel` is set to True in the `TrainParams` for the experiment, enables parallel execution of experiments.
* -t: Sets the number of episodes to 1000 and sets `is_test` to True for a quicker trial run of the experiment.

### Example
The following trains a deep Q-learning agent in the Repulsion environment, for 100000 episodes, with discount factor of 0.95:
```commandline
python experimentmanager.py DQN-cont
```
## Code structure
`experiment_manager.py` defines code for managing experiments, including rendering, recording, and returning scores. The folders `agents` and `custom_envs` contain files defining agents and environments, respectively, used in experiments. Experiments are stored in `SKEIN_DICT`, in `skein_definitions.py` and in the `more_skein_definitions` folder.

For more details on the project structure, see `tree.py`. 

## Code tree for experiment_manager.py

[![](https://mermaid.ink/img/pako:eNplkctuwjAQRX_F8ipIkA_IolJLoIWya3cEIcseh1HjceSHWgT8eyeQCqR6Zd0z546jnKT2BmQlbee_9UGFJD7rhgSf5y389BDQAaW9U6RaCGV_vEG0wimkstyJ2exJvBTzzWoyekMyL-IXDLz8SzmstyHT_gp2t7Tm9Gx9EJ1vUZ8X14H72nFqwVPLG-oxeoP6AbwWAchAENE7EDFlayd3-lYQ9CkTDBtapHZkS2YrNrUP5r850HVxXwgP-TtbKQcSIxKROyAKRUYg8bc4ldDTaKzZ2BQdKBaU6H2H-jgVY8ODGZGbwFrQiV_CDTGhjtcSOZUOuBUN_6bTkDQyHcBBIyu-GrAqd6mRDV14VOXkP46kZZVChqnMvVEJalRtUE5WVnURLr8vmqOk?type=png)](https://mermaid.live/edit#pako:eNplkctuwjAQRX_F8ipIkA_IolJLoIWya3cEIcseh1HjceSHWgT8eyeQCqR6Zd0z546jnKT2BmQlbee_9UGFJD7rhgSf5y389BDQAaW9U6RaCGV_vEG0wimkstyJ2exJvBTzzWoyekMyL-IXDLz8SzmstyHT_gp2t7Tm9Gx9EJ1vUZ8X14H72nFqwVPLG-oxeoP6AbwWAchAENE7EDFlayd3-lYQ9CkTDBtapHZkS2YrNrUP5r850HVxXwgP-TtbKQcSIxKROyAKRUYg8bc4ldDTaKzZ2BQdKBaU6H2H-jgVY8ODGZGbwFrQiV_CDTGhjtcSOZUOuBUN_6bTkDQyHcBBIyu-GrAqd6mRDV14VOXkP46kZZVChqnMvVEJalRtUE5WVnURLr8vmqOk)
