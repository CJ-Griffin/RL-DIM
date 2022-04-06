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
