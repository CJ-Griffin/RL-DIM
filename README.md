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
