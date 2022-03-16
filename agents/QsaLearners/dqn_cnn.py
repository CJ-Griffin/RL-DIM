from agents.QsaLearners.qsalearner import QsaLearner
from agents.networks import CNN
from utils import *


class DQN_CNN(QsaLearner):
    REQUIRES_FINITE_STATE_SPACE = False

    def __init__(self,
                 action_space: gym.Space,
                 state_space: gym.Space,
                 alpha: float = 0.1,
                 epsilon: float = 0.05,
                 buffer_size: int = 1000,
                 batch_size: int = 100,
                 update_freq: int = 100,
                 gamma: float = 0.99,
                 # is_state_one_hot: bool = False,
                 # is_action_one_hot: bool = False,
                 learning_rate: float = 1e-4,
                 ):
        super().__init__(action_space, state_space, epsilon, buffer_size, batch_size, update_freq, gamma)
        self._alpha = alpha
        self._batch_size = batch_size
        self._buffer_size = buffer_size

        self._Q_net = CNN(state_space=state_space, action_space=action_space)
        self.optimiser = torch.optim.Adam(self._Q_net.parameters(), lr=learning_rate)
        self.loss = torch.nn.MSELoss()

    def _init_Q_s(self, state):
        pass

    def get_greedy_action(self, state):
        x = imageify_state(state)  # .unsqueeze(0)
        Qs = self._Q_net(x)
        if self._debug_mode:
            p = 1e-3
            if np.random.choice([True, False], p=[p, 1.0-p]):
                print(Qs)
        a = Qs.argmax()
        a = int(a)
        return a

    def step(self, state: gym.core.ObsType,
             action: gym.core.ActType,
             reward: float,
             next_state,
             done: bool):
        state = imageify_state(state)
        next_state = imageify_state(next_state)
        self._memory.add(state, action, reward, next_state, done)
        if len(self._memory) >= self._batch_size:
            self.update()

    def update(self):
        states, actions, rewards, next_states, dones = self._memory.sample(sample_all=True, as_torch=True)
        Q_next_stars, _ = self._Q_net(next_states).max(dim=1)
        targets = (self._gamma * Q_next_stars) + rewards.flatten()
        prevs = self._Q_net(states)[torch.arange(len(states)), actions.flatten()]
        loss = self.loss(targets, prevs)
        loss.backward()
        self.optimiser.step()

    def _Q_to_string(self):
        return str(self._Q_net)
