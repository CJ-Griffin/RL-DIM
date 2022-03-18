from agents.QsaLearners.qsalearner import QsaLearner
from agents.networks import DNN, CNN
from running.run_parameters import TrainParams
from utils import *


class DQN(QsaLearner):
    REQUIRES_FINITE_STATE_SPACE = False
    NETWORK_CLASS = DNN

    def __init__(self, action_space: gym.Space, state_space: gym.Space, params: TrainParams):
        super().__init__(action_space, state_space, params)
        self._alpha = params.alpha
        self._batch_size = params.batch_size
        self._buffer_size = params.buffer_size

        self._Q_net = self.NETWORK_CLASS(state_space=self._state_space, action_space=self._action_space)
        self.optimiser = torch.optim.Adam(self._Q_net.parameters(), lr=params.learning_rate)
        self.loss = torch.nn.MSELoss()

    def _init_Q_s(self, state):
        pass

    def get_greedy_action(self, state):
        if self.NETWORK_CLASS == DNN:
            state = vectorise_state(state)
        elif self.NETWORK_CLASS == CNN:
            state = imageify_state(state)
        x = (torch.tensor(state)).float()
        if x.shape == torch.Size([]):
            x = x.reshape(1)
        Qs = self._Q_net(x)
        if self._should_debug:
            p = 1e-3
            if np.random.choice([True, False], p=[p, 1.0-p]):
                print(f"Q({state}) = {Qs}")
        a = Qs.argmax()
        a = int(a)
        return a

    def step(self, state: gym.core.ObsType,
             action: gym.core.ActType,
             reward: float,
             next_state,
             done: bool):
        if self.NETWORK_CLASS == DNN:
            state = vectorise_state(state)
            next_state = vectorise_state(next_state)
        elif self.NETWORK_CLASS == CNN:
            state = imageify_state(state)
            next_state = imageify_state(next_state)
        self._memory.add(state, action, reward, next_state, done)
        if len(self._memory) >= self._batch_size:
            self.update()

    def update(self):
        states, actions, rewards, next_states, dones = self._memory.sample(sample_all=True, as_torch=True)

        # I'm not detaching here - this could mess with backprop!
        Q_next_states, _ = self._Q_net(next_states).max(dim=1)
        # print(Q_next_states)
        # print(Q_next_states.shape)
        # print(dones.shape)
        Q_next_states[dones.flatten()] = 0.0
        # print(Q_next_states)
        targets = (self._gamma * Q_next_states) + rewards.flatten()
        prevs_0 = self._Q_net(states)[torch.arange(len(states)), :]
        prevs = self._Q_net(states)[torch.arange(len(states)), actions.flatten()]
        loss = self.loss(targets, prevs)
        if self._should_debug:
            self.debug_print(Q_next_states, actions, dones, prevs, prevs_0, rewards, states, targets)
        loss.backward()
        self.optimiser.step()

    def debug_print(self, Q_next_states, actions, dones, prevs, prevs_0, rewards, states, targets):
        p = 1e-1
        if np.random.choice([True, False], p=[p, 1.0 - p]):
            for i in range(len(states)):
                if states[i].shape[0] >= 3:
                    print(states[i].view(3, -1))
                else:
                    print(states[i])
                print(actions[i])
                print(rewards[i])
                print(Q_next_states[i])
                print(targets[i])
                print(prevs[i])
                print(prevs_0[i])
                print()
                if dones[i]: print("DONE\n")
                print()
            # print(Qs)
            print("State-action pairs from update")
            print([(int(a), float(tar)) for (a, tar) in zip(list(actions), list(targets))])

    def _Q_to_string(self):
        return str(self._Q_net)


class DQN_CNN(DQN):
    NETWORK_CLASS = CNN
