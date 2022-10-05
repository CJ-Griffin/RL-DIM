import numpy as np
from src.agents.action_value_learners.tabular_AV_learners.tabular_AV import TabularAV


class MonteCarlo(TabularAV):
    REQUIRES_FINITE_STATE_SPACE = True

    def should_update(self, done: bool) -> bool:
        return done

    # To be called only when self._memory has > self._batch_size items
    def update(self):
        states, actions, rewards, next_states, dones = self._memory.sample(sample_all=True,
                                                                           as_torch=False)
        assert int(dones[-1]) == 1, dones
        assert not np.any(dones[:-2]), dones
        T = len(rewards)
        returns = np.zeros(T)
        returns[-1] = rewards[-1]
        # sets rewards[-2], rewards[-3], ...., rewards[1], rewards[0]
        for t in range(T - 2, -1, -1):
            returns[t] = rewards[t] + (self._gamma * returns[t + 1])
        # for a,b,c in zip(rewards, returns,actions):
        #     print("|", a,b,c)
        assert len(returns) == len(rewards), (returns, rewards)
        # First visit
        seen = []

        for t in range(T):
            state = self.get_hashable_state(states[t])
            action = (actions[t])
            if True:  # (state, action) not in seen
                seen.append((state, action))
                ret = (returns[t])
                # print(f"{state:1.0f}, {action:1.0f}, " + f"{ret:2.2f}".zfill(5))
                n = self._Q_count[state][action]
                self._Q_count[state][action] += 1
                prev = float(self._Q[state][action])
                self._Q[state][action] += (ret - prev) / (n + 1)
