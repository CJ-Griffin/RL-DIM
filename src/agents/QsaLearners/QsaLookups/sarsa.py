from src.agents.QsaLearners.QsaLookups.lookup_learner import LookupLearner


class SARSA(LookupLearner):

    def should_update(self, done: bool) -> bool:
        return len(self._memory) >= self._batch_size

    def update(self):
        states, actions, rewards, next_states, dones = self._memory.sample(sample_all=True)
        # N = len(states)
        for i, (s, a, r, sn, d) in enumerate(zip(states, actions, rewards, next_states, dones)):
            if not d and i != self._batch_size - 1:
                next_qs = self._Q[sn]
                an = max(next_qs, key=next_qs.get)
                next_q = next_qs[an]
            else:
                next_q = 0.0

            td_error = r + (self._gamma * next_q) - self._Q[s][a]
            self._Q[s][a] += self._alpha * td_error
