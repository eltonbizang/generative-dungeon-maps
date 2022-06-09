from abc import ABC
from collections import deque

import numpy as np


class V(ABC):
    def __init__(self, num_states: int, num_actions: int, num_state_vars: int = None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_state_vars = num_state_vars

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class Q(V, ABC):
    def __init__(self, num_states: int, num_actions: int, alpha: float):
        super().__init__(num_states, num_actions)
        self.alpha = alpha
        self.current_state = 0
        self.current_action = 0
        self.t = 0

    def update(self, target):
        pass


class Policy:

    def __init__(self, epsilon, state_action_values, exploit_method=np.argmax):
        self.epsilon = epsilon
        self.exploit_method = exploit_method
        self.P = state_action_values

    def __call__(self, state):
        if np.random.random() < self.epsilon:
            return self.explore(state)
        else:
            return self.exploit(state)

    def explore(self, state):
        values = self.P[state]
        return np.random.randint(0, len(values))

    def exploit(self, state):
        values = self.P[state]
        return self.exploit_method(values)


class Trajectory(deque):
    pass


class Gain(float):
    def __init__(self, discount_rate: float = 0.99):
        self.discount_rate = discount_rate
        self._value = 0
        self.t = 0

    def __call__(self):
        return self._value

    def update(self, reward: float):
        self.t += 1
        self._value += self.discount_rate**self.t * reward
