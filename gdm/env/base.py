from typing import *

ActionType = NewType("Action", Callable)
RewardType = NewType("Reward", float)


def action_type(function):
    f: ActionType = function
    return f


def reward_type(function):
    f: RewardType = function
    return f


class Env:
    def __init__(self):
        self._action_space: Set[ActionType]
        self._reward_space: Set[RewardType]
        self._state_space = None
        self._observation_space = None

    def step(self, action: ActionType):
        """

        :return:
        """
        raise NotImplementedError

    def restart(self):
        """

        :return:
        """
        raise NotImplementedError

    def close(self):
        """

        :return:
        """
        raise NotImplementedError

    def save(self):
        """

        :return:
        """
        raise NotImplementedError

    def render(self):
        """

        :return:
        """
        raise NotImplementedError


class Agent:
    def __init__(self, values, policy):
        self.values = values
        self.observation = None
        self._policy = policy
        self._training = False
        self._evaluation = None

    def save(self):
        """

        :return:
        """
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError



