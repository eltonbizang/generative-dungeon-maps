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
