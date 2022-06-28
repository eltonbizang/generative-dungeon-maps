from math import inf
import sys

try:
    from gdm import DungeonMaps
except ModuleNotFoundError:
    sys.path.append("C:\\Users\\elton\\Desktop\\generative-dungeon-maps")
    from gdm import DungeonMaps

from base import Env, action_type
from typing import Tuple

Coord = Tuple[int, int]


def _state_encoder(agent_location: Coord,
                   treasure_location: Coord,
                   exit_location: Coord,
                   treasure_collected: bool) -> str:
    state = agent_location + treasure_location + exit_location + (int(treasure_collected), )
    return "".join(map(str, state))


def _state_decoder(code: str):
    assert type(code) == str and len(code) == 7
    agent_location = (int(code[0]), int(code[1]))
    treasure_location = (int(code[2]), int(code[3]))
    exit_location = (int(code[4]), int(code[5]))
    treasure_collected = int(code[6])
    return {"agent_location": agent_location, "treasure_location": treasure_location,
            "exit_location": exit_location, "treasure_collected": treasure_collected, "code": code}


def _select_action(code: int):
    # This is for a manual play
    try:
        assert code in range(6)
        if code == 0:
            return "exit"
        elif code == 1:
            return "left"
        elif code == 2:
            return "down"
        elif code == 3:
            return "right"
        elif code == 4:
            return "collect"
        elif code == 5:
            return "top"
    except AssertionError:
        print("\nWrong input. You must play a number between 0 and 5."
              "\n    0: Exit"
              "\n    1: Left"
              "\n    2: Down"
              "\n    3: Right"
              "\n    4: Collect"
              "\n    5: Up")
        return -1


class Dungeon(Env):
    """

    """

    def __init__(self, _map: DungeonMaps = None, timeout=inf, size: Tuple[int, int] = (4, 4)):
        """

        :param _map:
        :param timeout:
        :param size:
        """
        super().__init__()
        if _map:
            self._map = _map
        else:
            self._map = DungeonMaps()
        self._restart(timeout=timeout)
        self.actions = {i: action for i, action in enumerate(self._actions_space)}
        self.states = {i: _state_decoder(state) for i, state in enumerate(self._states_space)}

    def _restart(self, keep_init_conditions: bool = False, timeout=inf):
        if keep_init_conditions:
            raise NotImplementedError
        else:
            self._map = DungeonMaps(size=self._map.size)
        self._current_location: Coord = self._map.starting_point
        self._collected: bool = False
        self._time: int = 0
        self._timeout = timeout

    @property
    def _actions_space(self) -> frozenset:
        return frozenset(["left", "right", "top", "down", "collect", "exit"])

    @property
    def _states_space(self) -> frozenset:
        """
        Impossible states:
            - Case when the treasure is not collected yet: The treasure location and the exit location coincide
            - Case when the treasure is already collected: The treasure location and the agent's location don't coincide
        Construction:
            ____________________________________________________________________________________________________________
        Notes:
            - The treasure can only be collected if its location coincides with that of the agent.
            - The treasure location and the agent's location may coincide without the agent having collected
                the treasure yet: The next action in this state should ideally be to collect.
        """

        n, m = self._map.size
        location_space = [(i, j) for i in range(n) for j in range(m)]
        states_space = []
        for agent_location in location_space:
            for treasure_location in location_space:
                for output_location in location_space:
                    # Treasure is not collected yet
                    if treasure_location != output_location:
                        states_space.append(_state_encoder(agent_location, treasure_location, output_location, False))
                    # Treasure is already collected
                    if treasure_location == agent_location:
                        states_space.append(_state_encoder(agent_location, treasure_location, output_location, True))

        return frozenset(states_space)

    @property
    def state(self) -> dict:
        state = {"agent_location": self._current_location, "treasure_location": None,
                 "exit_location": self._map.ending_point, "treasure_collected": self._collected}
        state["treasure_location"] = self._map.treasure_point if not self._collected else state["agent_location"]
        state.update({"code": _state_encoder(**state)})
        return state

    @property
    def observation(self) -> dict:
        observation = self._map.get_walls_around(self._current_location)
        observation.update(treasure=self._collected)
        return observation

    @property
    def is_timeout(self) -> bool:
        return self._time > self._timeout

    @action_type
    def _down(self) -> Coord:
        x, y = self._current_location
        return x + 1, y

    @action_type
    def _top(self) -> Coord:
        x, y = self._current_location
        return x - 1, y

    @action_type
    def _left(self) -> Coord:
        x, y = self._current_location
        return x, y - 1

    @action_type
    def _right(self) -> Coord:
        x, y = self._current_location
        return x, y + 1

    @action_type
    def _collect(self) -> bool:  # We'll later handle the return of the function in case the treasure already collected
        if self._collected:
            raise NotImplementedError
        else:
            return self._current_location == self._map.treasure_point

    @action_type
    def _exit(self) -> bool:
        return self._current_location == self._map.ending_point

    def step(self, action: str):
        assert action in self._actions_space
        action = self.__getattribute__('_' + action)
        info = {}
        done = False
        reward = self._reward(action)
        if action not in self._get_possible_actions():
            return self.state, reward, done, info
        elif action in {self._left, self._right, self._top, self._down}:
            self._current_location = action()
        elif action == self._collect:
            self._collected = self._collect()
        elif action == self._exit:
            if self._exit():
                done = True
                reward += self._gain_adjustment()

        return self.state, reward, done, info

    def _reward(self, action) -> int:
        self._time += 1
        possible_actions = self._get_possible_actions()
        if action not in possible_actions:
            return -10
        elif action in {self._left, self._right, self._top, self._down}:  # shift action
            next_location = action()
            if self.state.get("treasure_location") == next_location:
                return +3
            else:
                return +1
        elif action == self._collect:
            if self._collect():
                return +3
            else:
                return -5
        elif action == self._exit:
            if self._exit():
                return 3
            else:
                return -5

    def _gain_adjustment(self) -> int:
        # this must be called directly at the end of the game
        reward = 0
        if self.is_timeout:
            reward = -10
        elif self._collected:
            reward += 10

        return reward

    def _get_possible_actions(self):
        possible_actions = set()
        observation = self.observation

        if not observation["top"][-1]:
            possible_actions.add(self._top)
        if not observation["down"][-1]:
            possible_actions.add(self._down)
        if not observation["right"][-1]:
            possible_actions.add(self._right)
        if not observation["left"][-1]:
            possible_actions.add(self._left)

        # The agent cannot exit if the treasure is not yet collected
        # The agent should no longer attempt to collect something if the treasure is already collected
        if not observation["treasure"]:
            possible_actions.add(self._collect)
        else:
            possible_actions.add(self._exit)

        return possible_actions

    def __str__(self):
        n, m = self._map.size

        def location_to_index(i, j):
            return ((4 * m + 2) * (2 * (i + 1) - 1)) + 4 * (j + 1) - 1

        x, y = self._current_location
        index = location_to_index(x, y)
        map_ = ''
        for i, c in enumerate(repr(self._map)):
            if i == index:
                map_ += "#"
            elif self._collected and c == "T":
                map_ += " "
            else:
                map_ += c
        return map_

    def manual_play(self):
        import os
        done = False
        print(self)
        print("state= ", self.state)
        while not done:
            previous_state = self.state

            action = _select_action(int(input("Play: ")))
            while action == -1:
                os.system('pause')
                os.system('cls')
                print(self, end="\n")
                action = _select_action(int(input("Play: ")))
            os.system('cls')
            print("action= ", action)
            new_state, reward, done, info = self.step(action)
            print(self, end="\n")
            print("previous_state= ", previous_state)
            print("new_state= ", new_state)
            print("reward= ", reward)
            print("done= ", done)


if __name__ == '__main__':
    dungeon = Dungeon()
    dungeon.manual_play()
