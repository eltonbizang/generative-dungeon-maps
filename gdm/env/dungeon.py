from math import inf
import sys

try:
    from gdm import DungeonMaps
except ModuleNotFoundError:
    sys.path.append("C:\\Users\\elton\\Desktop\\generative-dungeon-maps")
    from gdm import DungeonMaps

from base import Env, ActionType, action_type, reward_type
from typing import Tuple

Coord = Tuple[int, int]


class Dungeon(Env):

    def __init__(self, _map: DungeonMaps = None, timeout=inf, size: Tuple[int, int] = (4, 4)):

        super().__init__()
        self._action_space = frozenset([self._left, self._right, self._top, self._down, self._collect, self._exit])
        self._observation_space = {"left": None, "right": None, "top": None, "down": None, "collected": None}
        self._restart(_map=_map, timeout=timeout)
        self._state_encoder = self._state_encoder()
        self._action_encoder = self._action_encoder()

    def _restart(self, _map: DungeonMaps = None, timeout=inf):
        if _map:
            self._map = _map
        else:
            self._map = DungeonMaps()
        self._current_location: Coord = self._map.starting_point
        self._collected: bool = False
        self._time: int = 0
        self._timeout = timeout

    def action_decoding(self, code: int):
        assert code in range(6)
        if code == 0:
            return self._exit
        elif code == 1:
            return self._left
        elif code == 2:
            return self._down
        elif code == 3:
            return self._right
        elif code == 4:
            return self._collect
        elif code == 5:
            return self._top
        else:
            raise Exception("Code Error")

    def _state_encoder(self) -> dict:
        n, m = self._map.size
        location_space = [str(i) + str(j) for i in range(n) for j in range(m)]
        state_space = []
        for location in location_space:
            for treasure_location in location_space:
                for output_location in location_space:
                    # the output location can coincide with the treasure location
                    # only if the agent has already collected the treasure
                    if output_location == treasure_location:
                        if treasure_location != location:
                            continue
                    state_space.append(location + treasure_location + output_location)
        return {i: encoded_state for i, encoded_state in enumerate(state_space)}

    def _action_encoder(self) -> dict:
        return {i: action for i, action in enumerate(self._action_space)}

    def _state_encoding(self):
        state = self.state
        state = state["agent_location"] + state["treasure_location"] + state["exit_location"]
        return "".join(map(str, state))

    def _state_decoding(self, code: str):
        assert type(code) == str and len(code) == 6
        agent_location = (int(code[0]), int(code[1]))
        treasure_location = (int(code[2]), int(code[3]))
        exit_location = (int(code[4]), int(code[5]))
        treasure_collected = agent_location == treasure_location
        return {"agent_location": agent_location, "treasure_location": treasure_location,
                "exit_location": exit_location, "treasure_collected": treasure_collected}

    @property
    def state(self) -> dict:
        state = {"agent_location": self._current_location, "treasure_location": None,
                 "exit_location": self._map.ending_point, "treasure_collected": self._collected}
        state["treasure_location"] = self._map.treasure_point if not self._collected else state["agent_location"]
        return state

    @property
    def observation(self):
        observation = self._map.get_walls_around(self._current_location)
        observation.update(treasure=self._collected)
        return observation

    @property
    def timeout(self) -> bool:
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

    def step(self, action: ActionType):
        assert action in self._action_space
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
        assert action in self._action_space
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
        if self.timeout:
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
        location_to_index = lambda i, j: ((4 * m + 2) * (2 * (i + 1) - 1)) + 4 * (j + 1) - 1
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

            action = self.action_decoding(int(input("Play: ")))
            print("action= ", action)
            os.system('cls')
            new_state, reward, done, info = self.step(action)
            print(self, end="\n")
            print("previous_state= ", previous_state)
            print("new_state= ", new_state)
            print("reward= ", reward)
            print("done= ", done)


if __name__ == '__main__':
    dungeon = Dungeon()
    dungeon.manual_play()
