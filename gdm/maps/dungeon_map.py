from base import *
from random import randint, random


class DungeonMaps(Maps):
    """

    """
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._starting_point = None
        obj._ending_point = None
        obj._treasure_point = None
        obj._keypoint = set()
        return obj

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_starting_point()
        self._set_ending_point()
        self._set_treasure_point()
        self._ensure_path_between_keypoint()
        self._build_random_walls()

    @staticmethod
    def generate(self, dim: Tuple[int, int]):
        raise NotImplementedError

    def _set_starting_point(self, point: Tuple[int, int] = None):
        if self._starting_point is not None:
            raise KeypointError("The starting keypoint always exist in the map")
        else:
            if point is None:
                point = self.random_keypoint()
            elif point not in self.box:
                raise BoxCoordError("Invalid coordinates")
            if point in self._keypoint:
                raise KeypointError("This point is always a keypoint")
            else:
                self._starting_point = point
                self._keypoint.add(point)
                self[point] = 1

    def _set_ending_point(self, point: Tuple[int, int] = None):
        if self._ending_point is not None:
            raise KeypointError("The ending keypoint always exist in the map")
        else:
            if point is None:
                point = self.random_keypoint()
            elif point not in self.box:
                raise BoxCoordError("Invalid coordinates")
            if point in self._keypoint:
                raise KeypointError("This point is always a keypoint")
            else:
                self._ending_point = point
                self._keypoint.add(point)
                self[point] = 2

    def _set_treasure_point(self, point: Tuple[int, int] = None):
        if self._treasure_point is not None:
            raise KeypointError("The treasure keypoint always exist in the map")
        else:
            if point is None:
                point = self.random_keypoint()
            elif point not in self.box:
                raise BoxCoordError("Invalid coordinates")
            if point in self._keypoint:
                raise KeypointError("This point is always a keypoint")
            else:
                self._treasure_point = point
                self._keypoint.add(point)
                self[point] = 3

    def random_keypoint(self):
        keypoint = self._random_point()
        while keypoint in self._keypoint:
            keypoint = self._random_point()
        return keypoint

    def _random_point(self):
        n, m = self.dim
        return randint(0, n - 1), randint(0, m - 1)

    def _ensure_path_between_keypoint(self):
        ensured_path_to_treasure = self.random_path(self._starting_point, self._treasure_point)
        self._ensure_path(ensured_path_to_treasure)
        # if the ending point is visited during the path to the treasure,
        # a path to the ending point is obviously and implicitly ensured
        if self._ending_point not in ensured_path_to_treasure:
            self._ensure_path_between(self._treasure_point, self._ending_point)

    def _ensure_path_between(self, startpoint, endpoint):
        # draw a random path between 'startpoint' and 'endpoint'. Delete (open) all walls in this path
        ensured_path = self.random_path(startpoint, endpoint)
        self._ensure_path(path=ensured_path)

    def _ensure_path(self, path: list):
        for i in range(len(path) - 1):
            wall = self.wall_between(path[i], path[i + 1])
            self.permanently_open_walls.add(wall)

    def _build_wall_at(self, key):
        x, y = key
        if x % 2 == 0 and y % 2 != 0:
            self._grid[x, y] = -2
        elif x % 2 != 0 and y % 2 == 0:
            self._grid[x, y] = -1

    def _build_random_walls(self, p: float = 0.3):
        assert 0 <= p <= 1
        n, m = self._key_dilatation(self.dim)
        for i in range(0, n):
            for j in range(0, m):
                if not self._is_wall((i, j)):
                    continue
                elif (i, j) in self.permanently_open_walls:
                    continue
                elif random() > p:
                    continue
                else:
                    self._build_wall_at((i, j))
