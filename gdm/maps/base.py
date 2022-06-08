import numpy as np
from random import choice
from typing import Tuple


_char_map = {-3: '+', -2: '—', -1: '|', 0: '', 1: 'I', 2: 'O', 3: 'T'}


class Maps:
    """

    """

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        obj.dim = (4, 4)
        obj.permanently_closed_walls = set()
        obj.permanently_open_walls = set()
        return obj

    def __init__(self, dim: tuple = (4, 4)):
        self.dim = dim
        n, m = self.dim
        self._grid = np.zeros((2 * n + 1, 2 * m + 1), dtype=int)
        self._fill_unreachable_points()
        self._fill_external_walls()
        self.box = np.zeros(dim)

    def __setitem__(self, key, value):
        self.box[key] = value
        dil_key = self._key_dilatation(key)
        self._grid[dil_key] = value

    def __getitem__(self, item):
        return self.box[item]

    def __repr__(self):
        repr_ = "\n".join([
            '\t'.join([_char_map[value] for value in line]) for line in self._grid]
        )
        return repr_

    def __contains__(self, coord):
        n, m = self.box.shape
        x, y = coord
        try:
            x = int(x)
            y = int(y)
            if 0 <= x < n and 0 <= y < m:
                return True
            else:
                return False
        except:
            BoxCoordError("Coordinates must be integers")

    def _key_dilatation(self, key):
        if type(key) == int:
            return self._int_dilatation(key)
        elif type(key) == slice:
            return self._slice_dilatation(key)
        elif type(key) == tuple:
            x, y = key
            dil_x = self._item_dilatation(x)
            dil_y = self._item_dilatation(y)
            return dil_x, dil_y

    def _item_dilatation(self, item: int | slice | list):
        if type(item) == int:
            return self._int_dilatation(item)
        elif type(item) == slice:
            return self._slice_dilatation(item)
        else:
            try:
                return [self._int_dilatation(i) for i in item]
            except:
                raise KeyError("Item must be int, slice or iterable")

    def _slice_dilatation(self, slice_: slice) -> slice:
        if (start := slice_.start) is not None:
            start = start * 2 + 1
        else:
            start = 1
        if (stop := slice_.stop) is not None:
            stop = stop * 2 + 1
        if (step := slice_.step) is not None:
            step = step * 2
        else:
            step = 2
        return slice(start, stop, step)

    def _int_dilatation(self, x: int) -> int:
        return 2 * x + 1

    def _is_wall(self, coord: Tuple[int, int]):
        """The two coords must have opposite parities"""
        x, y = coord
        return x % 2 ^ y % 2

    def _is_box(self, coord: Tuple[int, int]):
        """Both coords must be odd"""
        x, y = coord
        return x % 2 & y % 2

    def _is_unreachable(self, coord: Tuple[int, int]):
        x, y = coord
        return (not x % 2) & (not y % 2)

    def _fill_external_walls(self):
        n, m = self._grid.shape
        i, j = range(1, n, 2), range(1, m, 2)
        self._grid[i, 0] = -1  # left filling
        self._grid[i, m - 1] = -1  # right filling
        self._grid[0, j] = -2  # top filling
        self._grid[n - 1, j] = -2  # bottom filling

    def _fill_unreachable_points(self):
        n, m = self._grid.shape
        for i in range(0, n, 2):
            self._grid[i, range(0, m, 2)] = -3

    def get_wall(self, wall_coord):
        assert self._is_wall(wall_coord)
        return self._grid[wall_coord]

    def get_walls_around(self, point: Tuple[int, int]):
        """

        :param point:
        :return:
        """
        assert point in self
        x, y = point
        x = self._int_dilatation(x)
        y = self._int_dilatation(y)
        if not self._is_box((x, y)):
            raise BoxCoordError(f"({point}) doesn't match a box coordinate")
        return {"top": ((x - 1, y), _char_map[self._grid[(x - 1, y)]]),
                "bottom": ((x + 1, y), _char_map[self._grid[(x + 1, y)]]),
                "left": ((x, y - 1), _char_map[self._grid[(x, y - 1)]]),
                "right": ((x, y + 1), _char_map[self._grid[(x, y + 1)]]), }

    def wall_between(self, point_1, point_2) -> Tuple[int, int]:
        if point_2 not in self.neighbours(point_1):
            raise Exception("These two points are not neighbours")
        x_1, y_1 = self._key_dilatation(point_1)
        x_2, y_2 = self._key_dilatation(point_2)
        return (x_2 + x_1) // 2, (y_2 + y_1) // 2

    def neighbours(self, point: Tuple[int, int]):
        assert point in self
        x, y = point
        potential_neighbours = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        neighbours = filter(lambda x: x in self, potential_neighbours)
        return list(neighbours)

    def random_path(self, startpoint, endpoint) -> list:
        """
        Neighbour are eligible when not yet visited and not a red point
        If at least one neighbour is eligible, a random eligible neighbor is set marked as current point
            and become the last visited point.
        If all neighbours are not eligible, the current point will be unmarked as visited and marked as redpoint
            then the current point become the previous point
        :param startpoint:
        :param endpoint:
        :return:
        """
        assert startpoint in self
        assert endpoint in self
        assert startpoint != endpoint
        visited_points = list()
        red_points = list()
        current_point = startpoint
        visited_points.append(current_point)

        def eligible(x):
            return (x not in visited_points) and (x not in red_points)

        while current_point != endpoint:
            neighbours = self.neighbours(current_point)
            eligible_neighbours = list(filter(eligible, neighbours))
            if eligible_neighbours:
                current_point = choice(eligible_neighbours)
                visited_points.append(current_point)
            else:
                red_points.append(visited_points.pop(-1))
                current_point = visited_points[-1]

        return visited_points


class BoxCoordError(Exception):
    pass


class KeypointError(Exception):
    pass


__all__ = [Maps, BoxCoordError, KeypointError]
