from gdm.maps.base import Maps
from unittest import TestCase
import numpy as np


class TestMaps(TestCase):

    def setUp(self) -> None:
        self.basic_map = Maps(size=(3, 3))
        self.repr = """
+	—	+	—	+	—	+
|						|
+		+		+		+
|						|
+		+		+		+
|						|
+	—	+	—	+	—	+"""

    def test_size(self):
        self.assertEqual(self.basic_map.size, (3, 3))

    def test_repr(self):
        self.assertEqual(repr(self.basic_map), self.repr)

    def test__get_item(self):
        self.assertEqual(list(self.basic_map[-1, :]), [0, 0, 0])

    def test__set_item(self):
        self.basic_map[-1] = -1
        self.basic_map[2, 1] = -2
        self.assertEqual(list(self.basic_map[2, :]), [-1, -2, -1])
        self.assertEqual(self.basic_map[2, 1], -2)

    def test__contains__(self):
        self.assertTrue((1, 2) in self.basic_map)
        self.assertTrue((0, 0) in self.basic_map)
        self.assertTrue((2, 2) in self.basic_map)
        self.assertTrue((0, 3) not in self.basic_map)
        self.assertTrue((-1, 2) not in self.basic_map)

    def test__key_dilatation(self):
        raise NotImplementedError

    def test_get_wall(self):
        self.assertEqual(self.basic_map.get_wall((0, 1)), -2)
        self.assertEqual(self.basic_map.get_wall((1, 0)), -1)
        self.assertRaises(AssertionError, self.basic_map.get_wall, (0, 0))

    def test_get_walls_around(self):
        self.assertEqual(self.basic_map.get_walls_around((1, 0)),
                         {"top": ((2, 1), ""),
                          "bottom": ((4, 1), ""),
                          "left": ((3, 0), "|"),
                          "right": ((3, 2), "")})
        self.assertEqual(self.basic_map.get_walls_around((2, 2)),
                         {'bottom': ((6, 5), '—'),
                          'left': ((5, 4), ''),
                          'right': ((5, 6), '|'),
                          'top': ((4, 5), '')})

    def test_wall_between(self):
        self.assertEqual(self.basic_map.wall_between((1, 1), (1, 2)), (3, 4))
        self.assertRaises(Exception, self.basic_map.wall_between, (1, 1), (2, 2))

    def test_neighbours(self):
        self.assertEqual(self.basic_map.neighbours((0, 0)), {(0, 1), (1, 0)})
        self.assertEqual(self.basic_map.neighbours((2, 0)), {(2, 1), (1, 0)})
        self.assertEqual(self.basic_map.neighbours((0, 2)), {(0, 1), (1, 2)})
        self.assertEqual(self.basic_map.neighbours((2, 2)), {(2, 1), (1, 2)})
        self.assertEqual(self.basic_map.neighbours((1, 1)), {(0, 1), (1, 0), (1, 2), (2, 1)})