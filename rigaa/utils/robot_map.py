"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for trnslating the test scenario specification to a set of 2D points
"""
import numpy as np
import logging as log

log.getLogger("matplotlib").setLevel(log.WARNING)


class Map:

    """
    This is a class to transform the input list of states into 2D points
    on a map
    """

    def __init__(self, map_size):
        self.map_size = map_size
        self.max_x = map_size
        self.max_y = map_size
        self.min_x = 0
        self.min_y = 0

        self.init_pos = [0, 1]

        self.map_points = []

        self.all_map_points = np.ones((self.map_size, self.map_size))
        self.current_level = 1

        self.create_init_box()

    def create_init_box(self):
        """
        > The function creates a box around the map, which defines its borders

        Returns:
          the map_size and the all_map_points.
        """
        self.all_map_points[0][:] = 0
        for i in range(1, self.map_size):
            self.all_map_points[i][0] = 0
            self.all_map_points[i][self.map_size - 1] = 0

        self.all_map_points[-1][:] = 0

        # self.all_map_points[-2][:3] = 0
        # self.all_map_points[-3][:3] = 0

        # self.all_map_points[1][-3:] = 0
        # self.all_map_points[2][-3:] = 0
        #

        return

    def horizontal(self, distance, position):
        """
        It takes in a distance and a position, and then it creates a horizontal line of points that are the
        same distance from the center position

        Args:
          distance: the size of the horizontal object
          position: the x-coordinate of the center of the horizontal object

        Returns:
          The new points that are being added to the map.
        """

        new_points = []

        init_pos = [position, self.current_level]
        if self.point_valid(init_pos):
            self.all_map_points[-init_pos[1]][init_pos[0]] = 0
        for i in range(1, round(distance / 2)):
            point_left = [init_pos[0] - i, init_pos[1]]
            point_right = [init_pos[0] + i, init_pos[1]]
            if self.point_valid(point_left):
                self.all_map_points[-point_left[1]][point_left[0]] = 0
            if self.point_valid(point_right):
                self.all_map_points[-point_right[1]][point_right[0]] = 0

        self.current_level += 1

        return new_points

    def horizontal2(self, distance, position):
        """
        It takes in a distance and a position, and then it creates a horizontal line of points that are the
        same distance from the center position

        Args:
          distance: the size of the horizontal object
          position: the x-coordinate of the center of the horizontal object

        Returns:
          The new points that are being added to the map.
        """

        new_points = []

        # init_pos = [self.current_level, position]
        init_pos = [position, self.current_level]
        if self.point_valid(init_pos):
            self.all_map_points[init_pos[1]][init_pos[0]] = 0
        for i in range(1, round(distance / 2)):
            point_left = [init_pos[0] - i, init_pos[1]]
            point_right = [init_pos[0] + i, init_pos[1]]
            if self.point_valid(point_left):
                self.all_map_points[point_left[1]][point_left[0]] = 0
            if self.point_valid(point_right):
                self.all_map_points[point_right[1]][point_right[0]] = 0

        self.current_level += 1

        return new_points

    def vertical(self, distance, position):
        """
        It takes in a distance and a position, and then it creates a vertical line of points that are the
        same distance from the center position

        Args:
          distance: the size of the vertical object
          position: the x-coordinate of the center of the vertical object

        Returns:
          The new points
        """

        new_points = []

        init_pos = [position, self.current_level]
        if self.point_valid(init_pos):
            self.all_map_points[-init_pos[1]][init_pos[0]] = 0
        for i in range(1, round(distance / 2)):
            point_down = [init_pos[0], init_pos[1] - i]
            point_up = [init_pos[0], init_pos[1] + i]
            if self.point_valid(point_down):
                self.all_map_points[-point_down[1]][point_down[0]] = 0
            if self.point_valid(point_up):
                self.all_map_points[-point_up[1]][point_up[0]] = 0

        self.current_level += 1

        return new_points

    def vertical2(self, distance, position):
        """
        It takes in a distance and a position, and then it creates a vertical line of points that are the
        same distance from the center position

        Args:
          distance: the size of the vertical object
          position: the x-coordinate of the center of the vertical object

        Returns:
          The new points
        """

        new_points = []

        init_pos = [position, self.current_level]
        if self.point_valid(init_pos):
            self.all_map_points[init_pos[1]][init_pos[0]] = 0
        for i in range(1, round(distance / 2)):
            point_down = [init_pos[0], init_pos[1] - i]
            point_up = [init_pos[0], init_pos[1] + i]
            if self.point_valid(point_down):
                self.all_map_points[point_down[1]][point_down[0]] = 0
            if self.point_valid(point_up):
                self.all_map_points[point_up[1]][point_up[0]] = 0

        self.current_level += 1

        return new_points

    def point_valid(self, point):
        """
        If the point is in the polygon or out of bounds, then it's not valid

        Args:
          point: the point to be checked

        Returns:
          a boolean value.
        """
        if (self.in_polygon(point)) or self.point_out_of_bounds(point):
            return False
        else:
            return True

    def point_out_of_bounds(self, a):
        """
        If the point is within the bounds of the grid, return False. Otherwise, return True

        Args:
          a: the point to be checked

        Returns:
          a boolean value.
        """
        if (0 <= a[0] and a[0] < self.max_x) and (0 <= a[1] and a[1] < self.max_y):
            return False
        else:
            # print("OUT OF BOUNDS {}".format(a))
            return True

    def in_polygon(self, a):
        """
        If the point is within 5 pixels of the top left corner or the bottom right corner, then it's in the
        polygon i.e in the prohibited area

        Args:
          a: the current vector
        """
        thresh = 5
        if (a[0] < thresh) and (a[1] < thresh):
            # print("IN POLYGON1 {}".format(a))
            return True
        elif (a[0] > self.max_x - thresh) and (a[1] > self.max_y - thresh):
            # print("IN POLYGON2  {}".format(a))
            return True
        else:
            return False

    def get_points_cords(self, points):
        """
        It takes map points as a 2D matrix and returns the list of the actual coordinates of the points

        Args:
          points: a list of lists of 0's and 1's. 0's are the points that are active.

        Returns:
          A list of points that represent the robot obstalces.
        """
        cords = []
        for i, row in enumerate(reversed(points)):
            for j, point in enumerate(row):
                if point == 0:
                    cords.append([j, i])

        # to_remove = [[1, 1], [1, 2], [2, 1], [2, 2], [self.max_x - 2, self.max_y - 2], [self.max_x - 3, self.max_y - 3], [self.max_x - 3, self.max_y - 2], [self.max_x - 2, self.max_y - 3]]
        # for r in to_remove:
        #    cords.remove(r)

        return cords

    def get_points_from_states(self, states, full=False):
        """
        It takes a list states describing the map with obstacles and returns a list of points

        Args:
          states: a list of all the states describing the obstacle map

        Returns:
          The actual 2D matrix of the map.
        """

        self.current_level = 1

        self.create_init_box()

        tc = states
        for state in tc:
            action = int(state[0])
            if action == 0:
                if full == True:
                    self.horizontal2(int(state[1]), int(state[2]))
                else:
                    self.horizontal(int(state[1]), int(state[2]))
            elif action == 1:
                if full == True:
                    self.vertical2(int(state[1]), int(state[2]))
                else:
                    self.vertical(int(state[1]), int(state[2]))
            else:
                log.error("ERROR: Invalid action")
        return self.all_map_points
