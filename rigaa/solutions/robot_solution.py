"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for representing the robot problem solution
"""

import matplotlib.pyplot as plt
from shapely.geometry import LineString
import sys
import config as cf
from rigaa.utils.robot_map import Map
from rigaa.utils.a_star import AStarPlanner
import logging as log

log.getLogger("matplotlib").setLevel(log.WARNING)
import time

if sys.platform.startswith("linux"):
    from rigaa.utils.get_d4rl_map import get_d4rl_map
    from rigaa.utils.evaluate_robot_ant_model import evaluate_robot_ant_model


class RobotSolution:
    """
    This is the class to contain all the information about the candidate solution
    It also contains the methods to evaluate the fitness of the solution, novelty and build the image
    """

    def __init__(self):

        self.map_points = []
        self.map_size = cf.robot_env["map_size"]
        self.states = []
        self.fitness = 0
        self.novelty = 0
        self.sx = cf.robot_env["sx"]  # [m]
        self.sy = cf.robot_env["sy"]  # [m]
        self.gx = cf.robot_env["gx"]  # [m]
        self.gy = cf.robot_env["gy"]  # [m]
        self.robot_path_x = []
        self.robot_path_y = []

        self.grid_size = cf.robot_env["grid_size"]  # [m]
        self.robot_radius = cf.robot_env["robot_radius"]  # [m]

    def eval_fitness(self):
        """
        > This function returns a fitness score
        """
        map_builder = Map(self.map_size)
        self.map_points = map_builder.get_points_from_states(self.states)
        points_list = map_builder.get_points_cords(self.map_points)

        ox = [t[0] for t in points_list]
        oy = [t[1] for t in points_list]

        a_star = AStarPlanner(ox, oy, self.grid_size, self.robot_radius)  # noqa: E501
        rx, ry, _ = a_star.planning(self.sx, self.sy, self.gx, self.gy)

        self.robot_path_x = rx
        self.robot_path_y = ry
        path = zip(rx, ry)

        if len(rx) > 2:
            test_road = LineString([(t[0], t[1]) for t in path])
            self.fitness = -test_road.length
        else:
            self.fitness = 0

        return self.fitness

    def eval_fitness_full(self):
        maze, waypoints = get_d4rl_map(self.states)
        start = time.time()
        if len(waypoints) < 3:
            self.fitness = 0
        else:
            fitness, reward = evaluate_robot_ant_model(maze, waypoints)
            end_time = time.time() - start
            self.fitness = -1 / fitness  # reward#
            log.info("Fitness %s", self.fitness)
            log.info("Evaluation time %s", end_time)
            # log.info("Reward %s", reward )
        return self.fitness

    def intersect(self, tc1, tc2):
        """
        Compute the intersection of two sets (two test cases)

        Args:
          state1: the first element to compare
          state2: the second element to compare

        Returns:
          The list of similar elements in the two test cases
        """
        intersection = []
        tc_size = min(len(tc1), len(tc2))
        for i in range(tc_size):
            if tc1[i][0] == tc2[i][0]:
                if (abs(tc1[i][1] - tc2[i][1]) <= 2) and (
                    abs(tc1[i][2] - tc2[i][2]) <= 2
                ):
                    intersection.append(tc1[i])

        return intersection

    def calculate_novelty(self, tc1, tc2):
        """
        > The novelty of two test cases is the proportion of states that are unique to each test case
        We implement novelty calculation according to Jaccard distance definition:
        intersection/(set1 size + set2 size - intersection)

        :param tc1: The first test case
        :param tc2: The test case that is being compared to the test suite
        :return: The novelty of the two test cases.
        """
        intersection = self.intersect(tc1, tc2)
        total_states = len(tc1) + len(tc2) - len(intersection)

        novelty = 1 - len(intersection) / total_states
        return -novelty

    @staticmethod
    def build_image(states, save_path="test.png"):
        """
        It takes a list of states and saves an image to the specified path

        Args:
          states: The list of states to build the image from.
          save_path: The path to save the image to. Defaults to test.png
        """

        fig, ax = plt.subplots(figsize=(12, 12))

        map_size = cf.robot_env["map_size"]

        map_builder = Map(map_size)
        grid_size = cf.robot_env["grid_size"]
        robot_radius = cf.robot_env["robot_radius"]
        sx, sy = cf.robot_env["sx"], cf.robot_env["sy"]
        gx, gy = cf.robot_env["gx"], cf.robot_env["gy"]
        map_points = map_builder.get_points_from_states(states, full=False)
        points_list = map_builder.get_points_cords(map_points)

        ox = [t[0] for t in points_list]
        oy = [t[1] for t in points_list]

        a_star = AStarPlanner(ox, oy, grid_size, robot_radius)  # noqa: E501
        rx, ry, _ = a_star.planning(sx, sy, gx, gy)

        ax.plot(rx, ry, "-r", label="Robot path")
        ax.scatter(ox, oy, s=150, marker="s", color="k", label="Walls")

        if len(rx) > 2:
            fitness = LineString([(t[0], t[1]) for t in zip(rx, ry)]).length
        else:
            fitness = 0
        top = map_size + 1
        bottom = -1
        ax.set_title("Test case fitenss " + str(fitness), fontsize=17)

        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.set_ylim(bottom, top)
        ax.set_xlim(bottom, top)
        ax.legend(fontsize=22)

        fig.savefig(save_path)

        plt.close(fig)
