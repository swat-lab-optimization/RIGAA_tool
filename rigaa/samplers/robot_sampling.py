"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script generating initial population for the robot problem
"""
import logging as log
import numpy as np
import sys
from pymoo.core.sampling import Sampling
from rigaa.solutions.robot_solution import RobotSolution
from rigaa.utils.robot_map import Map
from rigaa.utils.a_star import AStarPlanner

#if sys.platform.startswith("linux"):
from rigaa.rl_agents.robot_agent import generate_rl_map
import config as cf
import time


def generate_random_solution():
    """
    Given a grid size, robot radius, start and goal coordinates, generate a random solution

    Returns:
        states (list): List of states
        fitness (float): Fitness of the solution
    """
    grid_size = cf.robot_env["grid_size"]
    robot_radius = cf.robot_env["robot_radius"]
    sx = cf.robot_env["sx"]
    sy = cf.robot_env["sy"]
    gx = cf.robot_env["gx"]
    gy = cf.robot_env["gy"]

    map_size = cf.robot_env["map_size"]
    path_size = 0
    while path_size < 2:  # if the path is too short, generate a new solution
        states = []
        for i in range(0, map_size - 1):

            ob_type = np.random.randint(0, 2)
            value = np.random.randint(
                cf.robot_env["min_len"], cf.robot_env["max_len"] + 1
            )
            position = np.random.randint(
                cf.robot_env["min_pos"], cf.robot_env["max_pos"] + 1
            )
            states.append([ob_type, value, position])
        map_builder = Map(map_size)
        map_points = map_builder.get_points_from_states(states, full=True)
        points_list = map_builder.get_points_cords(map_points)
        ox = [t[0] for t in points_list]
        oy = [t[1] for t in points_list]
        a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
        rx, ry, _ = a_star.planning(sx, sy, gx, gy)
        path_size = len(rx)

    # RobotSolution.build_image(states)

    return states, -path_size


class RobotSampling(Sampling):
    """
    Module to sample the initial population
    """

    def __init__(self, init_pop_prob):
        super().__init__()
        self.init_pop_prob = init_pop_prob

    def _do(self, problem, n_samples, **kwargs):
        """
        This is a function to generate the initial population of the algorithm

        returns: a tensor of candidate solutions
        """
        X = np.full((n_samples, 1), None, dtype=object)
        for i in range(n_samples):
            r = np.random.random()
            s = RobotSolution()

            if r < self.init_pop_prob:
                start = time.time()
                states, fitness = generate_rl_map()

                log.debug("Individual produced by RL in %f sec", time.time() - start)
            else:
                start = time.time()
                states, fitness = generate_random_solution()
                log.debug(
                    "Individual produced by randomly in %f sec", time.time() - start
                )

            s.states = states
            s.fitness = fitness
            X[i, 0] = s

        log.debug("Initial population of %d solutions generated", n_samples)
        return X
