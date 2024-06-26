"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for training an RL agent for Maze generation
"""

from gym import Env
from gym.spaces import Box, MultiDiscrete
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import math
import time
import logging as log
import os
import config as cf
from rigaa.utils.robot_map import Map
from rigaa.utils.a_star import AStarPlanner


class RobotEnv(Env):
    """
    The RobotEnv class defines the environment for a reinforcement learning agent controlling a robot,
    including the action and observation spaces, step function, and rendering function.
    """

    def __init__(self, policy):
        """
        This is the initialization function for a robot environment with defined action and observation
        spaces, state variables, and fitness metrics.

        :param policy: The type of policy used for the reinforcement learning algorithm, either "Mlp" or
        "Cnn"
        """
        super(RobotEnv, self).__init__()
        self.policy = policy
        self.max_number_of_points = cf.robot_env["map_size"] - 2
        self.action_space = MultiDiscrete(
            [
                2,
                cf.robot_env["max_len"] - cf.robot_env["min_len"],
                (
                    cf.robot_env["max_pos"] - cf.robot_env["min_pos"]
                ),  # divide by 2 to reduce the action space
            ]
        )  # 0 - increase temperature, 1 - decrease temperature
        if policy == "MlpPolicy":
            self.observation_space = Box(
                low=0,
                high=self.max_number_of_points,
                shape=(self.max_number_of_points * 3,),
                dtype=np.int8,
            )
        elif policy == "CnnPolicy":
            self.observation_space = Box(
                low=0,
                high=255,
                shape=(cf.robot_env["map_size"], cf.robot_env["map_size"], 1),
                dtype=np.uint8,
            )
        else:
            raise ValueError("Invalid policy type")

        self.state = []
        self.prev_fitness = 0
        self.bonus = 0
        # self.all_states = []
        # self.all_fitness = []
        self.steps = 0
        self.points = []
        self.map_points = []
        self.reward = 0

        self.min_fitness = 70

        self.done = False

        self.position_explored = []
        self.sizes_explored = []

        self.episode = 0
        self.max_steps = 40
        self.fitness = 0

        self.max_fitness = 110

        self.evaluate = False

    def generate_init_state(self):
        """
        This function generates an initial state for a simulation by randomly assigning values to variables
        and initializing arrays.
        """
        self.state = np.zeros((self.max_number_of_points, 3))
        random_position = 0
        ob_type = np.random.randint(0, 2)
        value = np.random.randint(cf.robot_env["min_len"], cf.robot_env["max_len"] + 1)
        position = np.random.randint(
            cf.robot_env["min_pos"], cf.robot_env["max_pos"] + 1
        )
        self.state[random_position] = np.array([ob_type, value, position])
        self.position_explored = [[ob_type, position]]
        self.sizes_explored = [value]

    def eval_fitness(self, states):
        """
        The function evaluates the fitness of a given state by calculating the length of the path generated
        by the A* algorithm between the start and goal points on a map.

        :param states: The states parameter is a list of integers representing the order in which the
        obstacles should be placed on the map. The order of the obstacles affects the path generated by the
        A* algorithm
        :return: The fitness value of the current state of the genetic algorithm is being returned.
        """

        map_builder = Map(cf.robot_env["map_size"])
        map_points = map_builder.get_points_from_states(states)
        points_list = map_builder.get_points_cords(map_points)

        o_x = [t[0] for t in points_list]
        o_y = [t[1] for t in points_list]

        start = time.time()

        a_star = AStarPlanner(
            o_x, o_y, cf.robot_env["grid_size"], cf.robot_env["robot_radius"]
        )  # noqa: E501

        r_x, r_y, _ = a_star.planning(
            cf.robot_env["sx"],
            cf.robot_env["sy"],
            cf.robot_env["gx"],
            cf.robot_env["gy"],
        )

        path = zip(r_x, r_y)

        if len(r_x) > 2:
            path = LineString([(t[0], t[1]) for t in path])
            fitness = path.length
        else:
            fitness = -10

        return round(fitness, 3), points_list, map_points

    def step(self, action):
        """
        The `step` function takes an action, updates the state, evaluates the fitness, calculates the
        reward, and returns the updated observations, reward, done flag, and info dictionary.

        Args:
          action: The `action` parameter represents the action taken by the agent in the environment. It is
        expected to be a valid action that is contained within the action space of the environment.

        Returns:
          a numpy array of observations, the reward, a boolean indicating whether the episode is done, and
        an empty dictionary for additional information.
        """

        start = time.time()

        assert self.action_space.contains(action)

        self.state[self.steps] = self.set_state(action)

        eval_start = time.time()

        self.fitness, self.points, self.map_points = self.eval_fitness(
            self.state
        )  # - discount

        # log.debug(f"Eval time {time.time() - eval_start}")
        # current_state = self.state.copy()

        improvement = self.fitness - self.prev_fitness
        position = [action[0], action[2] + cf.robot_env["min_pos"]]
        value = action[1] + cf.robot_env["min_len"]

        if self.steps >= self.max_steps - 3 or self.fitness < 0:
            self.done = True

        if self.fitness < 0:
            reward = -100
        else:
            reward = self.fitness / 10

            if improvement > 0:
                reward += improvement * 10  # *10

            if not (value in self.sizes_explored):
                reward += 1
                self.sizes_explored.append(value)

            if not (position in self.position_explored):
                reward += 1
                self.position_explored.append(position)

            if self.fitness > self.max_fitness:
                reward += self.fitness

        # self.render()

        self.prev_fitness = self.fitness
        self.reward = reward
        # self.all_fitness.append(self.fitness)
        # self.all_states.append(current_state)

        self.steps += 1

        info = {}

        if self.policy == "MlpPolicy":
            observations = [coordinate for tuple in self.state for coordinate in tuple]
        elif self.policy == "CnnPolicy":
            map_points = self.map_points.astype("uint8") * 255
            observations = np.reshape(
                map_points, (cf.robot_env["map_size"], cf.robot_env["map_size"], 1)
            )
        else:
            raise ValueError("Invalid policy type")

        log.debug(f"Step time {time.time() - start} ")

        return np.array(observations, dtype=np.int8), reward, self.done, info

    def reset(self):
        self.generate_init_state()

        self.prev_fitness, points, map_points = self.eval_fitness(self.state)

        # self.all_states = []
        # self.all_fitness = []

        self.fitness = 0

        self.steps = 1

        self.done = False

        if self.policy == "MlpPolicy":
            observations = [coordinate for tuple in self.state for coordinate in tuple]
        elif self.policy == "CnnPolicy":
            map_points = map_points.astype("uint8") * 255
            observations = np.reshape(
                map_points, (cf.robot_env["map_size"], cf.robot_env["map_size"], 1)
            )
        else:
            raise ValueError("Invalid policy type")

        return np.array(observations, dtype=np.int8)

    def render(self, scenario=[], img_path="./", mode="human"):
        fig, ax = plt.subplots(figsize=(12, 12))

        road_x = []
        road_y = []
        for p in self.points:
            road_x.append(p[0])
            road_y.append(p[1])

        a_star = AStarPlanner(
            road_x, road_y, cf.robot_env["grid_size"], cf.robot_env["robot_radius"]
        )

        r_x, r_y, _ = a_star.planning(
            cf.robot_env["sx"],
            cf.robot_env["sy"],
            cf.robot_env["gx"],
            cf.robot_env["gy"],
        )

        path = list(zip(r_x, r_y))

        robot_path = LineString(path)
        fit = robot_path.length

        ax.plot(r_x, r_y, "-r", label="Robot path")

        title = f"Scenario reward: {self.reward}, scenario fitness: {self.fitness}"
        ax.set_title(title)

        ax.scatter(road_x, road_y, s=150, marker="s", color="k", label="Walls")

        map_size = cf.robot_env["map_size"]
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.set_xlim(0, map_size)
        ax.set_ylim(0, map_size)
        ax.legend(fontsize=22)

        os.makedirs(img_path, exist_ok=True)

        if self.evaluate:
            fig.savefig(f"{img_path}{self.episode}_{fit:.2f}.png", bbox_inches="tight")
        else:
            os.makedirs("debug", exist_ok=True)
            fig.savefig("debug\\debug_step_" + str(self.steps) + ".png")

        plt.close(fig)

    def set_state(self, action):
        """
        This function takes an action and returns a list with modified values for the second and third
        elements.

        :param action: The input action to be performed in the environment. It is a list containing three
        elements:
        :return: a list with three elements: the first element is the same as the first element of the input
        `action` list, the second element is the sum of the second element of the input `action` list and a
        constant value `cf.robot_env["min_len"]`, and the third element is the sum of the third element of the
        input `action` list and a constant value `
        """
        obs_size = action[1] + cf.robot_env["min_len"]
        position = (
            action[2] + cf.robot_env["min_pos"]
        )  # action[2]*2 to reduce the action space

        return [action[0], obs_size, position]


class RobotEnvEval(RobotEnv):
    def __init__(self, policy):
        super().__init__(policy)

    def step(self, action):
        assert self.action_space.contains(action)
        self.done = False

        self.state[self.steps] = self.set_state(action)
        reward = 0

        if self.steps >= self.max_steps - 3:
            self.done = True
            map_builder = Map(cf.robot_env["map_size"])
            map_points = map_builder.get_points_from_states(self.state)

            points_list = map_builder.get_points_cords(map_points)
            self.points = points_list

            # reward, _, _ = eval_fitness(self.state)
            # self.fitness = reward

        self.steps += 1

        info = {}

        obs = [coordinate for tuple in self.state for coordinate in tuple]

        return np.array(obs, dtype=np.int8), reward, self.done, info

    def reset(self):
        self.generate_init_state()

        # self.prev_fitness, points, map_points = eval_fitness(self.state)

        # self.all_states = []
        # self.all_fitness = []

        self.fitness = 0

        self.steps = 1

        self.done = False

        if self.policy == "MlpPolicy":
            observations = [coordinate for tuple in self.state for coordinate in tuple]
        elif self.policy == "CnnPolicy":
            map_points = map_points.astype("uint8") * 255
            observations = np.reshape(
                map_points, (cf.robot_env["map_size"], cf.robot_env["map_size"], 1)
            )
        else:
            raise ValueError("Invalid policy type")

        return np.array(observations, dtype=np.int8)
