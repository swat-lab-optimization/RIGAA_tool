"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for generating mazes with an RL agent 
"""

from stable_baselines3 import PPO
from rigaa.rl_envs.robot_env import RobotEnvEval
from rigaa.utils.robot_map import Map
from rigaa.utils.a_star import AStarPlanner
import config as cf
import numpy as np


def generate_rl_map():
    model_save_path = "models/2023-01-28-rl_model-0_600000_steps.zip"
    model = PPO.load(model_save_path)

    environ = RobotEnvEval()

    scenario_found = False

    i = 0
    while scenario_found == False:
        obs = environ.reset()
        done = False
        # i = 0
        while not done:
            action, _ = model.predict(obs)

            obs, rewards, done, info = environ.step(action)
        i += 1

        map_builder = Map(cf.robot_env["map_size"])
        map_points = map_builder.get_points_from_states(environ.state, full=True)

        points_list = map_builder.get_points_cords(map_points)

        grid_size = cf.robot_env["grid_size"]  # [m]
        robot_radius = cf.robot_env["robot_radius"]  # [m]

        ox = [t[0] for t in points_list]
        oy = [t[1] for t in points_list]
        sx = cf.robot_env["sx"]  # [m]
        sy = cf.robot_env["sy"]  # [m]
        gx = cf.robot_env["gx"]  # [m]
        gy = cf.robot_env["gy"]  # [m]

        a_star = AStarPlanner(ox, oy, grid_size, robot_radius)  # noqa: E501
        rx, ry, rtime = a_star.planning(sx, sy, gx, gy)

        if len(rx) < 3:
            fitness = 0
        else:
            fitness = len(rx)

        if (fitness > 65) or i > 15:
            scenario = environ.state
            scenario_found = True
            print("Found scenario after %d attempts" % i)

            i = 0

    return scenario, -fitness
