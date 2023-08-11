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
import logging as log

def generate_rl_map():
    model_save_path = "models/robot_agent_500000_steps.zip"
    model = PPO.load(model_save_path)
    policy = "MlpPolicy"

    environ = RobotEnvEval(policy)

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

        fitness, _, _ = environ.eval_fitness(environ.state)

        if (fitness > 65) or i > 15:
            scenario = environ.state
            scenario_found = True
            log.debug("Found scenario after %d attempts" % i)

            i = 0

    return scenario, -fitness
