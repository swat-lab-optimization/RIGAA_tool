"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for  generating test scenarios for an autonomous vehicle problem with an RL agent
"""

from stable_baselines3 import PPO
import numpy as np
import logging as log

from rigaa.rl_envs.vehicle_env import CarEnvEval
from rigaa.utils.vehicle_evaluate import interpolate_road
from rigaa.utils.vehicle_evaluate import evaluate_scenario
from rigaa.utils.car_road import Map
import config as cf

# from rigaa.samplers.vehicle_sampling import generate_random_road


def generate_rl_road():
    model_save_path = "models\\05-03-2023_vehicle_run1_0_2500000_steps.zip"
    model = PPO.load(model_save_path)

    environ = CarEnvEval()

    scenario_found = False
    i = 0
    while scenario_found == False:
        obs = environ.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)

            obs, rewards, done, info = environ.step(action)
        i += 1
        scenario = environ.state[: environ.steps]
        fitness, _, _ = environ.eval_fitness(scenario)

        fitness = abs(fitness)

        if (fitness > 3.2) or i > 10:

            scenario_found = True

            log.debug(f"Found scenario after {i} attempts")

            i = 0

    return scenario, -fitness
