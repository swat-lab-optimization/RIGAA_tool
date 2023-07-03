import logging as log
import numpy as np
from rigaa.solutions.robot_solution import RobotSolution
from rigaa.samplers.robot_sampling import generate_random_solution
from rigaa.rl_agents.robot_agent import generate_rl_map
import config as cf
import time

def generate_example(alg_type, problem, path, id):
    if problem == "robot":
        if alg_type == "robot":
            pass


if __name__ == "__main__":
    examples_num = 10
    for i in range(examples_num):
        generate_example("RL", "robot", "./examples/robot", i)