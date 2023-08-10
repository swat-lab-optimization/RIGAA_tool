import json
import os
import time
import logging as log
import argparse
import config as cf
from rigaa.utils.car_road import Map
from rigaa.solutions import VehicleSolution
from rigaa.solutions.robot_solution import RobotSolution


def parse_arguments():
    """
    This function parses the arguments passed to the script
    :return: The arguments that are being passed to the program
    """

    log.info("Parsing the arguments")
    parser = argparse.ArgumentParser(
        prog="play_scenario.py", description="A script to play a scenario"
    )

    parser.add_argument(
        "--problem",
        type=str,
        default="vehicle",
        help="Name of the problem to generate the test scenarios for. Available options: robot, vehicle",
    )
    parser.add_argument(
        "--scenario_path",
        nargs="+",
        help="The path to the json file with the scenarios",
        required=True,
    )
    parser.add_argument(
        "--run",
        type=int,
        help="The number of the run to execute",
        required=False,
        default=0,
    )
    in_arguments = parser.parse_args()
    return in_arguments


if __name__ == "__main__":
    args = parse_arguments()
    problem = args.problem
    scenario_path = args.stats_path
    run = args.run
    with open(scenario_path, "r") as f:
        scenarios = json.load(f)["run" + str(run)]

    for scenario in scenarios:
        if problem == "vehicle":
            s = VehicleSolution()
            s.states = scenarios[scenario]
            fitness = s.eval_fitness_full()
        elif problem == "robot":
            s = RobotSolution()
            s.states = scenarios[scenario]
            fitness = s.eval_fitness_full()
