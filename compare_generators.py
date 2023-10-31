"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for comparing random and RL-based test scenario generators
"""

import time
import json
import os
import csv
from datetime import datetime
from itertools import combinations
import sys
import argparse
import logging as log

log.getLogger("matplotlib").setLevel(log.WARNING)
from rigaa.samplers import GENERATORS
from rigaa.solutions.vehicle_solution import VehicleSolution
from rigaa.solutions.robot_solution import RobotSolution
from rigaa.utils.calc_novelty import calc_novelty
import config as cf
from rigaa.utils.save_tcs_images import save_tcs_images


def setup_logging(log_to, debug):
    """
    It sets up the logging system
    """
    # def log_exception(extype, value, trace):
    #    log.exception('Uncaught exception:', exc_info=(extype, value, trace))

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    # term_handler.setLevel(log.INFO)
    start_msg = "Started test generation"

    if log_to is not None:
        file_handler = log.FileHandler(log_to, "w", "utf-8")
        # file_handler.setLevel(log.DEBUG)
        log_handlers.append(file_handler)
        start_msg += " ".join([", writing logs to file: ", str(log_to)])

    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=log_level,
        handlers=log_handlers,
        force=True,
    )

    # sys.excepthook = log_exception

    log.info(start_msg)


def full_model_eval(scenario, problem):
    if "vehicle" in problem:
        vehicle = VehicleSolution()
        vehicle.states = scenario
        if problem == "vehicle_fr":
            fitness = vehicle.eval_fitness_full(road=scenario)
        else:
            fitness = vehicle.eval_fitness_full()
    elif problem == "robot":
        robot = RobotSolution()
        robot.states = scenario
        fitness = robot.eval_fitness_full()
    else:
        log.error("Invalid problem")
        sys.exit(1)
    return fitness


def get_stats(scenarios, fitness, times, problem):
    """
    The function "get_stats" takes in scenarios, fitness, times, and problem as parameters, and returns
    a dictionary containing the fitness, times, and novelty.
    
    :param scenarios: A list of scenarios, which are inputs to the problem being solved
    :param fitness: The fitness parameter is a measure of how well a solution performs in solving the
    given problem. It could be a numerical value representing the quality of the solution, such as a
    fitness score or an objective function value
    :param times: The "times" parameter is a list that contains the execution times of each scenario in
    the "scenarios" list
    :param problem: The `problem` parameter is the problem instance or configuration that is being
    solved or analyzed. It could be any information or data related to the problem that is needed for
    calculating the statistics
    :return: a dictionary containing the fitness, times, and novelty values.
    """
    res_dict = {}
    res_dict["fitness"] = fitness
    res_dict["times"] = times

    novelty_list = []
    novelty = 0

    '''
    for i in combinations(range(0, len(scenarios)), 2):
        current1 = scenarios[i[0]]  # res.history[gen].pop.get("X")[i[0]]
        current2 = scenarios[i[1]]  # res.history[gen].pop.get("X")[i[1]]
        nov = calc_novelty(current1, current2, problem)
        novelty_list.append(nov)
    novelty = sum(novelty_list) / len(novelty_list)
    '''

    res_dict["novelty"] = novelty

    return res_dict


def save_results(results, algo, problem):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y")

    stats_path = dt_string + "_" + cf.files["stats_path"] + "_" + algo + "_" + problem

    if not os.path.exists(stats_path):
        os.makedirs(stats_path)

    with open(
        os.path.join(stats_path, dt_string + "-stats.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4)
        log.info(
            "Stats saved as %s", os.path.join(stats_path, dt_string + "-stats.json")
        )


def compare_generators(problem, runs, test_scenario_num, full_model=False):
    """
    The function `compare_generators` compares the performance of two different generators by running
    them multiple times and collecting statistics on the generated scenarios.
    
    :param problem: The "problem" parameter is a string that represents the problem for which the
    generators are being compared. It is used to select the appropriate generator functions from the
    `GENERATORS` dictionary
    :param runs: The parameter "runs" represents the number of times the comparison between generators
    will be run. It determines how many times the generator functions will be called and evaluated
    :param test_scenario_num: The parameter "test_scenario_num" represents the number of test scenarios
    that will be generated and evaluated in each run of the comparison
    :param full_model: The `full_model` parameter is a boolean flag that determines whether to perform a
    full evaluation of the generated scenarios using a separate evaluation function (`full_model_eval`).
    If `full_model` is set to `True`, the evaluation function will be called for each generated scenario
    to calculate its fitness. If `, defaults to False (optional)
    """
    generator = GENERATORS[problem]
    #generator_rl = GENERATORS[problem + "_rl"]
    generator_rl = GENERATORS[problem + "_frenetic"]

    run_stats = {}
    run_stats_rl = {}
    f = open("results.csv", "w")
    writer = csv.writer(f)
    row = ["Model", "Simulator"]
    writer.writerow(row)
    f.close

    test_suite_ran = {}
    test_suite_rl = {}
    for run in range(runs):
        log.info("Running run %d", run)

        test_scenarios = []
        test_scenarios_rl = []

        times = []
        times_rl = []

        scenario_rl_fitness = []
        scenario_fitness = []

        full_eval = False
        if full_model:
            full_eval = True

        current_suite_ran = {}
        current_suite_rl = {}

        for i in range(test_scenario_num):

            start = time.time()
            scenario, fitness = generator()
            gen_time = time.time() - start
            times.append(gen_time)

            start = time.time()
            scenario_rl, fitness_rl = generator_rl()
            gen_time = time.time() - start
            times_rl.append(gen_time)

            fit_model1 = fitness
            fit_model2 = fitness_rl

            test_scenarios.append(scenario)
            test_scenarios_rl.append(scenario_rl)
            if full_eval:
                log.info("Evaluating random scenario")
                fitness = full_model_eval(scenario, problem)
                log.info("Fitness random %s", fitness)
                log.info("Evaluating RL scenario")
                fitness_rl = full_model_eval(scenario_rl, problem +"_fr")
                log.info("Fitness RL %s", fitness_rl)

            scenario_fitness.append(fitness)
            scenario_rl_fitness.append(fitness_rl)
            current_suite_ran[str(i)] = scenario
            current_suite_rl[str(i)] = scenario_rl
        test_suite_ran["run" + str(run)] = current_suite_ran
        test_suite_rl["run" + str(run)] = current_suite_rl

        results = get_stats(test_scenarios, scenario_fitness, times, problem)
        results_rl = get_stats(
            test_scenarios_rl, scenario_rl_fitness, times_rl, problem
        )

        run_stats["run" + str(run)] = results
        run_stats_rl["run" + str(run)] = results_rl

        save_results(run_stats, "random_gen", problem)
        save_results(run_stats_rl, "rl_gen", problem)

        save_tcs_images(current_suite_ran, problem, run, "random_gen")
        save_tcs_images(current_suite_rl, problem + "_fr", run, "rl_gen")

        # save_tcs_images(test_suite, problem, m, algo)


def parse_arguments():
    """
    Function for parsing the arguments
    """
    parser = argparse.ArgumentParser(
        prog="compare_generators.py",
        description="A script for comparing the random and RL generators for a given problem",
        epilog="For more information, please visit https://github.com/swat-lab-optimization/rigaa-tool ",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="robot",
        help="Name of the problem to generate the test scenarios for. Available options: robot, vehicle",
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of times to run the comparison"
    )
    parser.add_argument(
        "--tc_num",
        type=int,
        default=30,
        help="Number of test scenarios to generate for each run",
    )
    parser.add_argument(
        "--full",
        type=str,
        default="False",
        help="Whether to run the evaluation using a simulator: True, False",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_arguments()
    problem = args.problem  # "robot"
    runs = args.runs  # 10
    test_scenario_num = args.tc_num  # 30
    setup_logging("log.txt", False)
    if args.full == "True":
        full_model = True
    elif args.full == "False":
        full_model = False
    else:
        raise ValueError("full argument should be either True or False")
    compare_generators(problem, runs, test_scenario_num, full_model=full_model)
