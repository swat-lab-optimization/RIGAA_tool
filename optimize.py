"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for running search-based test generation
"""

import argparse
import sys
import logging as log

log.getLogger("matplotlib").setLevel(log.WARNING)

from pymoo.optimize import minimize
from pymoo.termination import get_termination

import config as cf
from rigaa import ALRGORITHMS
from rigaa.duplicate_elimination.duplicate_rem import DuplicateElimination
from rigaa.problems import PROBLEMS
from rigaa.samplers import SAMPLERS
from rigaa.search_operators import OPERATORS
from rigaa.utils.get_convergence import get_convergence
from rigaa.utils.get_stats import get_stats
from rigaa.utils.get_test_suite import get_test_suite
from rigaa.utils.random_seed import get_random_seed
from rigaa.utils.save_tc_results import save_tc_results, create_summary
from rigaa.utils.save_tcs_images import save_tcs_images
from rigaa.utils.callback import DebugCallback
from datetime import datetime
import os

def setup_logging(log_to, debug):
    """
    It sets up the logging system
    """

    if debug == "True":
        debug = True
    else:
        debug = False

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

    log.info(start_msg)


def parse_arguments():
    """
    Function for parsing the arguments
    """
    parser = argparse.ArgumentParser(
        prog="optimize.py",
        description="A tool for generating test cases for autonomous systems",
        epilog="For more information, please visit https://github.com/swat-lab-optimization/rigaa-tool ",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="vehicle",
        help="Problem to solve, possivle values: vehicle, robot",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="nsga2",
        help="Algorithm to use, possivle values: nsga2, ga, random",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument(
        "--save_results",
        type=str,
        default=True,
        help="Save results, possible values: True, False",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--debug",
        type=str,
        default=False,
        help="Run in debug mode, possible values: True, False",
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=None,
        help="Number of evaluations to run. This parameter overwrites number of generations in the config file",
    )
    parser.add_argument(
        "--n_offsprings",
        type=int,
        default=None,
        help="Number of offspring to generate in each generation. This parameter overwrites number of offspring in the config file",
    )
    parser.add_argument(
        "--full",
        type=str,
        default="False",
        help="Whether to run the evaluation using a simulator: True, False",
    )
    parser.add_argument(
        "--eval_time",
        type=str,
        default=None,
        help="Time to run the algorithm. n_eval should not be specified to use eval_time",
    )
    parser.add_argument(
        "--ro",
        type=float,
        default=0.2,
        help="Percentage of the initial population generated by the RL agent. Possible values from 0 to 1.",
    )

    arguments = parser.parse_args()
    return arguments

def main(
    problem,
    algo,
    runs_number,
    save_results,
    random_seed,
    debug,
    n_eval,
    full,
    eval_time,
    n_offsprings,
    ro=0.4
):
    """
    Function for running the optimization and saving the results"""

    log_file = "logs.txt"

    setup_logging(log_file, debug)

    log.info("Running the optimization")
    log.info(
        "Problem: %s, Algorithm: %s, Runs number: %s, Saving the results: %s",
        problem,
        algo,
        runs_number,
        save_results,
    )

    log.info(f"Using ro {ro}")

    if cf.ga["pop_size"] < cf.ga["test_suite_size"]:
        log.error("Population size should be greater or equal to test suite size")
        sys.exit(1)


    if n_offsprings is None:
        n_offsprings = int(cf.ga["pop_size"]/2)

    if algo == "rigaa" or algo == "rigaa_s":
        rl_pop_percent = ro
    else:
        rl_pop_percent = 0
    algorithm = ALRGORITHMS[algo](
        n_offsprings=n_offsprings,
        pop_size=cf.ga["pop_size"],
        sampling=SAMPLERS[problem](rl_pop_percent),
        crossover=OPERATORS[problem + "_crossover"](cf.ga["cross_rate"]),
        mutation=OPERATORS[problem + "_mutation"](cf.ga["mut_rate"]),
        eliminate_duplicates=DuplicateElimination(algo, problem),
        n_points_per_iteration=n_offsprings,
    )

    if n_eval is not None:
        termination = get_termination("n_eval", n_eval)
        log.info("The search will be terminated after %d evaluations", n_eval)
    elif eval_time is not None:
        termination = get_termination("time", eval_time)
        log.info("The search will be terminated after %s ", eval_time)
    else:
        termination = get_termination("n_gen", cf.ga["n_gen"])
        log.info("The search will be terminated after %d generations", cf.ga["n_gen"])

    if full == "True":
        full = True
    else:
        full = False

    tc_stats = {}
    tcs = {}
    tcs_convergence = {}
    tcs_all_stats = {}
    
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y")
    tcs_hyper = {}
    for m in range(runs_number):
        log.info("Executing run %d: ", m)
        if (random_seed is not None) and (m == 0):
            seed = random_seed
        else:
            seed = get_random_seed()

        log.info("Using random seed: %s", seed)


        sim_path = dt_string + "-rigaa-results_BEAM_NG_ro_" + str(rl_pop_percent)
        sim_path = os.path.join(sim_path, str(m))

        res = minimize(
            PROBLEMS[problem + "_" + algo](full=full, sim_path=sim_path),
            algorithm,
            termination,
            seed=seed,
            verbose=True,
            save_history=True,
            eliminate_duplicates=True,
            #callback=DebugCallback(debug),
        )

        log.info("Execution time, %f sec", res.exec_time)
        
        test_suite = get_test_suite(res, algo)
        tc_stats["run" + str(m)] = get_stats(res, problem, algo)
        tcs["run" + str(m)] = test_suite
        if full:
            tcs_all_stats["run" + str(m)] = res.problem.execution_data


        tcs_convergence["run" + str(m)], tcs_hyper["run" + str(m)] = get_convergence(res, n_offsprings)

        if save_results == "True":
            save_tc_results(tc_stats, tcs, tcs_convergence, tcs_hyper, tcs_all_stats, dt_string, algo, problem, "_ro_" + str(rl_pop_percent))
            save_tcs_images(test_suite, problem, m, algo, dt_string, "_ro_" + str(rl_pop_percent))


        if full:
            create_summary(sim_path, res.problem.executor.get_stats())



################################## MAIN ########################################

if __name__ == "__main__":
    args = parse_arguments()
    main(
        args.problem,
        args.algorithm,
        args.runs,
        args.save_results,
        args.seed,
        args.debug,
        args.n_eval,
        args.full,
        args.eval_time,
        args.n_offsprings,
        ro=args.ro
    )
#
# python optimize.py --problem "vehicle" --algorithm "rigaa" --runs 20 --save_results "True" --n_eval 65000 --debug "False"