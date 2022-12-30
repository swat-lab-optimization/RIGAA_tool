
import time
import json
import os
import csv
from datetime import datetime
from itertools import combinations
import logging as log
from rigaa.samplers import GENERATORS
from rigaa.solutions.vehicle_solution import VehicleSolution
from rigaa.utils.calc_novelty import calc_novelty
import config as cf

def setup_logging(log_to, debug):
    """
    It sets up the logging system
    """
    #def log_exception(extype, value, trace):
    #    log.exception('Uncaught exception:', exc_info=(extype, value, trace))

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    #term_handler.setLevel(log.INFO)
    start_msg = "Started test generation"

    if log_to is not None:
        file_handler = log.FileHandler(log_to, 'w', 'utf-8')
        #file_handler.setLevel(log.DEBUG)
        log_handlers.append(file_handler)
        start_msg += " ".join([", writing logs to file: ", str(log_to)])

    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',  level=log_level, handlers=log_handlers, force=True)

    #sys.excepthook = log_exception

    log.info(start_msg)


def full_model_eval(scenario):
    vehicle = VehicleSolution()
    vehicle.states = scenario
    fitness = vehicle.eval_fitness_full()
    return fitness

def get_stats(scenarios, fitness, times, problem):
    res_dict = {}
    res_dict["fitness"] = fitness
    res_dict["times"] = times
    
    novelty_list = []
    for i in combinations(range(0, len(scenarios)), 2):
        current1 = scenarios[i[0]] #res.history[gen].pop.get("X")[i[0]]
        current2 = scenarios[i[1]] #res.history[gen].pop.get("X")[i[1]]
        nov = calc_novelty(current1, current2, problem)
        novelty_list.append(nov)
    novelty = sum(novelty_list) / len(novelty_list)

    
    res_dict["novelty"] = novelty

    return res_dict


def save_results(results, algo, problem):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y")

    stats_path = dt_string + "_" + cf.files["stats_path"] + "_" + algo + "_" + problem

    if not os.path.exists(stats_path):
        os.makedirs(stats_path)


    with open(
        os.path.join(stats_path, dt_string + "-stats.json"), "w"
    , encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        log.info(
            "Stats saved as %s",
            os.path.join(stats_path, dt_string + "-stats.json")
        )




def compare_generators(problem, runs, test_scenario_num, full_model=False):
    generator = GENERATORS[problem]
    generator_rl = GENERATORS[problem + "_rl"]

    run_stats = {}
    run_stats_rl = {}
    f = open('results.csv', 'w')
    writer = csv.writer(f)
    row = ["Model", "Simulator"]
    writer.writerow(row)
    f.close

    for run in range(runs):
        log.info("Running run %d", run)

        test_scenarios = []
        test_scenarios_rl = []

        times = []
        times_rl = []

        scenario_rl_fitness = []
        scenario_fitness = []

        full_eval = False
        if full_model and problem=="vehicle":
            full_eval = True

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
                fitness = full_model_eval(scenario)
                fitness_rl = full_model_eval(scenario_rl)
                f = open('results.csv', 'a')
                writer = csv.writer(f)
                row1 = [fit_model1, fitness]
                row2 = [fit_model2, fitness_rl]
                writer.writerow(row1)
                writer.writerow(row2)
                f.close
                log.info("Finished evaluation, saving to file")

            scenario_fitness.append(fitness)
            scenario_rl_fitness.append(fitness_rl)




        results = get_stats(test_scenarios, scenario_fitness, times, problem)
        results_rl = get_stats(test_scenarios_rl, scenario_rl_fitness, times_rl, problem)

        run_stats["run" + str(run)] = results
        run_stats_rl["run" + str(run)] = results_rl

        save_results(run_stats, "random_gen", problem)
        save_results(run_stats_rl, "rl_gen", problem)


            
            

        
if __name__ == "__main__":
    problem = "vehicle"
    runs = 10
    test_scenario_num = 30
    setup_logging("log.txt", False)
    compare_generators(problem, runs, test_scenario_num, full_model=True)

