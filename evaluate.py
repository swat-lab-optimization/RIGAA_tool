import argparse
import logging as log
import json
from datetime import datetime
from stable_baselines3 import PPO
import sys
from itertools import combinations
import os
import time

from rigaa.rl_envs.robot_env import RobotEnvEval
from rigaa.rl_envs.vehicle_env import CarEnvEval
from rigaa.utils.calc_novelty import calc_novelty
from rigaa.utils.calc_novelty import calc_novelty

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

    
    if debug == "True":
        debug = True
    elif debug == "False":
        debug = False
    
    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',  level=log.DEBUG, handlers=log_handlers, force=True)

    #sys.excepthook = log_exception

    log.info(start_msg)

def evaluate(name, model, problem):

    log.info("Running the evaluation of the trained agent")
    img_path_save = name + "_rl_training_images/"

    if problem == "robot":
        environ = RobotEnvEval(policy)
        problem = "robot"
    elif problem == "vehicle":
        environ = CarEnvEval()
        problem = "vehicle"
    
    log.info(f"Saving image to {img_path_save}")


    environ.evaluate = True
    episodes = 30
    i = 0
    results = []
    scenario_list = []
    while environ.episode < episodes:
        obs = environ.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            start = time.time()
            obs, rewards, done, info = environ.step(action)
            #log.info("Step time", time.time() - start)
        i += 1
        # max_fitness = max(environ.all_fitness)
        fitness, _, _ = environ.eval_fitness(environ.state[:environ.steps])


        if (fitness > environ.min_fitness) or i > 15: # 15 attepts to produce a good scenario
            log.info(i)
            log.info("Round: {}".format(environ.episode))
            log.info("Max fitness: {}".format(fitness))
            scenario = environ.state
            environ.render( img_path=img_path_save)
            scenario_list.append(scenario)
            environ.episode += 1
            results.append(fitness)
            i = 0

    novelty_list = []
    for i in combinations(range(0, episodes), 2):
        current1 = scenario_list[i[0]]
        current2 = scenario_list[i[1]]
        nov = calc_novelty(current1, current2, problem)
        novelty_list.append(nov)
    novelty = abs(sum(novelty_list) / len(novelty_list))


    return results, novelty




def parse_arguments():
    '''
    Function for parsing the arguments
    '''
    parser = argparse.ArgumentParser(
                    prog = 'train.py',
                    description = 'A script for evaluating RL agents for test scenario generation',
                    epilog = "For more information, please visit https://github.com/swat-lab-optimization/rigaa-tool ")
    parser.add_argument('--problem', type=str, default="robot", help='Name of the problem to generate the test scenarios for. Available options: robot, vehicle')
    parser.add_argument('--save_path_name', type=str, default="", help='Name to use to save the logs and models')
    parser.add_argument('--policy', type=str, default="MlpPolicy", help='Type of policy to use. Should be specified only for robot enviromnet. Available options: MlpPolicy, CnnPolicy ')
    parser.add_argument('--debug', type=str, default=False, help='Run in debug mode, possible values: True, False')
    #parser.add_argument('--eval_flag', type=str, default="True", help='Evaluate the agent after training, possible values: True, False')
    parser.add_argument('--model_path', type=str, default="", help='Path of the model to evaluate')
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    
    args = parse_arguments()
    setup_logging("./rl_eval_log", args.debug)
    policy = args.policy
    m=0

    if args.problem == "robot":
        environ = RobotEnvEval(policy)
        problem = "robot"
    elif args.problem == "vehicle":
        environ = CarEnvEval()
        problem = "vehicle"

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y")
    name  = dt_string + "_" + args.problem + "_" + args.save_path_name


    if args.model_path != "":
        model = PPO.load(args.model_path)
    else:
        log.info("Please specify the path of the model to evaluate")
        sys.exit(0)

    final_results = {}
    final_novelty = {}
    scenario_list = []
    novelty_list = []

    results, novelty = evaluate(name, model, problem)

    final_novelty[str(m)] = novelty
    final_results[str(m)] = results

    res_save_path  = name + "_rl_training_stats/"
    if not os.path.exists(res_save_path):
        os.makedirs(res_save_path)


    log.info("Saving results to {}".format(res_save_path))

    with open(res_save_path + "_results-ppo.txt", "w") as f:
        json.dump(final_results, f, indent=4)

    with open(res_save_path + "_novelty-ppo.txt", "w") as f:
        json.dump(final_novelty, f, indent=4)




