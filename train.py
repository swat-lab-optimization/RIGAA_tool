"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for training RL agents for test scenario generation
"""
import json
import time
from itertools import combinations
from datetime import datetime
import logging as log
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import sys
import os
import config
from rigaa.rl_envs.robot_env import RobotEnv, RobotEnvEval
from rigaa.rl_envs.vehicle_env import CarEnvEval, CarEnv
from rigaa.utils.calc_novelty import calc_novelty
from evaluate import evaluate

def setup_logging(log_to, debug):
    """
    It sets up the logging system
    """

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Started test generation"

    if log_to is not None:
        file_handler = log.FileHandler(log_to, "w", "utf-8")
        log_handlers.append(file_handler)
        start_msg += " ".join([", writing logs to file: ", str(log_to)])

    if debug == "True":
        debug = True
    else:
        debug = False

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
        prog="train.py",
        description="A script for training RL agents for test scenario generation",
        epilog="For more information, please visit https://github.com/swat-lab-optimization/rigaa-tool ",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="vehicle",
        help="Name of the problem to generate the test scenarios for. Available options: robot, vehicle",
    )
    parser.add_argument(
        "--save_path_name",
        type=str,
        default="",
        help="Name to use to save the logs and models",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="MlpPolicy",
        help="Type of policy to use. Should be specified only for robot enviromnet. Available options: MlpPolicy, CnnPolicy ",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
        help="Number of steps to run the training for",
    )
    parser.add_argument(
        "--run_num",
        type=int,
        default=1,
        help="Number of times to run the training",
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0,
        help="The entropy coefficient for RL agent training",
    )
    parser.add_argument(
        "--debug",
        type=str,
        default=False,
        help="Run in debug mode, possible values: True, False",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":

    args = parse_arguments()
    setup_logging("./rl_train_log", args.debug)
    policy = args.policy
    run_num = args.run_num

    if args.problem == "robot":
        environ = RobotEnv(policy)
        problem = "robot"
    elif args.problem == "vehicle":
        environ = CarEnv()
        problem = "vehicle"

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y")
    name = dt_string + "_" + args.problem + "_" + args.save_path_name

    train_times = {}
    final_results = {}
    final_novelty = {}
    for m in range(run_num):
        checkpoint_callback_ppo = CheckpointCallback(
            save_freq=20000,
            save_path=name + "_rl_training_models",
            name_prefix=name + "_" + str(m),
        )
        log_path = dt_string + "_tensorboard_logs"

        start = time.time()

        model = PPO(
            policy,
            environ,
            verbose=True,
            ent_coef=float(args.ent_coef),
            tensorboard_log=log_path,
        )  # 0.005

        # Start training the agent
        model.learn(
            total_timesteps=args.num_steps,
            tb_log_name=name + "_" + str(m),
            callback=checkpoint_callback_ppo,
        )  
        train_time = time.time() - start
        log.info("Training time: {}".format(train_time))
        train_times[str(m)] = train_time

        results, novelty = evaluate(name, model, problem)

        final_novelty[str(m)] = novelty
        final_results[str(m)] = results

    res_save_path = name + "_rl_training_stats/"
    if not os.path.exists(res_save_path):
        os.makedirs(res_save_path)

    log.info("Saving results to {}".format(res_save_path))

    with open(res_save_path + "_results-ppo.txt", "w") as f:
        json.dump(final_results, f, indent=4)

    with open(res_save_path + "_novelty-ppo.txt", "w") as f:
        json.dump(final_novelty, f, indent=4)

    with open(res_save_path + "_train_time.txt", "w") as f:
        json.dump(train_times, f, indent=4)
