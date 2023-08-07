
import json
import os
import time
import logging as log

from simulator.code_pipeline.beamng_executor import BeamngExecutor # comment if using ubuntu
from simulator.code_pipeline.tests_generation import RoadTestFactory
from simulator.code_pipeline.validation import TestValidator


import config as cf
from rigaa.utils.car_road import Map

from rigaa.utils.get_d4rl_map import get_d4rl_map
from rigaa.utils.evaluate_robot_ant_model import evaluate_robot_ant_model

def execute_vehicle_scenario(scenario):
    """
    The function `execute_vehicle_scenario` executes a vehicle scenario by validating the test, creating
    a road test, executing the test using BeamNG, and returning the fitness value.
    
    Args:
      scenario: The "scenario" parameter is a data structure that contains information about the vehicle
    scenario. It is used to create a road test and execute it using the BeamNG simulator. The specific
    structure and content of the "scenario" parameter may vary depending on the implementation and
    requirements of the code.
    
    Returns:
      the fitness value.
    """
    test_validator = TestValidator(cf.vehicle_env["map_size"])

    map = Map(cf.vehicle_env["map_size"])
    road_points, states = map.get_points_from_states(scenario)

    the_test = RoadTestFactory.create_road_test(road_points)

    is_valid, validation_msg = test_validator.validate_test(the_test)

    print(is_valid)
    print(validation_msg)

    if (is_valid== True):

        res_path  = "BeamNG_res"
        if not(os.path.exists(res_path)):
            os.mkdir(res_path)

        executor  = BeamngExecutor(res_path, cf.vehicle_env["map_size"],
                                time_budget=360,
                                beamng_home="C:\\DIMA\\BeamNG\\BeamNG.tech.v0.26.2.0", 
                                beamng_user="C:\\DIMA\\BeamNG\\BeamNG.tech.v0.26.2.0_user", 
                                road_visualizer=None) #RoadTestVisualizer(map_size=cf.vehicle_env["map_size"])
        
        
        test_outcome, description, execution_data = executor._execute(the_test)
        #test_outcome, description, execution_data = executor.execute_test(the_test)
        

        fitness = -max([i.oob_percentage for i in execution_data])

        print("oob", fitness)
    else:
        fitness = 0

    return fitness



def execute_robot_scenario(scenario):
    maze, waypoints  = get_d4rl_map(scenario)
    start = time.time()
    if len(waypoints) < 3:
        fitness = 0
    else:
        #RobotSolution().build_image(self.states, save_path="test.png")
        #for i in maze:
        #    print(i)
        fitness, reward = evaluate_robot_ant_model(maze, waypoints)
        end_time = time.time() - start
        fitness = -1/fitness  # reward#
        log.info("Fitness %s", fitness)
        log.info("Evaluation time %s", end_time)
        #log.info("Reward %s", reward )
    return fitness


if __name__ == "__main__":
    problem = "robot"
    #scenario_path = "C:\\DIMA\\PhD\\RIGAA_tool\\06-01-2023-tcs_full_rigaa_vehicle\\22-12-2022-tcs.json"
    scenario_path = "/home/dmhum/RIGAA_tool/04-03-2023_tcs_fit_rigaa_robot/04-03-2023-tcs.json"
    run = 1
    tc = 0
    with open(scenario_path, "r") as f:
        scenarios = json.load(f)["run" + str(run)]
    
    for scenario in scenarios:
        if problem == "vehicle":
            execute_vehicle_scenario(scenarios[scenario])
        elif problem == "robot":
            execute_robot_scenario(scenarios[scenario])

