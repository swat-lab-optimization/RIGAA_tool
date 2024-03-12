"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for problem definition
"""
import logging as log
from pymoo.core.problem import ElementwiseProblem
import numpy as np
import config as cf
import os
import sys
import threading
import ctypes

class TimeoutError(Exception):
    pass
if sys.platform.startswith("win"):
    from simulator.code_pipeline.beamng_executor import BeamngExecutor
    from simulator.code_pipeline.tests_generation import RoadTestFactory
    from simulator.code_pipeline.validation import TestValidator
class VehicleProblem1Obj(ElementwiseProblem):
    """
    Module to calculate the fitness of the individuals
    """
    def __init__(self, full=False, sim_path=None):
        super().__init__(
            n_var=1, n_obj=1, n_ieq_constr=1
        )  # , xl=np.array([0, 10, 5]), xu=np.array([2, 80, 30])
        self.full = full
        self.execution_data = {}

        res_path = sim_path#"BeamNG_res"
        if not (os.path.exists(res_path)):
            os.mkdir(res_path)
        self.executor = BeamngExecutor(
            res_path,
            cf.vehicle_env["map_size"],
            time_budget=8000,
            beamng_home="",
            beamng_user="",
            road_visualizer=None,
        )
        self.current_test = 0
        self.n_sim = 0
        self.num_failures = 0

    def _evaluate(self, x, out, *args, **kwargs):
        """
        > This function evaluates the individual's fitness and novelty
        Individual is stored in the input vector x

        :param x: the input individual
        :param out: the fitness of the individual as well as the constraint
        """
        s = x[0]

        self.execution_data[str(self.current_test)] = {}
        
        if self.full:
            s.beamng_executor = self.executor
            s.fitness = s.eval_fitness_full()
            data = s.data
            self.execution_data[str(self.current_test)]["test"] = data["test"]
            self.execution_data[str(self.current_test)]["info_validity"] = data["info_validity"]
            self.execution_data[str(self.current_test)]["outcome"] = data["outcome"]
            self.execution_data[str(self.current_test)]["features"] = data["features"]

            self.num_failures +=  data["failure"]

            self.execution_data[str(self.current_test)]["num_failures"] = self.num_failures

            self.n_sim += data["sim"]
            self.execution_data[str(self.current_test)]["n_sim"] = self.n_sim
            

            out["G"] = 0.95 - s.fitness * (-1)
        else:
            s.fitness = s.eval_fitness()
            out["G"] = 4.5 - s.fitness * (-1)
        out["F"] = s.fitness
        # out["G"] = 5 - s.fitness * (-1)

        self.current_test += 1

        #log.debug("Evaluated individual %s, fitness %s", s, s.fitness)


class VehicleProblem2Obj(ElementwiseProblem):
    """
    Module to calculate the fitnes of the individuals
    """

    def __init__(self, full=False, sim_path=None, **kwargs):
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=1)
        self.full = full
        self.execution_data = {}
        res_path = sim_path#"BeamNG_res"
        if not (os.path.exists(res_path)):
            os.makedirs(res_path)
        self.executor = BeamngExecutor(
            res_path,
            cf.vehicle_env["map_size"],
            oob_tolerance=0.85,
            time_budget=8000,
            beamng_home="C:\\Users\\DmytroHUMENIUK\\Documents\\BeamNG.tech.v0.26.2.0",
            beamng_user="C:\\Users\\DmytroHUMENIUK\\Documents\\BeamNG.tech.v0.26.2.0_user",
            road_visualizer=None,
        )
        self.current_test = 0
        self.n_sim = 0
        self.num_failures = 0

    def _evaluate(self, x, out, *args, **kwargs):
        """
        > This function evaluates the individual's fitness and novelty
        Individual is stored in the input vector x

        :param x: the input individual
        :param out: the fitness and novelty of the individual as well as the constraint
        """
        s = x[0]

        self.execution_data[str(self.current_test)] = {}

        '''
        def target():
            nonlocal data
            s.fitness = s.eval_fitness_full()
            data = s.data
            return data
        '''
        
        
        if self.full:
            s.beamng_executor = self.executor
            
            #thread = threading.Thread(target=target)
            #thread.start()
            #thread.join(200)
            try:
                s.fitness = s.eval_fitness_full()
                data = s.data
                #s.fitness = s.eval_fitness_full()
                #data = s.data
             #   if thread.is_alive():
             #       log.info(f"Timeout happened")
                
              #      self.executor.close()

              #      raise TimeoutError("Timeout")

                #data = target()
                self.execution_data[str(self.current_test)]["test"] = data["test"]
                self.execution_data[str(self.current_test)]["info_validity"] = data["info_validity"]
                self.execution_data[str(self.current_test)]["outcome"] = data["outcome"]
                self.execution_data[str(self.current_test)]["features"] = data["features"]
                self.execution_data[str(self.current_test)]["fitness"] = s.fitness
                #log.info("Fitness ", s.fitness)
                self.num_failures +=  data["failure"]
                self.n_sim += data["sim"]
            except Exception as e:
            # Handle specific exceptions
                log.error(f"Exception occurred: {e}")
                #s.fitness = 0
                #data = s.data
                self.execution_data[str(self.current_test)]["test"] = None
                self.execution_data[str(self.current_test)]["info_validity"] = None
                self.execution_data[str(self.current_test)]["outcome"] = "ERROR"
                self.execution_data[str(self.current_test)]["features"] = None
                self.execution_data[str(self.current_test)]["fitness"] = 0
                
            self.execution_data[str(self.current_test)]["num_failures"] = self.num_failures

            
            self.execution_data[str(self.current_test)]["n_sim"] = self.n_sim
            out["G"] = 0.85 - s.fitness * (-1)
        else:
            s.fitness = s.eval_fitness()
            out["G"] = 4.5 - s.fitness * (-1)  # 4.5
            # if len(s.road_points) != len(s.states):
            #    print(s.states)
        algorithm = kwargs["algorithm"]

        solutions = algorithm.pop.get("X")
        if (solutions.size > 0) and (s.fitness < -1):
            top_solutions = solutions[0:5]
            best_scenarios = [
                top_solutions[i][0].states for i in range(len(top_solutions))
            ]

            novelty_list = []
            for i in range(len(best_scenarios)):
                nov = s.calculate_novelty(best_scenarios[i], s.states)
                novelty_list.append(nov)
            s.novelty = sum(novelty_list) / len(novelty_list)
        else:
            s.novelty = 0

        out["F"] = [s.fitness, s.novelty]
        self.current_test += 1

        log.debug(
            "Evaluated individual %s, fitness %s, novelty %s", s, s.fitness, s.novelty
        )
