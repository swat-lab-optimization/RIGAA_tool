"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: this is a script executed after each generation if the debuggins is enabled
"""

from pymoo.core.callback import Callback
import logging as log
import os

class DebugCallback(Callback):

    def __init__(self, debug) -> None:
        super().__init__()
        self.debug = debug

    def notify(self, algorithm):

        if self.debug:
            population = list(algorithm.pop.get("X"))
            fitness = list(algorithm.pop.get("F"))
            algorithm.n_gen
            log.debug("Current generation: %d ", algorithm.n_gen)
            log.debug("Current population: %s ", population)
            log.debug("Current population fitness: %s ", fitness)
            dir_name = "population"
            if not(os.path.exists(dir_name)):
                os.mkdir(dir_name)
            for sol, i in enumerate(algorithm.pop.get("X")):
                i[0].build_image(i[0].states, os.path.join(dir_name, str(algorithm.n_gen)+ "_"+ str(sol)+".png"))
            

        


