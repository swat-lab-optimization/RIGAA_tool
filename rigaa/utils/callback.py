from pymoo.core.callback import Callback
import logging as log
import os

class DebugCallback(Callback):

    def __init__(self, debug) -> None:
        super().__init__()
        self.debug = debug

    def notify(self, algorithm):
        '''
        for sol, i in enumerate(algorithm.pop.get("X")):
            if len(i[0].road_points) != len(i[0].states):
                print(i[0].states)
                i[0].states = i[0].states[:len(i[0].road_points)].copy()
        '''
        

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
            #current_dir = os.path.join(dir_name, str(algorithm.n_gen))
            #if not(os.path.exists(current_dir)):
            #    os.mkdir(current_dir)
            for sol, i in enumerate(algorithm.pop.get("X")):
                i[0].build_image(i[0].states, os.path.join(dir_name, str(algorithm.n_gen)+ "_"+ str(sol)+".png"))
            

        


