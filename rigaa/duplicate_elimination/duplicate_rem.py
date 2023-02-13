import logging as log
from pymoo.core.duplicate import ElementwiseDuplicateElimination
import time

# It's a duplicate elimination that compares the states of the two elements


class DuplicateElimination(ElementwiseDuplicateElimination):
    '''
    A class to eliminate duplicates in the population.
    '''

    def __init__(self, algo, problem):
        super().__init__()
        self.algo = algo
        self.first_gen = True
        self.problem = problem
        #self.count = 0
        #self.pop_size  = pop_size
        #self.algorithm = kwargs["algorithm"]

    def equal_fitness(self, a, b):
        if a.X[0].fitness == 0 or b.X[0].fitness == 0:
            return False
        else:
            return a.X[0].fitness == b.X[0].fitness

    def is_equal(self, a, b):
        #elif self.algo == "rigaa" and self.problem == "vehicle":
        #threshold = 0.2
        if self.problem == "robot":
            threshold = 0.2
        else:
            threshold = 0.2
        state1 = a.X[0].states
        state2 = b.X[0].states
        #eq_fit  = self.equal_fitness(a, b)

        # Calculating the novelty of the two states.
        novelty = abs(a.X[0].calculate_novelty(state1, state2))
        if (novelty  < threshold):
            #nov_value = round(novelty, 2)
            #id  = str(int(time.time())) + "_" + str(nov_value)
            #a.X[0].build_image(a.X[0].states, "duplicates\\" + id + "_a.png")
            #b.X[0].build_image(b.X[0].states, "duplicates\\" + id + "_b.png")
            log.debug("Duplicate %s and %s found", a.X[0], b.X[0])
        return (novelty < threshold)
