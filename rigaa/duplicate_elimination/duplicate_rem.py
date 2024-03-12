"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for duplicate elimination
"""

import logging as log
from pymoo.core.duplicate import ElementwiseDuplicateElimination
import time

# It's a duplicate elimination that compares the states of the two elements


class DuplicateElimination(ElementwiseDuplicateElimination):
    """
    A class to eliminate duplicates in the population.
    """

    def __init__(self, algo, problem):
        super().__init__()
        self.algo = algo
        self.first_gen = True
        self.problem = problem

    def equal_fitness(self, a, b):
        if a.X[0].fitness == 0 or b.X[0].fitness == 0:
            return False
        else:
            return a.X[0].fitness == b.X[0].fitness

    def is_equal(self, a, b):

        threshold = 0.3
        state1 = a.X[0].states
        state2 = b.X[0].states

        # Calculating the novelty of the two states.
        novelty = abs(a.X[0].calculate_novelty(state1, state2))
        if novelty < threshold:
            log.debug("Duplicate %s and %s found", a.X[0], b.X[0])
        return novelty < threshold
