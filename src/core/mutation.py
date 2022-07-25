import random

from jmetal.core.operator import Mutation
from jmetal.core.solution import BinarySolution
from jmetal.util.ckecking import Check

class PowerOffMutation(Mutation[BinarySolution]):

    def __init__(self, probability: float):
        super(PowerOffMutation, self).__init__(probability=probability)

    def execute(self, solution: BinarySolution) -> BinarySolution:
        Check.that(type(solution) is BinarySolution, "Solution type invalid")

        number_of_models, number_of_devices = solution.shape

        for i in range(number_of_devices):
            rand = random.random()
            if rand <= self.probability:
                for j in range(number_of_models):
                    solution.variables[j][i] = False

        return solution

    def get_name(self):
        return 'Power Off Mutation'
