import random

from jmetal.core.operator import Mutation
from jmetal.core.solution import BinarySolution
from jmetal.util.ckecking import Check


class PowerOffMutation(Mutation[BinarySolution]):
    def __init__(self, probability: float):
        super(PowerOffMutation, self).__init__(probability=probability)

    def execute(self, solution: BinarySolution) -> BinarySolution:
        Check.that(type(solution) is BinarySolution, "Solution type invalid")

        number_of_objectives = solution.number_of_objectives
        number_of_models = solution.number_of_variables
        number_of_devices = len(solution.variables[0])

        rand = random.random()

        if rand < (1 / (number_of_objectives * 2)):
            for i in range(number_of_devices):
                rand = random.random()
                if rand <= self.probability:
                    for j in range(number_of_models):
                        rand = random.random()
                        if rand < 0.5:
                            solution.variables[j][i] = False
        else:
            for i in range(number_of_models):
                for j in range(number_of_devices):
                    rand = random.random()
                    if rand <= self.probability:
                        solution.variables[i][j] = (
                            True if solution.variables[i][j] is False else False
                        )
        
        for i in range(number_of_models):
            if sum(solution.variables[i]) == 0:
                for j in range(number_of_devices):
                    rand = random.random()
                    if rand <= self.probability:
                        solution.variables[i][j] = True

        return solution

    def get_name(self):
        return "Power Off Mutation"
