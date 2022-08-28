#!/usr/bin/python3

import logging
import numpy as np
import random

from src.core.utils import (
    Infrastructure,
    Pipeline,
    ParetoTools,
    Objectives,
    WriteObjectivesToFileObserver,
    Latency,
)
from src.core.utils import (
    StoppingByNonDominance,
    StoppingByTotalDominance,
    StoppingByFullPareto,
    Constraints,
)
from src.core.mutation import PowerOffMutation

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.nsgaiii import (
    NSGAIII,
    UniformReferenceDirectionFactory,
)
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from jmetal.lab.visualization import Plot
from jmetal.operator import BitFlipMutation, SPXCrossover
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver, BasicObserver
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    get_non_dominated_solutions,
)
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.comparator import StrengthAndKNNDistanceComparator, DominanceComparator


class TravelingModel(BinaryProblem):
    def __init__(self, file_infrastructure, file_latencies, input_pipeline):
        super(TravelingModel, self).__init__()

        self.infra = Infrastructure(file_infrastructure).load()
        self.pipe = Pipeline(input_pipeline).load()
        # self.ld = np.loadtxt("src/resources/latencies.csv", dtype=float)
        self.ld = Latency(file_location=file_latencies).load()
        # self.ld = np.loadtxt(file_latencies, dtype=float)
        # s0, s1 = self.ld.shape
        # self.ld = np.reshape(self.ld, (s0, 1, s1))

        # number of models
        # self.number_of_models = self.pipe.shape[1]
        self.number_of_models = self.pipe.shape[0]
        self.number_of_objectives = 4
        # number of devices
        self.number_of_devices = len(self.infra.index)

        # number of constraints
        self.number_of_constraints = 4

        # self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        # self.obj_labels = ['consumption', 'resillience', 'performance']
        # self.obj_directions = [self.MAXIMIZE]
        # self.obj_labels = ['performance']

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        self.objectives = Objectives()
        solution.objectives[0] = -1 * self.objectives.get_resilience(
            self.infra, solution
        )
        solution.objectives[1] = -1 * self.objectives.get_performance(
            self.pipe, self.infra, solution
        )
        solution.objectives[2] = self.objectives.get_consumption(
            self.pipe, self.infra, solution
        )
        solution.objectives[3] = -1 * self.objectives.get_network_performance(
            ld=self.ld, pipe=self.pipe, infra=self.infra, solution=solution
        )

        self.__evaluate_constraints(solution)

        return solution

    def __evaluate_constraints(self, solution: BinarySolution) -> None:
        # constraints = [0.0 for _ in range(self.number_of_constraints)]
        constraints = []

        c = Constraints(solution, self.infra, self.pipe)
        """ 
        do not exceed total CPU per device
        """
        # x = s.transpose() * self.pipe.cpus.to_numpy()
        # sum_rows = np.sum(x, axis=1)
        # thread_count = self.infra.thread_count.to_numpy()
        # constraints.append(0 if not (thread_count < sum_rows).any() else -1)
        constraints.append(c.cpu_constraint())

        """
        do not exceed total RAM per device
        """
        # x = s.transpose() * self.pipe.memory.to_numpy()
        # sum_rows = np.sum(x, axis=1)
        # memory = self.infra.memory.to_numpy()
        # constraints.append(0 if not (memory < sum_rows).any() else -1)
        constraints.append(c.ram_constraint())

        """ 
        each model should be deployed in at least one device
        """
        # sum_rows = np.sum(s, axis=1)
        # the sum of all rows should be bigger or equal to one
        # constraints.append(0 if not (sum_rows < 1).any() else -1)
        constraints.append(c.deployment_constraint())

        """
        do not exceed total GPU per device
        """

        """ 
        enforce privacy constraints
        """
        constraints.append(c.privacy_constraint())

        solution.constraints = constraints

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(
            number_of_variables=self.number_of_models,
            number_of_objectives=self.number_of_objectives,
        )

        """
        sol = []
        for m in range(self.number_of_models):
            s = [False]*(self.number_of_devices-1) + [True]
            random.shuffle(s)
            sol.append(s)

        new_solution.variables = list(map(list, zip(*sol)))
        """
        for i in range(self.number_of_models):
            # new_solution.variables[i] = [
            #    True if random.random() > 0.5 else False
            #    for _ in range(self.number_of_devices)
            # ]
            new_solution.variables[i] = [
                True if random.random() < (2 / self.number_of_devices) else False
                for _ in range(self.number_of_devices)
            ]

        return new_solution

    def get_name(self) -> str:
        return "TravelingModel"


class Optimizer:
    def __init__(
        self,
        termination_criterion,
        file_infrastructure,
        file_latencies,
        input_pipeline,
        population_size=100,
        observer=None,
    ):
        self.termination_criterion = termination_criterion
        self.file_infrastructure = file_infrastructure
        self.file_latencies = file_latencies
        self.input_pipeline = input_pipeline
        self.observer = observer
        self.population_size = population_size

    def run(self):
        self.problem = TravelingModel(
            file_infrastructure=self.file_infrastructure,
            file_latencies=self.file_latencies,
            input_pipeline=self.input_pipeline,
        )

        self.algorithm = NSGAIII(
            problem=self.problem,
            population_size=self.population_size,
            # offspring_population_size=self.population_size,
            # reference_directions=UniformReferenceDirectionFactory(4, n_points=92),
            reference_directions=UniformReferenceDirectionFactory(4, n_points=92),
            # mutation=BitFlipMutation(probability=1.0 / self.problem.number_of_devices),
            mutation=PowerOffMutation(probability=1.0 / self.problem.number_of_devices),
            crossover=SPXCrossover(probability=1.0),
            termination_criterion=self.termination_criterion,
            # termination_criterion=StoppingByEvaluations(max_evaluations=self.max_evaluations),
            # termination_criterion=StoppingByNonDominance(idle_evaluations=200),
            # termination_criterion=StoppingByTotalDominance(idle_evaluations=50),
            # termination_criterion=StoppingByFullPareto(self.population_size),
            # dominance_comparator=StrengthAndKNNDistanceComparator()
            dominance_comparator=DominanceComparator(),
        )

        # if (self.interactive_plot):
        #    self.algorithm.observable.register(observer=ProgressBarObserver(max=self.max_evaluations))
        #    self.algorithm.observable.register(observer=VisualizerObserver(reference_front=self.problem.reference_front, display_frequency=100))
        #    basic = BasicObserver(frequency=1.0)
        #    self.algorithm.observable.register(observer=basic)

        if self.observer:
            self.algorithm.observable.register(observer=self.observer)

        self.algorithm.run()

        # self.front = get_non_dominated_solutions(self.algorithm.get_result())
        self.front = self.algorithm.get_result()
        # logging.warning(self.front[0].objectives)
        # logging.warning(self.front[1].objectives)
        # logging.warning(self.front[2].objectives)
        # logging.warning(self.front[3].objectives)

        # for v in self.front[0].variables:
        #    logging.warning(v)
        # logging.warning(self.front[0].objectives)
        # logging.info(f'Algorithm: ${self.algorithm.get_name()}')
        # logging.info(f'Problem: ${self.problem.get_name()}')
        logging.info(f"Computing time: ${self.algorithm.total_computing_time}")

    def get_front(self):
        return self.front

    def plot(self):
        # Plot front
        plot_front = Plot(
            title="Pareto front approximation. Problem: " + self.problem.get_name(),
            reference_front=self.problem.reference_front,
            axis_labels=self.problem.obj_labels,
        )
        plot_front.plot(
            self.front, label=self.algorithm.label, filename=self.algorithm.get_name()
        )

        # print variables and fitnesses
        print_function_values_to_file(self.front, "FUN." + self.algorithm.label)
        print_variables_to_file(self.front, "VAR." + self.algorithm.label)
