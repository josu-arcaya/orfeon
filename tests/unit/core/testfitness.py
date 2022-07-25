import unittest

# from unittest.mock import patch

import logging
import numpy as np
from jmetal.core.problem import BinaryProblem

# from sklearn import preprocessing

from src.core.optimizer import TravelingModel, Optimizer
from jmetal.core.solution import BinarySolution
from src.core.utils import Infrastructure, Pipeline, Objectives, Latency


class TestFitness(unittest.TestCase):

    #    @patch("jmetal.core.problem.BinarySolution")
    #    def test_consumption_basic(self, solution_constructor):
    #        o = TravelingModel(100,100)
    #        new_solution = solution_constructor()
    #        self.assertEqual(o.get_consumption(new_solution), 5)

    #    @patch("jmetal.core.problem.BinarySolution")
    #    def test_performance_basic(self, solution_constructor):
    #        o = TravelingModel(100,100)
    #        new_solution = solution_constructor()
    #        self.assertEqual(o.get_performance(new_solution), 10)
    def setUp(self):
        file_infrastructure = "tests/resources/infrastructure.csv"
        file_latencies = "tests/resources/latencies.csv"
        pipeline_location = "tests/resources/pipeline.yaml"
        with open(pipeline_location, "r") as input_data_file:
            input_pipeline = input_data_file.read()
        """
        self.o = Optimizer(max_evaluations=2000, file_infrastructure=file_infrastructure, input_pipeline=input_pipeline, interactive_plot = False)
        self.o.run()
        self.front = self.o.get_front()
        """
        # load pipeline
        self.pipe = Pipeline(input_pipeline).load()
        # load infrastructure
        self.infra = Infrastructure(file_infrastructure).load()
        # load latencies
        self.ld = Latency(file_location=file_latencies).load()

        self.problem = TravelingModel(
            file_infrastructure=file_infrastructure,
            file_latencies=file_latencies,
            input_pipeline=input_pipeline,
        )

        # number of models
        # number_of_models = self.p.shape[1]
        number_of_models = self.pipe.shape[0]
        # number of devices
        number_of_devices = len(self.infra.index)

        # all ones solution
        self.sol_ones = BinarySolution(
            number_of_variables=number_of_models, number_of_objectives=1
        )
        for i in range(len(self.sol_ones.variables)):
            self.sol_ones.variables[i] = [True for _ in range(number_of_devices)]

        # all zeros solution
        self.sol_zeros = BinarySolution(
            number_of_variables=number_of_models, number_of_objectives=1
        )
        for i in range(len(self.sol_zeros.variables)):
            self.sol_zeros.variables[i] = [False for _ in range(number_of_devices)]

    def test_performance(self):
        performance = Objectives().get_performance(self.pipe, self.infra, self.sol_ones)
        self.assertGreater(performance, 0)

        performance = Objectives().get_performance(
            self.pipe, self.infra, self.sol_zeros
        )
        # self.assertGreaterEqual(performance, 0)

    def test_resilience(self):
        resilience = Objectives().get_resilience(self.infra, self.sol_ones)
        self.assertGreater(resilience, 0)

        resilience = Objectives().get_resilience(self.infra, self.sol_zeros)
        self.assertEqual(resilience, 0)

        my_sol = self.sol_zeros
        my_sol.variables[0][11] = True
        my_sol.variables[1][11] = True
        my_sol.variables[2][11] = True
        my_sol.variables[3][11] = True
        resilience = Objectives().get_resilience(self.infra, my_sol)
        self.assertEqual(resilience, 1)

    def test_consumption(self):
        consumption = Objectives().get_consumption(self.pipe, self.infra, self.sol_ones)
        self.assertGreater(consumption, 0)

        consumption = Objectives().get_consumption(
            self.pipe, self.infra, self.sol_zeros
        )
        self.assertGreater(consumption, 0)

    def test_latency(self):

        network_performance = Objectives().get_network_performance(
            self.ld, self.pipe, self.infra, self.sol_zeros
        )
        #self.assertEqual(network_performance, 0)

        network_performance = Objectives().get_network_performance(
            self.ld, self.pipe, self.infra, self.sol_ones
        )
        # self.assertEqual(int(latency), 44026)
        self.assertGreater(network_performance, 0)

    """
    def setUp(self):
        infrastructure_location = 'src/test/resources/infrastructure.csv'
        pipeline_location = 'src/test/resources/pipeline.yaml'
        with open(pipeline_location, 'r') as input_data_file:
            self.input_pipeline = input_data_file.read()
        self.o = TravelingModel(infrastructure_location, self.input_pipeline)
        self.sol_random = self.o.create_solution()
        self.sol_zeros = self.o.create_solution()
        for s in range(len(self.sol_zeros.variables)):
            self.sol_zeros.variables[s] = \
                [False for _ in range(len(self.sol_zeros.variables))]
        self.sol_ones = self.o.create_solution()
        for s in range(len(self.sol_ones.variables)):
            self.sol_ones.variables[s] = \
                [True for _ in range(len(self.sol_ones.variables))]
    """

    """
    def test_create_solution(self):
        o = TravelingModel(self.input_data, self.file_location2, self.input_pipeline)
        new_solution = o.create_solution()
        self.assertTrue(len(new_solution.variables) >= 0)
        #s = np.asfarray(new_solution.variables, dtype=np.bool)
        #self.assertEqual(s.sum(axis=0).prod(), 1)

    def test_consumption(self):
        self.assertTrue(self.o.get_consumption(self.sol_random) >= 0)
        # check sol_ones
        consumption = self.o.get_consumption(self.sol_ones)
        self.assertEqual(round(consumption), 1508)
        # check sol_zeros
        consumption = self.o.get_consumption(self.sol_zeros)
        self.assertEqual(round(consumption), 276)

    def test_performance(self):
        self.assertTrue(self.o.get_performance(self.sol_random) >= 0)
        # check sol_ones
        performance = self.o.get_performance(self.sol_ones)
        self.assertEqual(round(performance), 23435)
        # check sol_zeros
        performance = self.o.get_performance(self.sol_zeros)
        self.assertEqual(round(performance), 0)

    def test_availability(self):
        self.assertTrue(self.o.get_availability(self.sol_random) >= 0)
        # check sol_ones
        availability = self.o.get_availability(self.sol_ones)
        self.assertEqual(round(availability), 324)
        # check sol_zeros
        availability = self.o.get_availability(self.sol_zeros)
        self.assertEqual(round(availability), 0)

    def test_evaluate(self):
        self.o.evaluate(self.sol_random)

    def test_name(self):
        self.assertEqual(self.o.get_name(), 'TravelingModel')
    """

    """
    def setUp(self):
        file_location2 = 'src/test/resources/infrastructure.csv'
        file_location = 'src/test/resources/infrastructure.json'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        pipeline_location = 'src/test/resources/pipeline.yaml'
        with open(pipeline_location, 'r') as input_data_file:
            input_pipeline = input_data_file.read()
        self.o = Optimizer(2000, file_location2, input_data, input_pipeline)
        self.o.run()
        self.front = self.o.get_front()
        self.infra = Infrastructure(input_data).load()
    
    def test_cpu_constraint(self):
        for sol in self.front:
            for i, dev in enumerate(sol.variables):
                #logging.warning("%s, %s, %s" % (i, sum(dev), self.infra.devices[i].resources.threads))
                self.assertTrue(sum(dev) <= self.infra.devices[i].resources.threads+4)

    def test_memory_constraint(self):
        for sol in self.front:
            for i, dev in enumerate(sol.variables):
                #logging.warning("%s, %s, %s" % (i, sum(dev), self.infra.devices[i].resources.memory))
                self.assertTrue(sum(dev) <= self.infra.devices[i].resources.memory+4)

    def test_deploy_constraint(self):
        for sol in self.front:
            x = np.asfarray(sol.variables, dtype=np.bool)
            columns = x.shape[1]
            for c in range(columns):
                #logging.warning(sum(x[:,c]))
                self.assertTrue(sum(x[:,c]) >= 1)

    def test_network_constraint(self):
        for sol in self.front:
            for dev in sol.variables:
                #logging.warning(sum(dev))
                self.assertTrue(max(dev) <= 10)
    
    def test_plot(self):
        self.o.plot()
    """


if __name__ == "__main__":
    unittest.main()
