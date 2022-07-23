import unittest

# from unittest.mock import patch

import logging
import numpy as np
from jmetal.core.problem import BinaryProblem
from jmetal.util.termination_criterion import StoppingByEvaluations

from src.core.optimizer import TravelingModel, Optimizer, Constraints
from src.core.utils import Infrastructure, Pipeline, WriteObjectivesToFileObserver, Constraints


class TestConstraints(unittest.TestCase):
    def setUp(self):
        file_infrastructure = "tests/resources/infrastructure.csv"
        file_latencies = "tests/resources/latencies.csv"
        pipeline_location = "tests/resources/pipeline.yaml"
        with open(pipeline_location, "r") as input_data_file:
            input_pipeline = input_data_file.read()

        self.o = Optimizer(
            termination_criterion=StoppingByEvaluations(max_evaluations=10000),
            file_infrastructure=file_infrastructure,
            file_latencies=file_latencies,
            input_pipeline=input_pipeline,
        )
        # observer=WriteObjectivesToFileObserver())
        self.o.run()
        self.front = self.o.get_front()
        # load pipeline
        self.pipe = Pipeline(input_pipeline).load()
        # load infrastructure
        self.infra = Infrastructure(file_infrastructure).load()

    # each model should be deployed in at least one device
    def test_deployment(self):
        for sol in self.front:
            s = np.asfarray(sol.variables, dtype=np.bool)
            z = np.sum(s, axis=1) - 1
            self.assertFalse((z < 0).any())

            c = Constraints(sol, self.infra, self.pipe)
            self.assertEqual(c.deployment_constraint(), 0)

    # do not exceed total CPU per device
    def test_cpu(self):
        for sol in self.front:
            s = np.asfarray(sol.variables, dtype=np.bool)
            # i = s.transpose()*self.p[0]
            i = s.transpose() * self.pipe.cpus.to_numpy()
            j = np.sum(i, axis=1)
            k = self.infra.thread_count.to_numpy()
            self.assertFalse((k < j).any())

            c = Constraints(sol, self.infra, self.pipe)
            self.assertEqual(c.cpu_constraint(), 0)

    # do not exceed total memory per device
    def test_memory(self):
        for sol in self.front:
            s = np.asfarray(sol.variables, dtype=np.bool)
            i = s.transpose() * self.pipe.memory.to_numpy()
            j = np.sum(i, axis=1)
            k = self.infra.memory.to_numpy()
            self.assertFalse((k < j).any())

            c = Constraints(sol, self.infra, self.pipe)
            self.assertEqual(c.ram_constraint(), 0)

    def test_privacy(self):
        # for sol in self.front:
        c = Constraints(self.front[0], self.infra, self.pipe)
        self.assertTrue(c.privacy_constraint() >= 0)


if __name__ == "__main__":
    unittest.main()
