import unittest
import logging

from src.core.utils import Infrastructure, Pipeline

class TestInfrastructure(unittest.TestCase):

    def test_load_infrastructure_1(self):
        file_location = 'tests/resources/infrastructure.csv'
        infra = Infrastructure(file_location).load()
        self.assertTrue( len(infra.index)>0 )
        self.assertEqual( infra.hostname[0], 'WKM0092')
        self.assertEqual( infra.core_count[5], 7)

class TestPipeline(unittest.TestCase):

    def test_load_pipeline(self):
        file_location = 'tests/resources/pipeline.yaml'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        p = Pipeline(input_data).load()
        # assert cpus in device 0
        #self.assertEqual(p[0][0], 4)
        self.assertEqual(p.cpus[0], 4)
        # assert memory in device 2
        #self.assertEqual(p[1][2], 8)
        self.assertEqual(p.memory[2], 8)
