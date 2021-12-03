#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import os
import time
from src.core.optimizer import Optimizer
from src.core.utils import StoppingByNonDominance, StoppingByTotalDominance, StoppingByFullPareto, WriteObjectivesToFileObserver, ParetoTools
from jmetal.util.termination_criterion import StoppingByEvaluations, StoppingByTime

LOGGER = logging.getLogger('optimizer')

def generate_barchar(file_infrastructure):
    total_times = []
    pipelines = ['pipeline_5.yml', 'pipeline_10.yml', 'pipeline_20.yml', 'pipeline_40.yml', 'pipeline_80.yml']
    for p in pipelines:
        file_pipeline = f"src/resources/{p}"
        with open(file_pipeline, 'r') as input_data_file:
            input_pipeline = input_data_file.read()
        pipe_time = []
        for i in range(100):
            start_time = time.time()
            LOGGER.info(f"Executing iteration {i} of {file_pipeline}.")
            Optimizer(
                file_infrastructure=file_infrastructure,
                input_pipeline=input_pipeline,
                termination_criterion=StoppingByTotalDominance(idle_evaluations=40)
            ).run()
            end_time = time.time()
            pipe_time.append(end_time-start_time)
        total_times.append(pipe_time)

    filename = '/tmp/times'
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(total_times)

def generate_pareto(file_infrastructure):
    file_pipeline = f"src/resources/pipeline_40.yml"
    population_size = 1000
    with open(file_pipeline, 'r') as input_data_file:
        input_pipeline = input_data_file.read()
    o = Optimizer(
        file_infrastructure=file_infrastructure,
        input_pipeline=input_pipeline,
        termination_criterion=StoppingByFullPareto(offspring_size=population_size),
        population_size=population_size
    )
    o.run()
    pt = ParetoTools(o.get_front())
    pt.save()

def generate_fitnesses(file_infrastructure):
    file_pipeline = f"src/resources/pipeline_20.yml"
    with open(file_pipeline, 'r') as input_data_file:
        input_pipeline = input_data_file.read()
    Optimizer(
        file_infrastructure=file_infrastructure,
        input_pipeline=input_pipeline,
        #termination_criterion=StoppingByTotalDominance(idle_evaluations=100),
        termination_criterion=StoppingByEvaluations(max_evaluations=40*2000),
        observer=WriteObjectivesToFileObserver(),
        population_size=40
    ).run()

def generate_memory(file_infrastructure, number_of_models):
    file_pipeline = f"src/resources/pipeline_{number_of_models}.yml"
    with open(file_pipeline, 'r') as input_data_file:
        input_pipeline = input_data_file.read()
    Optimizer(
        file_infrastructure=file_infrastructure,
        input_pipeline=input_pipeline,
        termination_criterion=StoppingByTime(max_seconds=300)
    ).run()

def main():
    text = 'This application optimizes PADL defined analytic models in heterogeneous infrastructures.'
    parser = argparse.ArgumentParser(description=text)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-p', '--pareto', action='store_true',
                          help='Generate pareto front.', required=False)
    required.add_argument('-b', '--barchar', action='store_true',
                          help='Generate barchar metrics', required=False)
    required.add_argument('-f', '--fitnesses', action='store_true',
                          help='Generate barchar metrics', required=False)
    required.add_argument('-m', '--memory', type=str, default=None,
                          help='Indicate number of moels (e.g., 5, 10, 20, 40, 80)', required=False)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')

    file_infrastructure='src/resources/infrastructure.csv'

    if args.barchar:
        generate_barchar(file_infrastructure)

    if args.pareto:
        generate_pareto(file_infrastructure)

    if args.fitnesses:
        generate_fitnesses(file_infrastructure)

    if args.memory:
        generate_memory(file_infrastructure, args.memory)

if __name__ == '__main__':
    main()
