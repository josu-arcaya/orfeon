#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import os
import time
from src.core.optimizer import Optimizer
from src.core.utils import (
    StoppingByNonDominance,
    StoppingByTotalDominance,
    StoppingByFullPareto,
    WriteObjectivesToFileObserver,
    ParetoTools,
    Evaluate,
)
from jmetal.util.termination_criterion import StoppingByEvaluations, StoppingByTime

LOGGER = logging.getLogger("optimizer")

def evaluate_solution(file_solution: str):
    e = Evaluate(file_solution=file_solution)
    print(f"Cost = {e.cost()}")
    print(f"Model Performance = {e.model_performance()}")
    print(f"Resilience = {e.resilience()}")
    print(f"Network Performance = {e.network_performance()}")

def generate_barchar(file_infrastructure, file_latencies):
    total_times = []
    pipelines = [
        "pipeline_5.yml",
        "pipeline_10.yml",
        "pipeline_20.yml",
        "pipeline_40.yml",
        "pipeline_80.yml",
    ]
    for p in pipelines:
        file_pipeline = f"src/resources/{p}"
        with open(file_pipeline, "r") as input_data_file:
            input_pipeline = input_data_file.read()
        pipe_time = []
        # do it 100 times
        for i in range(2):
            start_time = time.time()
            LOGGER.info(f"Executing iteration {i} of {file_pipeline}.")
            Optimizer(
                file_infrastructure=file_infrastructure,
                file_latencies=file_latencies,
                input_pipeline=input_pipeline,
                termination_criterion=StoppingByTotalDominance(idle_evaluations=40),
            ).run()
            end_time = time.time()
            pipe_time.append(end_time - start_time)
        total_times.append(pipe_time)

    filename = "/tmp/times"
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, "w") as out_file:
        writer = csv.writer(out_file)
        writer.writerows(total_times)


def generate_pareto(file_infrastructure, file_latencies):
    file_pipeline = f"src/resources/pipeline_40.yml"
    population_size = 200
    with open(file_pipeline, "r") as input_data_file:
        input_pipeline = input_data_file.read()
    o = Optimizer(
        file_infrastructure=file_infrastructure,
        file_latencies=file_latencies,
        input_pipeline=input_pipeline,
        # termination_criterion=StoppingByFullPareto(offspring_size=population_size),
        termination_criterion=StoppingByTime(max_seconds=180),
        population_size=population_size,
    )
    o.run()
    pt = ParetoTools(o.get_front())
    pt.save()


def generate_fitnesses(file_infrastructure, file_latencies):
    file_pipeline = f"src/resources/pipeline_40.yml"
    population_size = 80
    with open(file_pipeline, "r") as input_data_file:
        input_pipeline = input_data_file.read()
    Optimizer(
        file_infrastructure=file_infrastructure,
        file_latencies=file_latencies,
        input_pipeline=input_pipeline,
        # termination_criterion=StoppingByTotalDominance(idle_evaluations=100),
        # termination_criterion=StoppingByEvaluations(max_evaluations=40 * 2000),
        termination_criterion=StoppingByEvaluations(
            max_evaluations=population_size * 2000
        ),
        observer=WriteObjectivesToFileObserver(),
        population_size=population_size,
    ).run()


def generate_memory(file_infrastructure, file_latencies, number_of_models):
    file_pipeline = f"src/resources/pipeline_{number_of_models}.yml"
    with open(file_pipeline, "r") as input_data_file:
        input_pipeline = input_data_file.read()
    Optimizer(
        file_infrastructure=file_infrastructure,
        file_latencies=file_latencies,
        input_pipeline=input_pipeline,
        termination_criterion=StoppingByTime(max_seconds=300),
    ).run()


def main():
    text = "This application optimizes PADL defined analytic models in heterogeneous infrastructures."
    parser = argparse.ArgumentParser(description=text)
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-p",
        "--pareto",
        action="store_true",
        help="Generate pareto front.",
        required=False,
    )
    required.add_argument(
        "-t",
        "--times",
        action="store_true",
        help="Generate time metrics",
        required=False,
    )
    required.add_argument(
        "-f",
        "--fitnesses",
        action="store_true",
        help="Generate fitnesses metrics",
        required=False,
    )
    required.add_argument(
        "-m",
        "--memory",
        type=str,
        default=None,
        help="Indicate number of models (e.g., 5, 10, 20, 40, 80)",
        required=False,
    )
    required.add_argument(
        "-e",
        "--evaluate",
        type=str,
        default=None,
        help="Indicate a csv solution",
        required=False,
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )

    file_infrastructure = "src/resources/infrastructure.csv"
    file_latencies = "src/resources/latencies.csv"

    if args.times:
        generate_barchar(
            file_infrastructure=file_infrastructure, file_latencies=file_latencies
        )

    if args.pareto:
        generate_pareto(
            file_infrastructure=file_infrastructure, file_latencies=file_latencies
        )

    if args.fitnesses:
        generate_fitnesses(
            file_infrastructure=file_infrastructure, file_latencies=file_latencies
        )

    if args.memory:
        generate_memory(
            file_infrastructure=file_infrastructure,
            file_latencies=file_latencies,
            number_of_models=args.memory,
        )

    if args.evaluate:
        evaluate_solution(file_solution=args.evaluate)


if __name__ == "__main__":
    main()
