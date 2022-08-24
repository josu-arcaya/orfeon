#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import random
import yaml
import numpy as np
from numpy.random import choice

total_location = ["IE", "ES", "US"]


class Model:
    def __init__(
        self,
        model_name: str,
        cpus: int,
        memory: float,
        location: str,
        privacy_type: int,
        link: int,
    ):
        self.model = model_name
        self.resources = {"cpus": cpus, "memory": memory}
        self.privacy = {"location": location, "type": privacy_type}
        self.link = link


class PADL:
    def __init__(self, version: str):
        self.version = version
        self.pipeline = []


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )

    text = "This application generates PADL defined analytic models."
    parser = argparse.ArgumentParser(description=text)
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-m",
        "--models",
        type=str,
        help="The number of models to be generated.",
        required=True,
    )

    args = parser.parse_args()
    number_of_models = int(args.models)

    total_cpus = 0
    total_memory = 0

    p = PADL("1.0")
    random.seed(1)
    np.random.seed(1)
    for i in range(number_of_models):
        model_size = choice([0, 1, 2], 1, p=[0.8, 0.15, 0.05])[0]
        # model_size = random.random()
        if model_size == 0:
            # small model
            cpus = random.randint(1, 7)
            memory = random.randint(1, 16)
        elif model_size == 1:
            # medium model
            cpus = random.randint(7, 28)
            memory = random.randint(16, 64)
        elif model_size == 2:
            # large model
            cpus = random.randint(28, 32)
            memory = random.randint(64, 251)
        m = Model(
            model_name=f"m{i}",
            cpus=cpus,
            memory=memory,
            location=random.choice(total_location),
            privacy_type=random.randint(0, 2),
            link=random.randint(0, number_of_models - 1),
        )
        p.pipeline.append(m.__dict__)

        total_cpus += cpus
        total_memory += memory

    with open(r"/tmp/pipeline.yaml", "w") as file:
        documents = yaml.dump(p, file)

    logging.info(f"{number_of_models} models generated")
    logging.info(f"{total_cpus} total cpus")
    logging.info(f"{total_memory} total memory")


if __name__ == "__main__":
    main()
