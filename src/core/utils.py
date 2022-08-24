from email.mime import base
import json
import logging
import numpy as np
import os
import pandas as pd
import pycountry
import pycountry_convert as pc
import yaml
import datetime

from collections import namedtuple
from math import ceil

from jmetal.core.observer import Observer
from jmetal.core.solution import BinarySolution
from jmetal.util.comparator import DominanceComparator
from jmetal.util.termination_criterion import TerminationCriterion

continent_codes = {
    "AF": 100,
    "AN": 101,
    "AS": 102,
    "EU": 103,
    "NA": 104,
    "OC": 105,
    "SA": 106,
}

LOGGER = logging.getLogger("jmetal")


class Parser:
    def __init__(self, input_data: str):
        self.x = self.file2obj(input_data)

    def object_hook(self, d):
        return namedtuple("X", d.keys())(*d.values())

    def file2obj(self, data):
        pass


class Latency:
    def __init__(self, file_location: str):
        ld = np.loadtxt(file_location, dtype=float)
        x, y = ld.shape
        # normalize
        ld /= ld.max() * (x * y)

        self.ld = np.reshape(ld, (x, 1, y))

    def load(self):
        return self.ld


class Infrastructure:
    def __init__(self, file_location: str):
        self.infrastructure = pd.read_csv(file_location)

        x = lambda txt: np.fromstring(txt[1:-1], sep=",")
        self.infrastructure.consumption = self.infrastructure.consumption.apply(x)

        # normalize consumption
        my_max = self.infrastructure.consumption.apply(max).max()
        my_min = self.infrastructure.consumption.apply(min).min()
        self.infrastructure.consumption = (self.infrastructure.consumption - my_min) / (
            my_max - my_min
        )
        # self.infrastructure.consumption /= ceil(self.infrastructure.consumption.apply(max).sum())

        self.infrastructure.parallelization = self.infrastructure.parallelization.apply(
            x
        )

        y = lambda row: int(
            pycountry.countries.get(alpha_2=row["country_code"]).numeric
        )
        self.infrastructure["country"] = self.infrastructure.apply(y, axis=1)

        # convert alpha2 country_code to continent name
        z = lambda row: continent_codes[
            pc.country_alpha2_to_continent_code(row["country_code"])
        ]
        self.infrastructure["continent"] = self.infrastructure.apply(z, axis=1)

        # normalize bandwidth
        self.infrastructure.bandwidth = (
            self.infrastructure.bandwidth / self.infrastructure.bandwidth.max()
        )

        # normalize performance
        self.infrastructure.performance = (
            self.infrastructure.performance - self.infrastructure.performance.min()
        ) / (
            self.infrastructure.performance.max()
            - self.infrastructure.performance.min()
        )

        # normalize resilience
        self.infrastructure.resillience = (
            self.infrastructure.resillience - self.infrastructure.resillience.min()
        ) / (
            self.infrastructure.resillience.max()
            - self.infrastructure.resillience.min()
        )

    def load(self):
        return self.infrastructure


class Pipeline(Parser):
    def file2obj(self, data):
        y = yaml.load(data, Loader=yaml.FullLoader)
        return json.loads(json.dumps(y), object_hook=self.object_hook)

    def load(self):

        columns = ["cpus", "memory", "country", "privacy_type", "continent", "link"]
        data = {
            columns[0]: [],
            columns[1]: [],
            columns[2]: [],
            columns[3]: [],
            columns[4]: [],
            columns[5]: [],
        }
        for i in range(len(self.x.pipeline)):
            data[columns[0]].append(float(self.x.pipeline[i].resources.cpus))
            data[columns[1]].append(float(self.x.pipeline[i].resources.memory))
            # get numeric from alpha_2 country code
            country_numeric = pycountry.countries.get(
                alpha_2=self.x.pipeline[i].privacy.location
            ).numeric
            data[columns[2]].append(int(country_numeric))
            data[columns[3]].append(int(self.x.pipeline[i].privacy.type))
            # get continent from alpha_2 country code
            data[columns[4]].append(
                continent_codes[
                    pc.country_alpha2_to_continent_code(
                        self.x.pipeline[i].privacy.location
                    )
                ]
            )
            data[columns[5]].append(int(self.x.pipeline[i].link))
        b = pd.DataFrame(data=data, columns=columns)

        return b


class ParetoTools:
    def __init__(self, front):
        self.front = front

    def save(self):
        filename = f"/tmp/pareto"
        with open(filename, "w") as pareto_file:
            for s in self.front:
                pareto_file.write(
                    f"{abs(s.objectives[0])},{abs(s.objectives[1])},{s.objectives[2]},{abs(s.objectives[3])}\n"
                )


class Objectives:
    def get_consumption(
        self, pipe: Pipeline, infra: Infrastructure, solution: BinarySolution
    ) -> int:
        s = np.asfarray(solution.variables, dtype=np.bool)

        threads_required = s.transpose().dot(pipe.cpus.to_numpy()).astype(int)

        consumption = [
            m[min(n, len(m) - 1)] for m, n in zip(infra.consumption, threads_required)
        ]

        _, number_of_devices = s.shape
        return sum(consumption) / number_of_devices * 4

    def get_performance(
        self, pipe: Pipeline, infra: Infrastructure, solution: BinarySolution
    ) -> int:
        s = np.asfarray(solution.variables, dtype=np.bool)

        base_performance = s * infra.performance.to_numpy()

        threads_required = s.transpose().dot(pipe.cpus.to_numpy()).astype(int)

        coefficient = [
            m[min(n - 1, len(m) - 1)]
            for m, n in zip(infra.parallelization, threads_required)
        ]

        number_of_models, _ = s.shape
        # return (base_performance * coefficient).max(1).sum() / number_of_models

        x = base_performance * coefficient
        x = np.ma.masked_array(x, mask=s == 0)
        return np.nanmean(x, axis=1).sum() / number_of_models

        # return (base_performance*coefficient).mean(1).sum() / number_of_models

    def get_resilience(self, infra: Infrastructure, solution: BinarySolution) -> int:
        s = np.asfarray(solution.variables, dtype=np.bool)
        if s.sum() == 0:
            return 0
        x = s * infra.resillience.to_numpy()
        x = np.ma.masked_array(x, mask=s == 0)

        number_of_models, _ = s.shape
        # return np.nanmin(x, axis=1).sum() / number_of_models
        return np.nanmean(x, axis=1).sum() / number_of_models

    # https://blog.devgenius.io/linux-how-to-measure-network-performance-c859a98abbf0
    def get_network_performance(
        self,
        ld: np.array,
        pipe: Pipeline,
        infra: Infrastructure,
        solution: BinarySolution,
    ) -> int:

        # throughput
        # latency

        s = np.asfarray(solution.variables, dtype=np.bool)

        z = s * ld.transpose()

        c = np.empty(shape=s.shape)
        for index, value in pipe.link.iteritems():
            c[index] = s[value]

        c = np.array([c])

        base_latency = z * c.transpose()
        base_bandwidth = s * infra.bandwidth.to_numpy()

        number_of_models, _ = s.shape
        # network_performance = (
        #    base_bandwidth.max(1).sum() - np.sum(base_latency)
        # ) / number_of_models

        x = np.ma.masked_array(base_bandwidth, mask=s == 0)
        x = np.ma.masked_array(x, mask=s == 0)

        network_performance = (
            np.nanmean(x, axis=1).sum() - np.sum(base_latency)
        ) / number_of_models

        return network_performance


class WriteObjectivesToFileObserver(Observer):
    def __init__(self) -> None:
        self.filename = "/tmp/fitnesses"
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def update(self, *args, **kwargs):
        evaluations = kwargs["EVALUATIONS"]
        LOGGER.info(f"Evaluations: {evaluations}")

        # solutions = np.array([s.objectives for s in kwargs['SOLUTIONS'][:10]])
        s = np.array([s.objectives for s in kwargs["SOLUTIONS"]])

        # means = s.mean(0)
        objective0 = s[np.argsort(s[:, 0])][:5].mean(0)[0]
        # objective0 = s[np.argsort(s[:,0])][0][0]
        objective1 = s[np.argsort(s[:, 1])][:5].mean(0)[1]
        # objective1 = s[np.argsort(s[:,1])][0][1]
        objective2 = s[np.argsort(s[:, 2])][:5].mean(0)[2]
        # objective2 = s[np.argsort(s[:,2])][0][2]
        objective3 = s[np.argsort(s[:, 3])][:5].mean(0)[3]

        with open(self.filename, "a") as out_file:
            # out_file.write(f"{abs(means[0])},{abs(means[1])},{abs(means[2])}\n")
            out_file.write(
                f"{abs(objective0)},{abs(objective1)},{abs(objective2)},{abs(objective3)}\n"
            )


class StoppingByNonDominance(TerminationCriterion):
    def __init__(self, idle_evaluations: int):
        super(StoppingByNonDominance, self).__init__()
        self.idle_evaluations = idle_evaluations
        self.evaluations = 0
        self.best_solution = None

    def update(self, *args, **kwargs):
        current_solution = kwargs["SOLUTIONS"][0]
        if self.best_solution is None:
            self.best_solution = current_solution
        else:
            result = DominanceComparator().compare(self.best_solution, current_solution)
            if result != 1:
                self.evaluations += 1
                LOGGER.info(f"{self.evaluations} evaluations without improvement.")
            else:
                self.evaluations = 0
            self.best_solution = current_solution

    @property
    def is_met(self):
        return self.evaluations >= self.idle_evaluations


class StoppingByTotalDominance(TerminationCriterion):
    def __init__(self, idle_evaluations: int):
        super(StoppingByTotalDominance, self).__init__()
        self.idle_evaluations = idle_evaluations
        self.evaluations = 0
        self.best_objectives = None
        self.seconds = 0.0

    def update(self, *args, **kwargs):
        self.seconds = kwargs["COMPUTING_TIME"]

        s = np.array([s.objectives for s in kwargs["SOLUTIONS"]])
        objective0 = s[np.argsort(s[:, 0])][0][0]
        objective1 = s[np.argsort(s[:, 1])][0][1]
        objective2 = s[np.argsort(s[:, 2])][0][2]
        objective3 = s[np.argsort(s[:, 3])][0][3]
        current_objectives = [objective0, objective1, objective2, objective3]

        s = kwargs["SOLUTIONS"][0]
        if self.best_objectives is None:
            self.best_objectives = current_objectives
            # LOGGER.info(self.best_objectives)
        else:
            # LOGGER.info(self.best_objectives)
            # LOGGER.info(current_objectives)
            if all([x <= y for x, y in zip(self.best_objectives, current_objectives)]):
                self.evaluations += 1
            else:
                self.best_objectives = [
                    min(x, y) for x, y in zip(self.best_objectives, current_objectives)
                ]
                self.evaluations = 0
        LOGGER.info(f"{self.evaluations} evaluations without improvement")

    @property
    def is_met(self):
        return self.evaluations >= (self.idle_evaluations - (self.seconds / 14) ** 2)
        # return self.evaluations >= (self.idle_evaluations)


class StoppingByConstraintsMet(TerminationCriterion):
    def __init__(self):
        super(StoppingByConstraintsMet, self).__init__()
        self.constraints_met = False

    def update(self, *args, **kwargs):
        seconds = kwargs["COMPUTING_TIME"]

        c = np.array([s.constraints for s in kwargs["SOLUTIONS"]])
        if np.sum(c) == 0:
            self.constraints_met = True
        LOGGER.info(f"{str(datetime.timedelta(seconds=seconds))} - {np.sum(c)}")

    @property
    def is_met(self):
        return self.constraints_met


class StoppingByFullPareto(TerminationCriterion):
    def __init__(self, offspring_size: int):
        super(StoppingByFullPareto, self).__init__()
        self.offspring_size = offspring_size

    def update(self, *args, **kwargs):
        self.current_offspring_size = len(kwargs["SOLUTIONS"])
        LOGGER.info(f"Current size = {self.current_offspring_size}.")

    @property
    def is_met(self):
        return self.offspring_size <= self.current_offspring_size


class Constraints:
    def __init__(self, solution, infra, pipe):
        self.s = np.asfarray(solution.variables, dtype=np.bool)
        self.infra = infra
        self.pipe = pipe

    def __privacy(self, location, type):
        # the location of the devices
        location_devices = self.infra[location].to_numpy()
        # the location of the models
        location_models = self.pipe[location].to_numpy()

        # the location of the devices for the given solution
        sol_loc_dev = self.s * location_devices
        # the location of the models for the given solution
        sol_loc_mod = self.s * location_models[np.newaxis].T

        # models with privacy type equals to {type}
        privacy_mask = np.where(self.pipe.privacy_type.to_numpy() == type, 1, 0)[
            np.newaxis
        ].T

        # apply the mask for the given type
        device_matrix = sol_loc_dev * privacy_mask
        model_matrix = sol_loc_mod * privacy_mask

        # returns true if both matrixes are equal
        return (device_matrix == model_matrix).all()

    def privacy_constraint(self):
        if not self.__privacy("country", 2):
            return -1
        if not self.__privacy("continent", 1):
            return -1
        return 0

    def cpu_constraint(self):
        x = self.s.transpose() * self.pipe.cpus.to_numpy()
        sum_rows = np.sum(x, axis=1)
        thread_count = self.infra.thread_count.to_numpy()
        return 0 if not (thread_count < sum_rows).any() else -1

    def deployment_constraint(self):
        sum_rows = np.sum(self.s, axis=1)
        # the sum of all rows should be bigger or equal to one
        return 0 if not (sum_rows < 1).any() else -1

    def ram_constraint(self):
        x = self.s.transpose() * self.pipe.memory.to_numpy()
        sum_rows = np.sum(x, axis=1)
        memory = self.infra.memory.to_numpy()
        return 0 if not (memory < sum_rows).any() else -1


class Evaluate:
    def __init__(self, file_solution: str):
        self.file_solution = file_solution

        s = np.genfromtxt(file_solution, delimiter=",")
        number_of_models, number_of_devices = s.shape

        self.solution = BinarySolution(
            number_of_variables=number_of_models, number_of_objectives=1
        )
        for i in range(number_of_models):
            self.solution.variables[i] = [False for _ in range(number_of_devices)]

        for i in range(number_of_models):
            for j in range(number_of_devices):
                if s[i][j]:
                    self.solution.variables[i][j] = True

        file_infrastructure = "src/resources/infrastructure.csv"
        pipeline_location = "src/resources/pipeline_40.yml"
        with open(pipeline_location, "r") as input_data_file:
            input_pipeline = input_data_file.read()

        # load pipeline
        self.pipe = Pipeline(input_pipeline).load()
        # load infrastructure
        self.infra = Infrastructure(file_infrastructure).load()

        self.c = Constraints(self.solution, self.infra, self.pipe)

    def cost(self) -> float:
        cost = Objectives().get_consumption(self.pipe, self.infra, self.solution)
        return cost

    def model_performance(self) -> float:
        model_performance = Objectives().get_performance(
            self.pipe, self.infra, self.solution
        )
        return model_performance

    def resilience(self) -> float:
        resilience = Objectives().get_resilience(self.infra, self.solution)
        return resilience

    def network_performance(self) -> float:
        file_latencies = "src/resources/latencies.csv"
        ld = Latency(file_location=file_latencies).load()
        network_performance = Objectives().get_network_performance(
            ld, self.pipe, self.infra, self.solution
        )
        return network_performance

    def constraint_privacy(self) -> bool:
        return self.c.privacy_constraint()

    def constraint_cpu(self) -> bool:
        return self.c.cpu_constraint()

    def constraint_deployment(self) -> bool:
        return self.c.deployment_constraint()

    def constraint_ram(self) -> bool:
        return self.c.ram_constraint()
