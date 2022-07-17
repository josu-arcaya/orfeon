from email.mime import base
import json
import logging
import numpy as np
import os
import pandas as pd
import pycountry
import pycountry_convert as pc
import yaml
from collections import namedtuple

from jmetal.core.observer import Observer
from jmetal.core.solution import BinarySolution
from jmetal.util.comparator import DominanceComparator
from jmetal.util.termination_criterion import TerminationCriterion

continent_codes = {'AF': 100, 'AN': 101, 'AS': 102, 'EU': 103, 'NA': 104, 'OC': 105, 'SA': 106}

LOGGER = logging.getLogger('jmetal')

class Parser:

    def __init__(self, input_data: str):
        self.x = self.file2obj(input_data)

    def object_hook(self, d):
        return namedtuple('X', d.keys())(*d.values())

    def file2obj(self, data):
        pass

class Infrastructure():

    def __init__(self, file_location: str):
        self.infrastructure = pd.read_csv(file_location)

        x = lambda txt : np.fromstring(txt[1:-1], sep=',')
        self.infrastructure.consumption = self.infrastructure.consumption.apply(x)
        self.infrastructure.parallelization = self.infrastructure.parallelization.apply(x)

        y = lambda row : int(pycountry.countries.get(alpha_2=row['country_code']).numeric)
        self.infrastructure['country'] = self.infrastructure.apply(y, axis=1)

        # convert alpha2 country_code to continent name
        z = lambda row : continent_codes[pc.country_alpha2_to_continent_code(row['country_code'])]
        self.infrastructure['continent'] = self.infrastructure.apply(z, axis=1)

    def load(self):
        return self.infrastructure

class Pipeline(Parser):

    def file2obj(self, data):
        y = yaml.load(data, Loader=yaml.FullLoader)
        return json.loads(json.dumps(y), object_hook=self.object_hook)

    def load(self):

        columns = ['cpus','memory', 'country', 'privacy_type', 'continent','link']
        data = {columns[0]: [], columns[1]: [], columns[2]: [], columns[3]: [], columns[4]: [], columns[5]: []}
        for i in range(len(self.x.pipeline)):
            data[columns[0]].append(float(self.x.pipeline[i].resources.cpus))
            data[columns[1]].append(float(self.x.pipeline[i].resources.memory))
            # get numeric from alpha_2 country code
            country_numeric = pycountry.countries.get(alpha_2=self.x.pipeline[i].privacy.location).numeric
            data[columns[2]].append(int(country_numeric))
            data[columns[3]].append(int(self.x.pipeline[i].privacy.type))
            # get continent from alpha_2 country code
            data[columns[4]].append(continent_codes[pc.country_alpha2_to_continent_code(self.x.pipeline[i].privacy.location)])
            data[columns[5]].append(int(self.x.pipeline[i].link))
        b = pd.DataFrame(data=data, columns=columns)

        return b

class ParetoTools():

    def __init__(self, front):
        self.front = front

    def save(self):
        filename = f"/tmp/pareto"
        with open(filename, "w") as pareto_file:
            for s in self.front:
                pareto_file.write(f"{abs(s.objectives[0])},{abs(s.objectives[1])},{s.objectives[2]}\n")

class Objectives():

    def get_consumption(self, pipe: Pipeline, infra: Infrastructure, solution: BinarySolution) -> int:
        s = np.asfarray(solution.variables, dtype=np.bool)

        threads_required = s.transpose().dot(pipe.cpus.to_numpy()).astype(int)
        
        consumption = [m[min(n, len(m)-1)] for m,n in zip(infra.consumption, threads_required)]
        return sum(consumption)

    def get_performance(self, pipe: Pipeline, infra: Infrastructure, solution: BinarySolution) -> int:
        s = np.asfarray(solution.variables, dtype=np.bool)

        base_performance = s*infra.performance.to_numpy()

        threads_required = s.transpose().dot(pipe.cpus.to_numpy()).astype(int)

        coefficient = [m[min(n-1, len(m)-1)] for m,n in zip(infra.parallelization, threads_required)]

        return (base_performance*coefficient).max(1).sum()
        #return (base_performance*coefficient).mean(1).sum()

    def get_resilience(self, infra: Infrastructure, solution: BinarySolution) -> int:
        s = np.asfarray(solution.variables, dtype=np.bool)
        if s.sum() == 0:
            return 0
        x = s*infra.resillience.to_numpy()
        x = np.ma.masked_array(x, mask=s==0)
        return np.nanmin(x,axis=1).sum()

    # https://blog.devgenius.io/linux-how-to-measure-network-performance-c859a98abbf0
    def get_network_performance(self, ld: np.array, pipe: Pipeline, infra: Infrastructure, solution: BinarySolution) -> int:

        # bandwidth
        # throughput
        # latency

        s = np.asfarray(solution.variables, dtype=np.bool)
        
        z = s*ld.transpose()

        c = np.empty(shape=s.shape)
        for index, value in pipe.link.iteritems():
            c[index] = s[value]

        #c = np.array(s[pipe.link[0]])
        #c = np.vstack([c, s[pipe.link[1]]])
        #c = np.vstack([c, s[pipe.link[2]]])
        #c = np.vstack([c, s[pipe.link[3]]])

        c = np.array([c])

        #print(z.shape)
        #print(c.shape)

        z*c.transpose()
        
        base_bandwidth = s.dot(infra.bandwidth.to_numpy()).sum()

        # check the latency
        return base_bandwidth

        #return base_bandwidth + np.sum(z)


class WriteObjectivesToFileObserver(Observer):
    
    def __init__(self) -> None:
        self.filename = '/tmp/fitnesses'
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def update(self, *args, **kwargs):
        evaluations = kwargs['EVALUATIONS']
        LOGGER.info(f"Evaluations: {evaluations}")

        #solutions = np.array([s.objectives for s in kwargs['SOLUTIONS'][:10]])
        s = np.array([s.objectives for s in kwargs['SOLUTIONS']])
        
        #means = s.mean(0)
        objective0 = s[np.argsort(s[:,0])][:3].mean(0)[0]
        #objective0 = s[np.argsort(s[:,0])][0][0]
        objective1 = s[np.argsort(s[:,1])][:3].mean(0)[1]
        #objective1 = s[np.argsort(s[:,1])][0][1]
        objective2 = s[np.argsort(s[:,2])][:3].mean(0)[2]
        #objective2 = s[np.argsort(s[:,2])][0][2]
        objective3 = s[np.argsort(s[:,3])][:3].mean(0)[3]
        
        with open(self.filename, 'a') as out_file:
            #out_file.write(f"{abs(means[0])},{abs(means[1])},{abs(means[2])}\n")
            out_file.write(f"{abs(objective0)},{abs(objective1)},{abs(objective2)},{abs(objective3)}\n")

class StoppingByNonDominance(TerminationCriterion):

    def __init__(self, idle_evaluations: int):
        super(StoppingByNonDominance, self).__init__()
        self.idle_evaluations = idle_evaluations
        self.evaluations = 0
        self.best_solution = None
        

    def update(self, *args, **kwargs):
        current_solution = kwargs['SOLUTIONS'][0]
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
        self.seconds = kwargs['COMPUTING_TIME']

        s = np.array([s.objectives for s in kwargs['SOLUTIONS']])
        objective0 = s[np.argsort(s[:,0])][0][0]
        objective1 = s[np.argsort(s[:,1])][0][1]
        objective2 = s[np.argsort(s[:,2])][0][2]
        current_objectives = [objective0, objective1, objective2]

        s = kwargs['SOLUTIONS'][0]
        if self.best_objectives is None:
            self.best_objectives = current_objectives
            #LOGGER.info(self.best_objectives)
        else:
            #LOGGER.info(self.best_objectives)
            #LOGGER.info(current_objectives)
            if all([x<=y for x,y in zip(self.best_objectives, current_objectives)]):
                self.evaluations += 1
            else:
                self.best_objectives = [min(x,y) for x,y in zip(self.best_objectives, current_objectives)]
                self.evaluations = 0
        LOGGER.info(f"{self.evaluations} evaluations without improvement")


    @property
    def is_met(self):
        return self.evaluations >= (self.idle_evaluations - (self.seconds/10)**2)

class StoppingByFullPareto(TerminationCriterion):

    def __init__(self, offspring_size: int):
        super(StoppingByFullPareto, self).__init__()
        self.offspring_size = offspring_size

    def update(self, *args, **kwargs):
        self.current_offspring_size = len(kwargs['SOLUTIONS'])
        LOGGER.info(f"Current size = {self.current_offspring_size}.")

    @property
    def is_met(self):
        return self.offspring_size <= self.current_offspring_size
