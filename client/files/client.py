#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import psycopg2
import re
import socket
import subprocess
import uuid
from datetime import datetime
from ipwhois import IPWhois
from requests import get

"""
    According to Eurostat
    https://ec.europa.eu/eurostat/statistics-explained/index.php/Electricity_price_statistics
"""
electricity_price = 0.2247

def count_cpus():
    process = subprocess.Popen(['cat', '/proc/cpuinfo'],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    last_processor = re.findall('processor\\t:\s(\d+)', stdout.decode('utf-8'))
    return int(last_processor[-1]) + 1

def processor_frequency():
    process = subprocess.Popen(['/usr/bin/lscpu'],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    match = re.search('CPU max MHz:\s+(\d+)', stdout.decode('utf-8'))
    if match:
        return match.group(1)
    match = re.search('CPU MHz:\s+(\d+)', stdout.decode('utf-8'))
    if match:
        return match.group(1)
    return -1

def count_memory():
    process = subprocess.Popen(['free', '-mh'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    memory = re.search('Mem:\s+(\d+\.?\d+)G', stdout.decode('utf-8'))[1]
    return memory

def get_resillience():
    process = subprocess.Popen(['/opt/lynis/lynis', 'audit', 'system', '--no-colors'],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        cwd='/opt/lynis')
    stdout, stderr = process.communicate()
    resillience = re.search('Hardening index\s:\s(\d+)', stdout.decode('utf-8'))[1]
    return resillience

def get_performance():
    # disable tensorflow ugly logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from ai_benchmark import AIBenchmark

    benchmark = AIBenchmark()
    results = benchmark.run_inference()
    return results.inference_score

def get_parallelization(thread_count):
    parallelization = []
    # calculate base time
    process = subprocess.Popen(['stress-ng', '--cpu', '1', '--cpu-ops', '8000'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    base_time = re.search('completed in (\d+.\d+)s', stderr.decode('utf-8'))[1]
    parallelization.append( 1.0 )
    # the following will be a fraction of base time
    for i in range(2, thread_count+1):
        process = subprocess.Popen(['stress-ng', '--cpu', str(i), '--cpu-ops', str(8000*i)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        completion_time = re.search('completed in (\d+.\d+)s', stderr.decode('utf-8'))[1]
        parallelization.append( float(base_time) / float(completion_time) )
    return parallelization

def get_consumption(thread_count):
    consumption = []
    if os.getenv('INSTANCE_COST') is not None :
        for c in os.getenv('INSTANCE_COST').split():
            consumption.append(float(c))
        return consumption
    for i in range(1, thread_count+1):
        p_stress = subprocess.Popen(['stress-ng', '--cpu', str(i)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        p_powerstat = subprocess.Popen(['powerstat', '-R', '-c', '-z'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = p_powerstat.communicate()
        if (p_powerstat.returncode != 0):
            return None
        consumption.append( float(re.search('CPU:\s+(\d+.\d+) Watts on average', stdout.decode('utf-8'))[1])*electricity_price )
        p_stress.terminate()
    return consumption

def get_location():
    ip_address = get('https://api.ipify.org').text
    obj = IPWhois(ip_address)
    country_code = obj.lookup_rdap(depth=1)['asn_country_code']
    return country_code

def insert_data(thread_count, frequency, memory, resillience, performance, parallelization, consumption, country_code, cloud_type, instance_type, ts, stage):
    try:
        host = os.getenv('DB_HOST')
        password = os.getenv('DB_PASS')
        connection = psycopg2.connect(user="postgres",
            password=password,
            host=host,
            port="5432",
            database="optimizer")

        cursor = connection.cursor()
        sql = """ INSERT INTO servers (hostname, thread_count, frequency, memory, resillience, performance, parallelization, consumption, country_code, cloud_type, instance_type, ts, stage) 
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) """
        cursor.execute(sql,
            (str(uuid.uuid4()), thread_count, frequency, memory, resillience, performance, parallelization, consumption, country_code, cloud_type, instance_type, ts, stage))

        connection.commit()
        logging.info("Record inserted successfully")
    except Exception as error:
        logging.error(f"Failed to insert record, {error}.")
        exit(-1)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s')

    if os.geteuid() != 0:
        logging.error('Please re-run as root.')
        exit(-1)

    if os.getenv('DB_HOST') is None or os.getenv('DB_PASS') is None:
        logging.error('Please define DB_HOST and DB_PASS environment variables.')
        exit(-1)

    if os.getenv('CLOUD_TYPE') is None:
        logging.error('Please define CLOUD_TYPE environment variable.')
        exit(-1)

    if os.getenv('INSTANCE_TYPE') is None:
        logging.error('Please define INSTANCE_TYPE environment variable.')
        exit(-1)

    if os.getenv('STAGE') is None:
        logging.error('Please define STAGE environment variable.')
        exit(-1)

    text = 'This agent retrieves metrics to be used by the optimization process.'
    parser = argparse.ArgumentParser(description=text)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-r', '--resilience', action='store_true',
                          help='Resilience metric', required=False)
    required.add_argument('-p', '--performance', action='store_true',
                          help='Performance metric', required=False)
    required.add_argument('-m', '--multithreading', action='store_true',
                          help='Parallelization metric', required=False)
    required.add_argument('-l', '--location', action='store_true',
                          help='Location metric', required=False)
    required.add_argument('-c', '--consumption', action='store_true',
                          help='Consumption metric', required=False)
    required.add_argument('-f', '--frequency', action='store_true',
                          help='Frequency metric', required=False)
    required.add_argument('-a', '--all', action='store_true',
                          help='Enable all metrics', required=False)

    thread_count = count_cpus()
    logging.info(f"Thread Count: {thread_count}")

    memory = count_memory()
    logging.info(f"Memory: {memory}")

    args = parser.parse_args()
    if args.resilience:
        resillience = get_resillience()
        logging.info(f"Resillience: {resillience}")
    
    if args.performance:
        performance = get_performance()
        logging.info(f"Performance: {performance}")

    if args.multithreading:
        parallelization = get_parallelization(thread_count)
        logging.info(f"Parallelization = {parallelization}")

    if args.location:
        country_code = get_location()
        logging.info(f"Country code = {country_code}")

    if args.consumption:
        consumption = get_consumption(thread_count)
        logging.info(f"Consumption = {consumption}")

    if args.frequency:
        frequency = processor_frequency()
        logging.info(f"CPU MHz = {frequency}")

    if args.all:
        resillience = get_resillience()
        logging.info(f"Resillience: {resillience}")
        performance = get_performance()
        logging.info(f"Performance: {performance}")
        parallelization = get_parallelization(thread_count)
        logging.info(f"Parallelization = {parallelization}")
        consumption = get_consumption(thread_count)
        logging.info(f"Consumption = {consumption}")
        country_code = get_location()
        logging.info(f"Country code = {country_code}")
        frequency = processor_frequency()
        logging.info(f"CPU MHz = {frequency}")
        insert_data(thread_count, frequency, memory, resillience,
            performance, parallelization,
            consumption, country_code,
            os.getenv('CLOUD_TYPE'), os.getenv('INSTANCE_TYPE'),
            datetime.now(), os.getenv('STAGE'))

if __name__ == '__main__':
    main()
