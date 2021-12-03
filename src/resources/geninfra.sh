#!/bin/bash

psql -d optimizer -h optimizer.cjxwnlyzdbn4.us-east-2.rds.amazonaws.com -U postgres -c "\copy (select * from public.servers where stage='prod') TO '/tmp/hola.csv' CSV HEADER";
