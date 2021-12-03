\connect optimizer

CREATE TABLE servers (
  "hostname" varchar PRIMARY KEY,
  "thread_count" integer,
  "frequency" float,
  "memory" float,
  "resillience" integer,
  "performance" integer,
  "parallelization" float[],
  "consumption" float[],
  "country_code" varchar,
  "cloud_type" varchar,
  "ts" timestamp,
  "env" varchar,
  "instance_type" varchar
);
