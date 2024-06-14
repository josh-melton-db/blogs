# Databricks notebook source
from pyspark.sql.functions import rand, pandas_udf, col
import pandas as pd

def generate_initial_df(num_rows, num_devices, num_trips):
    return (
        spark.range(num_rows)
        .withColumn('device_id', (rand()*num_devices).cast('int'))
        .withColumn('trip_id', (rand()*num_trips).cast('int'))
        .withColumn('sensor_reading', (rand()*1000))
        .drop('id')
    )

df = generate_initial_df(5000000, 10000, 50)
df.display()

# COMMAND ----------

@pandas_udf('double')
def calculate_sqrt(sensor_reading: pd.Series) -> pd.Series:
    return sensor_reading.apply(lambda x: x**0.5)

df = df.withColumn('sqrt_reading', calculate_sqrt(col('sensor_reading')))
df.display()

# COMMAND ----------

def denormalize(pdf: pd.DataFrame) -> pd.DataFrame:
   aggregated_df = pdf.groupby('device_id', as_index=False).agg(
       {'trip_id': lambda x: list(x), 'sensor_reading': 'mean', 'sqrt_reading': 'mean'}
   )
   return aggregated_df

expected_schema = 'device_id int, trip_id array<int>, sensor_reading long, sqrt_reading long'
df = df.groupBy('device_id').applyInPandas(denormalize, schema=expected_schema)
df.display()

# COMMAND ----------

from collections.abc import Iterator

def renormalize(itr: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for pdf in itr:
        # Unpack the list of values from the trip_id column into their own rows
        pdf = pdf.explode('trip_id')
        yield pdf

expected_schema = 'device_id int, trip_id int, sensor_reading long, sqrt_reading long'
df = df.mapInPandas(renormalize, schema=expected_schema)
df.display()

# COMMAND ----------


