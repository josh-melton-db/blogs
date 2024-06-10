# Databricks notebook source
# MAGIC %md
# MAGIC ## For the blog post corresponding to this notebook, [check here](https://community.databricks.com/t5/technical-blog/grouped-pandas-optimization/ba-p/68666)

# COMMAND ----------

from pyspark.sql.functions import rand
import pandas as pd

def generate_initial_df(num_rows, num_devices, num_trips):
    return (
        spark.range(num_rows)
        .withColumn('device_id', (rand()*num_devices).cast('int'))
        .withColumn('trip_id', (rand()*num_trips).cast('int'))
        .withColumn('sensor_reading', (rand()*1000))
        .drop('id')
    )

df = generate_initial_df(5000000, 100, 1000)
df.display()

# COMMAND ----------

def normalize_device_and_trip(pdf: pd.DataFrame) -> pd.DataFrame:
   reading = pdf.sensor_reading
   pdf['normalized_reading'] = reading.mean() / reading.std()
   return pdf

expected_schema = 'device_id int, trip_id int, sensor_reading long, normalized_reading long'
df.groupBy('device_id', 'trip_id').applyInPandas(normalize_device_and_trip, expected_schema).write.format("noop").mode("overwrite").save()

# COMMAND ----------

print(df.count() / df.select('device_id').distinct().count()) # 50,000
print(df.count() / df.select('device_id', 'trip_id').distinct().count()) # ~50

# COMMAND ----------

def normalize_trip(pdf: pd.DataFrame) -> pd.DataFrame:
   reading = pdf.sensor_reading
   pdf['normalized_reading'] = reading.mean() / reading.std()
   return pdf

def normalize_device(pdf: pd.DataFrame) -> pd.DataFrame:
    return pdf.groupby("trip_id").apply(normalize_trip)

expected_schema = 'device_id int, trip_id int, sensor_reading long, normalized_reading long'
df.groupBy('device_id').applyInPandas(normalize_device, expected_schema).write.format("noop").mode("overwrite").save()

# COMMAND ----------


