# Databricks notebook source
# MAGIC %md
# MAGIC ## For the corresponding blog post, [check here](https://community.databricks.com/t5/technical-blog/scaling-pandas-with-databricks-passing-parameters-to-pandas-udfs/ba-p/65123)

# COMMAND ----------

import pandas as pd

df = spark.createDataFrame(pd.DataFrame({'type': ['turbine', 'turbine', 'propeller', 'turbine', 'propeller', 'propeller'],
                                          'sensor_reading': [10, 7, 25, 12, 29, 36]}))

def normalize(pdf: pd.DataFrame) -> pd.DataFrame:
    reading = pdf.sensor_reading
    pdf['normalized'] = reading.mean() / reading.std()
    return pdf

df.groupBy('type').applyInPandas(normalize, 'type string, sensor_reading long, normalized long').show()

# COMMAND ----------

# We don't have a way to pass a value like the mean of the whole dataframe - this throws an error
def normalize_plus_value(pdf: pd.DataFrame, value: int) -> pd.DataFrame:
    reading = pdf.sensor_reading
    pdf['normalized'] = value + (reading.mean() / reading.std())
    return pdf

df.groupBy('type').applyInPandas(normalize_plus_value, 'type string, sensor_reading long, normalized long').show()

# COMMAND ----------

def normalize_with_value(value: int):
    # Returning this function "injects" the value into the function we'll use for applyInPandas
    def normalize(pdf: pd.DataFrame) -> pd.DataFrame:
        reading = pdf.sensor_reading
        pdf['normalized'] = value - (reading.mean() / reading.std())
        return pdf
    return normalize

# Now we can initialize the function with a value inserted
average = df.selectExpr('avg(sensor_reading) as average').collect()[0][0]
dynamic_normalize = normalize_with_value(average)
df.groupBy('type').applyInPandas(dynamic_normalize, 'type string, sensor_reading long, normalized long').show()

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, lit
from statsmodels.tsa.arima.model import ARIMA
import mlflow

# Fit and run an ARIMA model using a Pandas UDF with the hyperparameters passed in
def create_arima_forecaster(order):
    @pandas_udf("double")
    def forecast_arima(value: pd.Series) -> pd.Series:
        mlflow.sklearn.autolog(disable=True)
        model = ARIMA(value, order=order)
        model_fit = model.fit()
        return model_fit.predict()
    return forecast_arima

# Minimal Spark code - just select one column and add another. We can still use Pandas for our logic
forecast_arima = create_arima_forecaster((1, 2, 3))
df.withColumn('predicted_reading', forecast_arima('sensor_reading')).show()

# COMMAND ----------

from hyperopt import hp, fmin, tpe, Trials
from pyspark.ml.evaluation import RegressionEvaluator


# Define the hyperparameter search space
search_space = {'p': 1, 'd': hp.quniform('d', 2, 3, 1), 'q': hp.quniform('q', 2, 4, 1)}

# Define the objective function to be minimized
def objective(params):
    order = (params['p'], params['d'], params['q'])
    forecast_arima = create_arima_forecaster(order)
    arima_output = df.withColumn('predicted_reading', forecast_arima('sensor_reading'))
    evaluator = RegressionEvaluator(predictionCol="predicted_reading", labelCol="sensor_reading", metricName="rmse")
    rmse = evaluator.evaluate(arima_output)
    return rmse

# Run the hyperparameter optimization
trials = Trials()
best = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=6, trials=trials)
print('Best hyperparameters: ', best)

# COMMAND ----------


