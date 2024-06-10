# Databricks notebook source
# MAGIC %pip install dbldatagen
# MAGIC
# MAGIC import dbldatagen as dg
# MAGIC from pyspark.sql.types import StringType, FloatType
# MAGIC import pandas as pd
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC from sklearn.ensemble import GradientBoostingClassifier
# MAGIC import mlflow
# MAGIC from mlflow.models import infer_signature
# MAGIC from pyspark.sql.functions import struct, col

# COMMAND ----------

data_spec = (dg.DataGenerator(spark, name="feature_data", rows=1000)
            .withColumn("sensor1", FloatType(), minValue=0, maxValue=100, random=True)
            .withColumn("sensor2", FloatType(), minValue=0, maxValue=50, random=True)
            .withColumn("sensor3", FloatType(), minValue=0, maxValue=25, random=True)
            .withColumn("result", StringType(), values=['class1', 'class2', 'class3', 'class4', 'class5', 
                                                        'class6', 'class7', 'class8', 'class9', 'class10'], random=True))

df_features = data_spec.build()

# COMMAND ----------

features = df_features.toPandas()

X = features[['sensor1', 'sensor2', 'sensor3']]
y = features['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# COMMAND ----------

import numpy as np

class ProbabilityModel(mlflow.pyfunc.PythonModel):
    def __init__(self, sklearn_model, n=3):
        self.sklearn_model = sklearn_model
        self.n = n
    
    def predict(self, context, model_input):
        predictions = model.predict_proba(model_input)
        top_n_indices = np.argsort(-predictions, axis=1)[:, :self.n]
        top_n_classes = model.classes_[top_n_indices]
        top_n_probabilities = predictions[np.arange(len(predictions))[:, None], top_n_indices]
        
        pred_dict = {}
        for i in range(self.n):
            pred_dict['predicted_class_'+str(i+1)] = top_n_classes[:, i]
            pred_dict['predicted_probability_'+str(i+1)] = top_n_probabilities[:, i]
        
        return pd.DataFrame(pred_dict)

prob_model = ProbabilityModel(model)
predictions = prob_model.predict('', X_test[:5])
display(predictions)

# COMMAND ----------

signature = infer_signature(sample, predictions)
with mlflow.start_run() as run:
    logged_model = mlflow.pyfunc.log_model("model", python_model=prob_model, input_example=sample, signature=signature)

# COMMAND ----------

# Load the model from MLflow into a Spark UDF
model_uri = f"runs:/{run.info.run_id}/model"
loaded_model = mlflow.pyfunc.load_model(model_uri)
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri)

# COMMAND ----------

# Apply the model to a Spark DataFrame in parallel
predictions_df = (
    df_features.withColumn("predictions", predict_udf(struct(*df_features.columns)))
    .select("*", "predictions.*")
    .drop("predictions")
)
predictions_df.display()

# COMMAND ----------


