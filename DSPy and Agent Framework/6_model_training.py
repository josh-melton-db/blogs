# Databricks notebook source
# MAGIC %md
# MAGIC ## Warning: This notebook is not working yet

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-genai databricks-sdk mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml

with open("rag_config.yaml", "r") as file:
    rag_config = yaml.safe_load(file)
model_fqdn = rag_config.get("demo_config").get("model_fqdn")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, ArrayType

# Define the schema for the JSON data
message_schema = StructType([
    StructField("role", StringType(), True),
    StructField("content", StringType(), True),
])

request_schema = StructType([
    StructField("messages", ArrayType(message_schema), True)
])

response_schema = StructType([
    StructField("id", StringType(), True),
    StructField("object", StringType(), True),
    StructField("created", StringType(), True),
    StructField("choices", ArrayType(
        StructType([
            StructField("message", message_schema, True)
        ])
    ), True)
])

schema = StructType([
    StructField("databricks_output", StructType([
        StructField("trace", StructType([
            StructField("data", StructType([
                    StructField("request", StringType()),
                    StructField("response", StringType())
                ]))
            ]))
        ])),
]) #TODO: naming

# COMMAND ----------

from pyspark.sql.functions import from_json, col, array

df = spark.sql(f"SELECT * FROM {model_fqdn}_1_payload")
parsed_df = df.withColumn("parsed_request", from_json(col("response"), schema).getItem("databricks_output").getItem("trace").getItem("data").getItem("request"))
parsed_df = parsed_df.withColumn("parsed_request", from_json(col("parsed_request"), request_schema).getItem("messages").getItem(0))
parsed_df = parsed_df.withColumn("parsed_response", from_json(col("response"), response_schema).getItem("choices").getItem(0).getItem("message"))
parsed_df = parsed_df.withColumn("messages", array(col("parsed_request"), col("parsed_response")))
parsed_df = parsed_df.select("messages")
parsed_df.display()

# COMMAND ----------

table_name = "josh_melton.generated_rag_demo.ft_test"
parsed_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(table_name)

# COMMAND ----------

from databricks.model_training import foundation_model as fm
#Return the current cluster id to use to read the dataset and send it to the fine tuning cluster. See https://docs.databricks.com/en/large-language-models/foundation-model-training/create-fine-tune-run.html#cluster-id
def get_current_cluster_id():
  import json
  return json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().safeToJson())['attributes']['clusterId']


#Let's clean the model name
registered_model_name = model_fqdn + "_ft"

run = fm.create(
    data_prep_cluster_id=get_current_cluster_id(),  # required if you are using delta tables as training data source. This is the cluster id that we want to use for our data prep job.
    model="databricks/dbrx-instruct",  # Here we define what model we used as our baseline
    train_data_path=table_name,
    task_type="CHAT_COMPLETION",  # Change task_type="INSTRUCTION_FINETUNE" if you are using the fine-tuning API for completion.
    register_to=registered_model_name,
    training_duration="5ep", #only 5 epoch to accelerate the demo. Check the mlflow experiment metrics to see if you should increase this number
    learning_rate="5e-7",
)

print(run)

# COMMAND ----------


