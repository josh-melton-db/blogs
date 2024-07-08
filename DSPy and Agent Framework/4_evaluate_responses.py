# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-sdk databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries
from databricks import agents
import pandas as pd
import os
import json
from openai import OpenAI
import concurrent.futures
from pyspark.sql.functions import concat, col, named_struct, lit, array
import requests
import mlflow
import yaml
from utils.demo import generate_questions
from utils.inference_log_parser import unpack_and_split_payloads, dedup_assessment_logs, get_table_url

# COMMAND ----------

# DBTITLE 1,Get Config
with open("rag_config.yaml", "r") as file:
    rag_config = yaml.safe_load(file)
chunk_table = rag_config.get("demo_config").get("chunk_table")
chunks_df = spark.table(chunk_table)

chunk_text_key = rag_config.get("chunk_column_name")
chunk_id_key = rag_config.get("chunk_id_column_name")
doc_uri_key =  rag_config.get("document_source_id")
inference_table_uc_fqn = rag_config.get("demo_config").get("inference_table_uc_fqn")
request_log_output_uc_fqn = rag_config.get("demo_config").get("request_log_output_uc_fqn")
assessment_log_output_uc_fqn = rag_config.get("demo_config").get("assessment_log_output_uc_fqn")
model_fqdn = rag_config.get("demo_config").get("model_fqdn")
synthetic_eval_set_table_uc_fqn = rag_config.get("demo_config").get("synthetic_eval_set_table_uc_fqn")
chat_endpoint = rag_config.get("chat_endpoint")
endpoint_name = rag_config.get("demo_config").get("endpoint_name")

# COMMAND ----------

# DBTITLE 1,Generate Questions About Chunks
synthetic_data_raw = [q for q in generate_questions(chunks_df, chunk_text_key, chunk_id_key, dbutils) if q]
print(synthetic_data_raw)

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')
from mlflow import MlflowClient
client = MlflowClient()
version = client.get_registered_model(model_fqdn).aliases['challenger']
model_uri = f"models:/{model_fqdn}/{version}"

# COMMAND ----------

with mlflow.start_run():
    eval_results = mlflow.evaluate(
        data=pd.DataFrame(synthetic_data_raw), 
        model=model_uri,
        model_type="databricks-agent"
    )
    
eval_results.metrics

# COMMAND ----------

# Evaluation results including LLM judge scores/rationales for each row in your evaluation set
per_question_results_df = eval_results.tables["eval_results"]
(
    spark.createDataFrame(per_question_results_df)
    .write.mode("overwrite")
    .saveAsTable(synthetic_eval_set_table_uc_fqn+"_eval_metrics")
)

# You can click on a row in the `trace` column to view the detailed MLflow trace
display(per_question_results_df)

# COMMAND ----------

payload_df = spark.table(f"{model_fqdn}_1_payload") # change the 1 to the version of the model you deployed to the endpoint first
display(payload_df)

# COMMAND ----------

# DBTITLE 1,Send Inferences to App for Human Eval
request_ids = payload_df.select("databricks_request_id").rdd.flatMap(lambda x: x).collect()
agents.enable_trace_reviews(model_name=model_fqdn, request_ids=request_ids)

# COMMAND ----------


