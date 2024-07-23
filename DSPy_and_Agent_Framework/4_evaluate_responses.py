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
from utils.demo import generate_questions, query_chain
from utils.inference_log_parser import unpack_and_split_payloads, dedup_assessment_logs, get_table_url

# COMMAND ----------

# DBTITLE 1,Get Config
with open("rag_config.yaml", "r") as file:
    rag_config = yaml.safe_load(file)
chunk_table = rag_config.get("demo_config").get("chunk_table")

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

payload_df = spark.table(f"{model_fqdn}_payload")
payload_count = payload_df.count()

# COMMAND ----------

# DBTITLE 1,Generate Questions About Chunks
from utils.demo import query_chain

chunks_df = spark.table(chunk_table)
synthetic_data_raw = [q for q in generate_questions(chunks_df, chunk_text_key, doc_uri_key, dbutils) if q]
for question in synthetic_data_raw:
    response = query_chain(question["request"], endpoint_name, dbutils)
    question["request_id"] = response["id"]
    question["response"] = response["choices"][0]["message"]["content"]
synthetic_data_pdf = pd.DataFrame(synthetic_data_raw)
(
    spark.createDataFrame(synthetic_data_pdf)
    .write.mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable(synthetic_eval_set_table_uc_fqn)
)
display(synthetic_data_pdf)

# COMMAND ----------

synthetic_data_pdf = synthetic_data_pdf.sort_values(by="expected_response", key=lambda x: x.str.len())
synthetic_data_pdf = synthetic_data_pdf.head(100)[.6:]

with mlflow.start_run():
    eval_results = mlflow.evaluate(
        data=synthetic_data_pdf,
        model_type="databricks-agent"
    )

eval_results.metrics

# COMMAND ----------

from pyspark.sql.functions import expr

# Evaluation results including LLM judge scores/rationales for each row in your evaluation set
per_question_results_pdf = eval_results.tables["eval_results"]
(
    spark.createDataFrame(per_question_results_pdf)
    .withColumn("evaluation_round", lit("baseline"))
    .withColumn("expected_retrieved_context", expr("expected_retrieved_context[0].doc_uri"))
    .write.mode("overwrite").option("mergeSchema", "true")
    .saveAsTable(synthetic_eval_set_table_uc_fqn+"_eval_metrics")
)

# You can click on a row in the `trace` column to view the detailed MLflow trace
display(per_question_results_pdf)

# COMMAND ----------

from time import sleep

total_reqs = payload_count + len(synthetic_data_pdf)
minutes = 0
while payload_count < total_reqs and minutes < 10: # TODO: fix this logic
    sleep(30)
    payload_count = spark.table(f"{model_fqdn}_payload").count()
    minutes += 0.5

payload_df = spark.table(f"{model_fqdn}_payload")
display(payload_df)

# COMMAND ----------

# DBTITLE 1,Send Inferences to App for Human Eval
from pyspark.sql.functions import col, lit, concat, length, expr

payload_df = (
    payload_df # filter down to requests that don't meet end users' criteria for SME review
    .withColumn("content", expr("request:messages[0]:content"))
    .where(~(col("content").contains("1.") & col("content").contains("2."))) 
    .where(length(col("content")) > 200)  
    .drop("content")
)
request_ids = payload_df.select("databricks_request_id").rdd.flatMap(lambda x: x).collect()
agents.enable_trace_reviews(model_name=model_fqdn, request_ids=request_ids)

# COMMAND ----------


