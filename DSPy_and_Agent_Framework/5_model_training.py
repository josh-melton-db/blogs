# Databricks notebook source
# MAGIC %pip install langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 
# MAGIC %pip install --upgrade databricks-vectorsearch databricks-sdk databricks-agents
# MAGIC %pip install --upgrade databricks-genai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml

with open("rag_config.yaml", "r") as file:
    rag_config = yaml.safe_load(file)

model_fqdn = rag_config.get("demo_config").get("model_fqdn")
synthetic_eval_set_table_uc_fqn = rag_config.get("demo_config").get("synthetic_eval_set_table_uc_fqn")
ft_table = rag_config.get("demo_config").get("source_table")+"_raft"
num_docs = rag_config.get("vector_search_parameters").get("k")

# COMMAND ----------

import random

chunk_column_name = rag_config.get("chunk_column_name")
document_source_id = rag_config.get("document_source_id")
chunk_table = rag_config.get("demo_config").get("chunk_table")
chunk_df = spark.read.table(chunk_table)
chunk_list = chunk_df.select("issue_description_chunk").rdd.flatMap(lambda x: x).collect()
prompt_ls = rag_config.get("chat_prompt_template").replace("{question}.", "").split("{context}")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)
index = vsc.get_index(endpoint_name=rag_config.get("vector_search_endpoint_name"), index_name=rag_config.get("vector_search_index"))

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

@udf(returnType=StringType())
def get_raft_docs_udf(query_text, n_retrieve_docs, n_adversarial_docs):
    retrieved = index.similarity_search(
        query_text=query_text, 
        columns=[chunk_column_name],
        num_results=n_retrieve_docs
    )['result']['data_array']
    retrieve_docs = [result[0] for result in retrieved]
    adversarial_docs = random.sample(chunk_list, n_adversarial_docs)
    return " ".join(retrieve_docs + adversarial_docs)

# COMMAND ----------

from pyspark.sql.functions import col, lit, concat, length

ft_data_df = ( # Return data that the system had trouble on according to our judge or our user expectations
    spark.read.table(synthetic_eval_set_table_uc_fqn+"_eval_metrics")
    .where(
        (col("`response/llm_judged/relevance_to_query/rating`") == 'no') |
        (col("`response/llm_judged/correctness/rating`") == 'no') |
        (col("`response/llm_judged/safety/rating`") == 'no') | 
        ~(col("response").contains("1.") & col("response").contains("2."))
    ) # Add distractor documents as described in RAFT paper
    .withColumn("raft_docs", get_raft_docs_udf(col("response"), lit(num_docs), lit(1))) 
    .withColumn("prompt", concat(lit(prompt_ls[0]), col("raft_docs"), lit(prompt_ls[1]), col("request")))
    .select("prompt", "expected_response")
    .withColumnRenamed("expected_response", "response")
)
ft_data_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(ft_table)
ft_data_df.display()

# COMMAND ----------

from databricks.model_training import foundation_model as fm
#Return the current cluster id to use to read the dataset and send it to the fine tuning cluster. See https://docs.databricks.com/en/large-language-models/foundation-model-training/create-fine-tune-run.html#cluster-id
def get_current_cluster_id():
  import json
  return json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().safeToJson())['attributes']['clusterId']

#Let's use a new model name
registered_model_name = model_fqdn + "_ft"

run = fm.create(
    data_prep_cluster_id=get_current_cluster_id(),  # required if you are using delta tables as training data source. This is the cluster id that we want to use for our data prep job.
    model="meta-llama/Meta-Llama-3-8B-Instruct",  # Here we define what model we used as our baseline
    train_data_path=ft_table,
    task_type="INSTRUCTION_FINETUNE",  # task_type="CHAT_COMPLETION" or "CONTINUED_PRETRAIN" also supported
    register_to=registered_model_name,
    training_duration="10ep", # only 10 epochs. Check the mlflow experiment metrics to see if you should increase this number
    learning_rate="5e-7", # small learning rate to not overfit to our small dataset
)
# For production use cases, use Provisioned Throughput APIs
# https://docs.databricks.com/en/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis.html#get-provisioned-throughput-in-increments

print(run)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, AutoCaptureConfigInput
from utils.demo import wait_for_run_to_finish, get_latest_model_version

# wait_for_run_to_finish(run) # TODO: fix this, spins forever

source_table_ls = rag_config.get("demo_config").get("source_table").split(".")
catalog = source_table_ls[0]
schema = source_table_ls[1]
serving_endpoint_name = rag_config.get("demo_config").get("endpoint_name") + "_ft"
w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=registered_model_name, 
            entity_version=get_latest_model_version(registered_model_name),
            workload_size="Small",
            workload_type="GPU_MEDIUM",
            scale_to_zero_enabled=True 
        )
    ],
    auto_capture_config = AutoCaptureConfigInput(catalog_name=catalog, schema_name=schema, enabled=True, table_name_prefix="ft_log")
)

force_update = False #Set this to True to release a newer version (the demo won't update the endpoint to a newer model version by default)
existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_name}, this will take a while to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config) # TODO: catch the timeout error
else:
  print(f"endpoint {serving_endpoint_name} already exist...")
  if force_update:
    w.serving_endpoints.update_config_and_wait(served_entities=endpoint_config.served_entities, name=serving_endpoint_name)

# COMMAND ----------

import requests
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()


def ft_model_inference(prompt):
    data = {
        "inputs": {"prompt": [prompt]},
        "params": rag_config.get("chat_model_parameters")
    }
    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(url=f"{API_ROOT}/serving-endpoints/{serving_endpoint_name}/invocations", json=data, headers=headers)
    return response.json()["predictions"][0]["candidates"][0]["text"] # TODO: catch the missing field

ft_pdf = (
    ft_data_df
    .withColumnRenamed("response", "expected_response")
    .withColumnRenamed("prompt", "request")
).toPandas()
ft_pdf["response"] = ft_pdf["request"].apply(ft_model_inference)
display(ft_pdf)

# COMMAND ----------

import mlflow
with mlflow.start_run():
    eval_results = mlflow.evaluate(
        data=ft_pdf,
        model_type="databricks-agent"
    )

eval_results.metrics

# COMMAND ----------

# from pyspark.sql.functions import row_number, expr, desc
# from pyspark.sql.window import Window

# windowSpec = Window.partitionBy("initial_request_id").orderBy(desc("timestamp_ms"))
# suggestions_df = (
#     spark.read.table(f"{model_fqdn}_payload")
#     .withColumn("initial_request_id", expr("request:dataframe_records[0]:request_id"))
#     .withColumn("rownum", row_number().over(windowSpec))
#     .where("rownum = 1") # TODO: is this grabbing just one record from each request for human evals?
#     .withColumn("suggested_output", expr("request:dataframe_records[0]:text_assessments[0]:suggested_output"))
# )
# suggestions_df.display()

# COMMAND ----------

# from pyspark.sql.functions import from_json, col, array, expr

# payload_df = (
#     spark.table(f"{model_fqdn}_payload")
#     .withColumn("parsed_request", expr("response:databricks_output:trace:data:request:messages[0]"))
#     .withColumn("parsed_response", expr("response:choices[0]:message"))
#     .where("parsed_request is not null and parsed_response is not null")
#     .withColumn("messages", array(col("parsed_request"), col("parsed_response")))
#     .select("messages")
# )
# payload_df.display()

# COMMAND ----------


