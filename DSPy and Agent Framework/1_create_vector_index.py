# Databricks notebook source
# MAGIC %pip install -U --quiet databricks-sdk langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 tokenizers transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, ResourceDoesNotExist
from databricks.sdk.service.vectorsearch import (
    EndpointType,
    DeltaSyncVectorIndexSpecRequest,
    VectorIndexType,
    EmbeddingSourceColumn,
    PipelineType,
    EndpointStatusState
)
import pyspark.sql.functions as func
from pyspark.sql.types import MapType, StringType
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from pyspark.sql.types import *
from typing import List
from utils.demo import check_dbr
import yaml

# COMMAND ----------

# DBTITLE 1,Set Configuration
with open("rag_config.yaml", "r") as file:
    rag_config = yaml.safe_load(file)
index_name = rag_config.get("vector_search_index")
embedding_endpoint = rag_config.get("embedding_endpoint")
vector_search_endpoint_name = rag_config.get("vector_search_endpoint_name")
chunk_column_name = rag_config.get("chunk_column_name")
chunk_size = rag_config.get("chunk_size")
chunk_overlap = rag_config.get("chunk_overlap")
demo_config = rag_config.get("demo_config")

source_table = demo_config.get("source_table")
source_column_name = demo_config.get("source_column_name")
chunk_table = demo_config.get("chunk_table")

# COMMAND ----------

# DBTITLE 1,Define Tokenizer Function
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
useArrow = check_dbr(spark)

@func.udf(returnType=ArrayType(StringType()), useArrow=useArrow)
def split_char_recursive(content: str) -> List[str]:
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(content)
    return [doc for doc in chunks]

# COMMAND ----------

# DBTITLE 1,Chunk Docs
chunked_docs = (
    spark.read.table(source_table)
    .select("*", func.explode(split_char_recursive(func.col(source_column_name))).alias(chunk_column_name))
    .select("*", func.md5(func.col(chunk_column_name)).alias("chunk_id"))
)
chunked_docs.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(chunk_table)
spark.sql(f"ALTER TABLE {chunk_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)") # Required for Vector Search
chunked_docs.display()

# COMMAND ----------

# DBTITLE 1,Create Vector Search Index
w = WorkspaceClient()

# If index already exists, re-sync
try:
    w.vector_search_indexes.sync_index(index_name=index_name)
# Otherwise, create new index
except ResourceDoesNotExist as ne_error:
    w.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=vector_search_endpoint_name,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    embedding_model_endpoint_name=embedding_endpoint,
                    name=chunk_column_name,
                )
            ],
            pipeline_type=PipelineType.TRIGGERED,
            source_table=chunk_table,
        ),
    )

# COMMAND ----------

print("Vector index:\n")
print(w.vector_search_indexes.get_index(index_name).status.message)

# COMMAND ----------


