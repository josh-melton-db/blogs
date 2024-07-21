import yaml
from databricks.sdk import WorkspaceClient
import re
import random
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import concat, col, named_struct, lit, array, expr, regexp_replace, lower
import os
from openai import OpenAI
import requests
import concurrent.futures
import json
import time

# TODO: reduce number of args - use kwargs, dictionary
def get_config(catalog, source_schema, source_table_name, source_id_name, source_column_name, vs_endpoint,
               target_schema=None, num_docs=4, chunk_size_tokens=300, chunk_overlap_tokens=100,
               embedding_endpoint="databricks-gte-large-en", chat_model="databricks-meta-llama-3-70b-instruct", chat_prompt=None,
               mlflow_run_name="generated_rag_demo", rag_app_name="generated_rag_demo"):
    w = WorkspaceClient()
    username = w.current_user.me().user_name.replace('@', '_').replace('.', '_')
    source_table = f"{catalog}.{source_schema}.{source_table_name}"
    chunk_table = f"{catalog}.{source_schema}.{source_table_name}_chunked"
    chunk_column_name = source_column_name + "_chunk"
    chunk_id_column_name = "chunk_id"
    index_name = f"{catalog}.{source_schema}.{source_table_name}_index" # TODO: unique per user
    # TODO: automatically retrieve vs endpoint via sdk?
    # TODO: model serving endpoint name is too long

    if not target_schema:
        target_schema = source_schema
    model_name = "rag_chain_model"
    model_fqdn = f"{catalog}.{target_schema}.{model_name}"
    endpoint_name = f"agents_{model_fqdn}".replace(".", "-")
    synthetic_eval_set_table_uc_fqn = f"{catalog}.{target_schema}.synthetic_eval_set"
    inference_table_uc_fqn = f"{catalog}.{target_schema}.`agents-{model_name}_payload`" 
    request_log_output_uc_fqn = f"{catalog}.{target_schema}.{model_name}_request_log"
    assessment_log_output_uc_fqn = f"{catalog}.{target_schema}.{model_name}_assessment_log"
    if not chat_prompt:
        chat_prompt = "You are a trusted assistant that helps answer questions based only on the provided information. If you do not know the answer to a question, you truthfully say you do not know.  Here is some context which might or might not help you answer: {context}.  Answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer, do not mention the context or the question. Based on this history and context, answer this question: {question}."
    prompt_vars = ["context", "question"]
    chunk_template = "`{chunk_text}`"

    return dict(
        embedding_endpoint = embedding_endpoint, 
        document_source_id = source_id_name,
        vector_search_endpoint_name = vs_endpoint,
        vector_search_index = index_name,
        chunk_column_name = chunk_column_name,
        chunk_id_column_name = chunk_id_column_name,
        chunk_template = chunk_template,
        chunk_size = chunk_size_tokens, 
        chunk_overlap = chunk_overlap_tokens, 
        chat_prompt_template = chat_prompt,
        chat_prompt_template_variables = prompt_vars,
        vector_search_parameters = dict(
            k = num_docs, 
        ),
        chat_endpoint = chat_model,
        chat_model_parameters = dict(
            temperature = 0.01,
            max_tokens = 500,
        ),
        demo_config = dict(
            source_table = source_table,
            source_column_name = source_column_name,
            chunk_table = chunk_table,
            target_schema = target_schema,
            model_fqdn = model_fqdn,
            endpoint_name = endpoint_name,
            synthetic_eval_set_table_uc_fqn = synthetic_eval_set_table_uc_fqn,
            inference_table_uc_fqn = inference_table_uc_fqn,
            request_log_output_uc_fqn = request_log_output_uc_fqn,
            assessment_log_output_uc_fqn = assessment_log_output_uc_fqn,
            mlflow_run_name = mlflow_run_name, 
            rag_app_name = rag_app_name
        ),
    )

def save_config(dbutils, rag_config, fname="rag_config.yaml"):
    dbx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_path = dbx.notebookPath().get()
    folder_path = '/'.join(str(x) for x in notebook_path.split('/')[:-1])
    with open(f"/Workspace/{folder_path}/{fname}", 'w') as outfile:
        yaml.dump(rag_config, outfile, default_flow_style=False)

def generate_category(chat_model, text_domain, category_ls):
    category_prompt = "Given the domain '{text_domain}', generate a classification or category for a piece of text in the domain. For example, if the domain was 'airplane pilot notes' a category might be 'control panel malfunction'. Come up with a category different from the following, if available: {category_ls}. Give only the category, in three words or less, no description, no filler, nothing about a response, ONLY THE CATEGORY:"
    category = chat_model.predict(category_prompt.format(text_domain=text_domain, category_ls=category_ls))
    category_ls.append(category)
    return category_ls

def generate_categories(chat_model, text_domain, category_ls):
    while len(category_ls) < 25:
        category_ls = generate_category(chat_model, text_domain, category_ls)
    cleaned_categories = [re.sub(r'[^a-zA-Z\s]', '', category) for category in category_ls]
    return list(set(cleaned_categories))

def generate_symptom(chat_model, domain, category, symptom_ls):
    symptom_prompt = "Generate an issue set for the category '{category}' within the domain '{domain}'. For example, if the domain was 'airplane pilot notes' and the category was 'control panel malfunction' an issue set might be 'altitude gauge showing irregular readings'. Come up with a symptom different from the following, if available: {symptom_ls}. Give only the symptom, in ten words or less, no description, no filler, nothing like a numbered list, ONLY THE SYMPTOM:"
    symptom = chat_model.predict(symptom_prompt.format(category=category, domain=domain, symptom_ls=symptom_ls))
    symptom_ls.append(symptom)
    return symptom

def generate_symptoms(chat_model, categories, text_domain):
    symptoms_sets = {}
    for category in categories:
        symptom_ls = []
        num_documents = random.randint(7, 12)
        for _ in range(num_documents):
            symptom = generate_symptom(chat_model, text_domain, category, symptom_ls)
            symptom_ls.append(symptom)
        symptoms_sets[category] = list(set(symptom_ls))
    cleaned_symptoms = {}
    for category in symptoms_sets.keys():
        cleaned_symptoms[category] = [re.sub(r'[^a-zA-Z\s]', '', symptom_set) for symptom_set in symptoms_sets[category]]
    return cleaned_symptoms

def generate_document(chat_model, issues, category, document_ls):
    document_prompt = "Given the issue set {issues}, generate a piece of text reporting the issues in detail. Indicate some relationship to {category}, although not directly. Indicate whether you think there is a potential resolution to the problem. Use an objective, fact-based, expert perspective. Give only the text, in one hundred words or less, no filler, nothing to indicate you're not the expert writing notes, don't explicitly say you were given a category or restate the category, no lists, only detailed notes and reporting of the issues"
    document = chat_model.predict(document_prompt.format(issues=issues, category=category))
    document_ls.append(document)
    return document_ls

def generate_documents(chat_model, symptom_sets):
    data_dict = {}
    for category in symptom_sets.keys():
        document_ls = []
        for symptoms in symptom_sets[category]:
            doc = generate_document(chat_model, symptoms, category, document_ls)
        data_dict[category] = document_ls
    return data_dict

def create_df(spark, data_dict, text_col_name):
    data = [(category, item) for category, items in data_dict.items() for item in items]
    schema = StructType([
        StructField("category", StringType(), True),
        StructField(text_col_name, StringType(), True)
    ])
    return spark.createDataFrame(data, schema=schema)

def reset_tables(spark, catalog, schema, target_schema, tried=False):
    try:
        spark.sql(f"drop schema if exists {catalog}.{schema} CASCADE")
        spark.sql(f"drop schema if exists {catalog}.{target_schema} CASCADE")
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{target_schema}")
    except Exception as e:
        if 'NO_SUCH_CATALOG_EXCEPTION' in str(e) and not tried:
                spark.sql(f'create catalog {config["catalog"]}')
                reset_tables(spark, catalog, schema, True)
        else:
            raise

def generate_source_data(chat_model, text_domain, category_ls, text_col_name, text_id_name, catalog, schema, table, spark):
    print('Generating data, this may take a couple minutes')
    categories = generate_categories(chat_model, text_domain, category_ls)
    symptoms = generate_symptoms(chat_model, categories, text_domain)
    documents = generate_documents(chat_model, symptoms) 
    df = create_df(spark, documents, text_col_name)
    df = df.withColumn("issue_description", regexp_replace(col("issue_description"), col("category"), ""))
    df = df.withColumn("issue_description", regexp_replace(lower(col("issue_description")), lower(col("category")), ""))
    df = df.withColumn("issue_description", regexp_replace(col("issue_description"), ":", ""))
    df = df.withColumn(text_id_name, expr("substring(md5(cast(rand() as string)), 1, 7)"))

    source_table = f"{catalog}.{schema}.{table}"
    df.write.saveAsTable(source_table)

def check_dbr(spark):
    dbr_majorversion = int(spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion").split(".")[0])
    return dbr_majorversion >= 14

def get_table_url(table_fqdn, dbutils):
    split = table_fqdn.split(".")
    url = f"https://{dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()}/explore/data/{split[0]}/{split[1]}/{split[2]}"
    return url

from mlflow.utils import databricks_utils as du
os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = "false"

# # Temporary workarounds given the Private Preview state of the product
# def parse_deployment_info(deployment_info):
#     browser_url = du.get_browser_hostname()
#     message = f"""Deployment of {deployment_info.model_name} version {deployment_info.model_version} initiated.  This can take up to 15 minutes and the Review App & REST API will not work until this deployment finishes. 

#     View status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}
#     Review App: {deployment_info.rag_app_url}"""
#     return message

def generate_question(chunk_row, chunk_text_key, model_endpoint, root, token):
    os.environ["OPENAI_API_KEY"] = token
    os.environ["OPENAI_BASE_URL"] = f"{root}/serving-endpoints/"
    # source: https://thetechbuffet.substack.com/p/evaluate-rag-with-synthetic-data
    PROMPT_TEMPLATE = """\ 
    Your task is to formulate exactly 1 question from given context.

    The question must satisfy the rules given below:
    1.The question should make sense to humans even when read without the given context.
    2.The question should be fully answered from the given context.
    3.The question should be framed from a part of context that contains important information. It can also be from tables,code,etc.
    4.The answer to the question should not contain any links.
    5.The question should be of moderate difficulty.
    6.The question must be reasonable and must be understood and responded by humans.
    7.Do no use phrases like 'provided context', 'context', etc in the question
    8.Avoid framing question using word "and" that can be decomposed into more than one question.
    9.The question should not contain more than 10 words, make of use of abbreviation wherever possible.
        
    context: {context}"""

    q_system_prompt = "You are an expert at resolving customer issues.  You are also an expert at generating questions that a human would likely ask about specific content from the issues. You pride yourself on your ability to be realistic, yet a bit creative, and you know that a human will evaluate your output, so you put extra effort into following instructions exactly. Do not use leading numbers. DO NOT REFER TO THE INSTRUCTIONS OR CONTEXT DIRECTLY, DO NOT PROVIDE AN ANSWER."
    client = OpenAI()
    prompt = PROMPT_TEMPLATE.format(context=chunk_row[chunk_text_key])
    question = client.chat.completions.create(
                model=model_endpoint,
                messages=[
                    {"role": "system", "content": q_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0
            ).choices[0].message.content.replace("1. ", "")
    
    a_system_prompt = "You are an expert at summarizing customer issues. You are also an expert at synthesizing information into short, numbered lists. You know that a human will evaluate your output, so you put extra effort into following instructions exactly. DO NOT REFER TO THE INSTRUCTIONS OR PROVIDE ANY INFORMATION BESIDES THE ANSWER"
    prompt = f"""Given the question and the context, provide the answer as concisely as possible. Leverage the context heavily to guide your answer. Use a numbered list like 1. <information> \n2. <information \n3...
    context: {chunk_row[chunk_text_key]}
    question: {question}
    """
    answer = client.chat.completions.create(
                model=model_endpoint,
                messages=[
                    {"role": "system", "content": a_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            ).choices[0].message.content
    return {"question": question, "answer": answer}

def process_one_chunk(row, chunk_text_key, chunk_id_key, root, token):
    MAX_TRIES = 4
    model_endpoint = "databricks-dbrx-instruct"
    tries = 0
    try: 
        gen_questions = generate_question(row, chunk_text_key, model_endpoint, root, token)
        while "chunk" in gen_questions and tries < MAX_TRIES:
            tries = tries + 1
            gen_questions = generate_question(row, chunk_text_key, model_endpoint, root, token)
        out_data = {}
        out_data["expected_retrieved_context"] = [{"doc_uri": row[chunk_id_key]}]
        out_data["request"] = gen_questions["question"]
        out_data["expected_response"] = gen_questions["answer"]
        return out_data
    except Exception as e:
        print(f"failed to parse output for doc {row[chunk_id_key]}\n", e)

def generate_questions(chunks_df, chunk_text_key, chunk_id_key, dbutils):
    NUM_THREADS = 7
    root = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    parsed_json_chunks = []
    json_df = chunks_df.toJSON().collect()
    for row in json_df:
        parsed_row = json.loads(row)
        parsed_json_chunks.append(parsed_row)
    # Create a ThreadPoolExecutor, wait for all tasks to complete and get the results
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(process_one_chunk, row, chunk_text_key, chunk_id_key, root, token) for row in parsed_json_chunks]
        return [future.result() for future in concurrent.futures.as_completed(futures)]

def query_chain(question, endpoint_name, dbutils):
    root = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    model_input_sample = {
        "messages": [{
            "role": "user",
            "content": question,
        }]
    }
    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {token}"}
    url = f"{root}/serving-endpoints/{endpoint_name}/invocations"
    response = requests.post(url=url, json=model_input_sample, headers=headers)
    return response.json()

#Helper fuinction to Wait for the fine tuning run to finish
def wait_for_run_to_finish(run):
  print_train = False
  for i in range(300):
    events = run.get_events()
    for e in events:
      if "FAILED" in e.type or "EXCEPTION" in e.type:
        raise Exception(f'Error with the fine tuning run, check the details in run.get_events(): {e}')
    if events[-1].type == 'TRAIN_FINISHED':
      print('Run finished')
      return events
    if i % 30 == 0:
      print(f'waiting for run {run.name} to complete...')
    if events[-1].type == 'TRAIN_UPDATED' and not print_train:
      print_train = True
      display(events)
    time.sleep(10)

def get_latest_model_version(model_name):
    from mlflow.tracking import MlflowClient
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version