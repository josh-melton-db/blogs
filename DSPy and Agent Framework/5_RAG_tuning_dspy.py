# Databricks notebook source
# MAGIC %md
# MAGIC In this notebook, we'll take the data we curated with RAG Studio and fine tune an AI system using [DSPy](https://dspy-docs.vercel.app/), an open source framework for "programming - not prompting - language models". The aim is to eliminate brittle attachments to models or prompts by defining the process of our AI system, and allowing AI to optimize it for us. We can swap new models in and out for various pieces of the system and be confident our results are efficient and accurate without hand-tuning prompting techniques or relying on subjective, imprecise evaluations of prompting techniques.

# COMMAND ----------

# DBTITLE 1,Install Libraries
# MAGIC %pip install dspy-ai mlflow --upgrade -q # transformers
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

# DBTITLE 1,Setup
import yaml
with open("rag_config.yaml", "r") as file:
    rag_config = yaml.safe_load(file)
synthetic_eval_set_table_uc_fqn = rag_config.get("demo_config").get("synthetic_eval_set_table_uc_fqn")
index_name = rag_config.get("vector_search_index")
doc_id = rag_config.get("document_source_id")
chunk_column = rag_config.get("chunk_column_name")
assessment_log_output_uc_fqn = rag_config.get("demo_config").get("assessment_log_output_uc_fqn")
model_fqdn = rag_config.get("demo_config").get("model_fqdn")

# COMMAND ----------

# DBTITLE 1,Set DSPy Models
import dspy
from dspy.retrieve.databricks_rm import DatabricksRM

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# Set up the models
lm = dspy.Databricks(model="databricks-mpt-7b-instruct", model_type="completions", api_key=token, 
                     api_base=url + '/serving-endpoints', max_tokens=1000)
# lm = dspy.HFModel(model='apple/OpenELM-450M-Instruct') # Use this to run a local HF model like Apple's super small ELM series
judge = dspy.Databricks(model="databricks-dbrx-instruct", model_type="chat", api_key=token, 
                       api_base=url + "/serving-endpoints", max_tokens=200)
dspy.settings.configure(lm=lm)

# COMMAND ----------

# DBTITLE 1,Define our Respond Signature
class Respond(dspy.Signature):
    """Generates a response to the request given some context"""
    request = dspy.InputField(desc="Request from an end user")
    context = dspy.InputField(desc="Context retrieved from vector search")
    response = dspy.OutputField(desc="Numbered response to the user's question given the retrieved context")

# COMMAND ----------

# DBTITLE 1,Create DSPy Module
class RAG(dspy.Module):
    """Generates a response to the request using retrieved input for grounding"""
    def __init__(self):
        super().__init__()
        self.retrieve = DatabricksRM( # Set up retrieval from our vector search
            databricks_index_name=index_name,
            databricks_endpoint=url, 
            databricks_token=token,
            columns=["category", doc_id, chunk_column],
            text_column_name=chunk_column,
            docs_id_column_name=doc_id,
            k=2
        )
        self.respond = dspy.ChainOfThought(Respond) # Responses will use chain of thought, i.e. "think this through step by step..."

    def forward(self, request):
        context = self.retrieve(request, query_type="text").docs
        return self.respond(request=request, context=str(context))

# COMMAND ----------

golden_dataset = spark.read.table(synthetic_eval_set_table_uc_fqn+"_eval_metrics").toPandas()

# Filter down to rows that pass our quality checks
golden_dataset = golden_dataset[
    (golden_dataset["response/llm_judged/safety/rating"] == 'yes') &
    (golden_dataset["response/llm_judged/groundedness/rating"] == 'yes') &
    (golden_dataset["response/llm_judged/relevance_to_query/rating"] == 'yes') &
    (golden_dataset["response"].str.contains("1."))
]

# Concatenate the 'retrieved_context' column together
golden_dataset["retrieved_context"] = golden_dataset["retrieved_context"].apply(lambda x: ";\n".join([obj["content"] for obj in x]))
display(golden_dataset)

# COMMAND ----------

# DBTITLE 1,Create Datasets
golden_dataset.sort_values(by="agent/total_output_token_count", inplace=True) # Shorter examples will be used for training
split = int(len(golden_dataset) * .7)
trainset = [dspy.Example(request=row['request'], retrieved_context=row['retrieved_context'], response=row['response']).with_inputs('request')
           for i, row in golden_dataset[:split].iterrows()]
testset = [dspy.Example(request=row['request'], retrieved_context=row['retrieved_context'], response=row['response']).with_inputs('request')
           for i, row in golden_dataset[split:].iterrows()]

# COMMAND ----------

# DBTITLE 1,Define Assessment
class AssessResponse(dspy.Signature):
    """Assess the quality of an outline along the specified dimension."""
    request = dspy.InputField()
    response_to_assess = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

# COMMAND ----------

# DBTITLE 1,Define Metric
import mlflow

def metric(gold, pred, trace=None):
    request, context = gold.request, gold.retrieved_context
    response = pred.response
    with dspy.context(lm=judge):
        specific_q = "Does the response provide a very detailed, specific response to the request?"
        value_add_q = "Does the response avoid simply repeating back the provided request and add value to the conversation?"
        not_duplicative_q = "Are the items in the list of responses unique rather than repetitive?"
        listed_q = "Is the response formatted in a bulleted or numbered list?"
        specific_eval =  dspy.Predict(AssessResponse)(request=request, response_to_assess=response, assessment_question=specific_q)
        value_add_eval = dspy.Predict(AssessResponse)(request=request, response_to_assess=response, assessment_question=value_add_q)
        not_duplicative_eval = dspy.Predict(AssessResponse)(request=request, response_to_assess=response, assessment_question=not_duplicative_q)
        listed_eval = dspy.Predict(AssessResponse)(request=request, response_to_assess=response, assessment_question=listed_q)
        length_eval = max((600 - len(response)) / 150, 0) # Our users asked for short, numbered lists, so we'll incentivize that behavior 


    evals = [specific_eval, value_add_eval, not_duplicative_eval, listed_eval]
    results = ['yes' in m.assessment_answer.lower() for m in evals]
    score = (sum(results) + length_eval) / (len(evals) + 4) # Total score over total possible score
    if trace is not None: # When training, we'll only return positive signal at a high score
        print(results, length_eval, evals)
        return score > .6
    return score

# COMMAND ----------

def evaluate(testset, system):
    scores = []
    for x in testset:
        pred = system(**x.inputs())
        score = metric(x, pred)
        scores.append(score)
    return sum(scores) / len(testset)

# COMMAND ----------

baseline_score = evaluate(testset, RAG())
print("Baseline score:    ", baseline_score)

# COMMAND ----------

with dspy.context(lm=judge): 
    judge_score = evaluate(testset, RAG())
print("Judge score:    ", judge_score)

# COMMAND ----------

# DBTITLE 1,Optimize DSPy Module
from dspy.teleprompt import BootstrapFewShotWithOptuna # Which optimizer you use depends on the number of examples you've curated

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) few-shot examples of our CoT program.
config = dict(max_rounds=5)

# Use the proprietary training dataset you've collected and labelled with Agent Framework to
# optimize your model. The metric is going to tell the optimizer how well it's doing
optimizer = BootstrapFewShotWithOptuna(metric=metric, teacher_settings=dict({'lm': judge}), **config)
optimized_rag = optimizer.compile(student=RAG(), trainset=trainset, max_demos=5)

# COMMAND ----------

optimized_score = evaluate(testset, optimized_rag)
print('BootstrapFewShotWithOptuna score:  ', optimized_score)

# COMMAND ----------

# DBTITLE 1,Compare Metrics
print("% Improvement over raw:      ", 100*(optimized_score - baseline_score) / baseline_score)
print("% Comparison to DBRX:        ", 100*(optimized_score - judge_score) / judge_score)

# COMMAND ----------

# MAGIC %md
# MAGIC On average, the DSPy optimized MPT-7b system scores noticably higher than the baseline and comparable performance the un-optimized DBRX model (which is over 4x more expensive per output token). Alternatively, you could optimize a DBRX-powered system to deliver significantly improved performance for a more complex system or metric. Whether you aim for greater accuracy or reduced cost, you've used the data curated by Agent Framework to develop a proprietary improvement to the ROI of your AI systems!
