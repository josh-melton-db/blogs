# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents dspy-ai mlflow mlflow-skinny
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.__version__

# COMMAND ----------

import yaml

with open("rag_config.yaml", "r") as file:
    rag_config = yaml.safe_load(file)

model_fqdn = rag_config.get("demo_config").get("model_fqdn")
synthetic_eval_set_table_uc_fqn = rag_config.get("demo_config").get("synthetic_eval_set_table_uc_fqn")
ft_table = rag_config.get("demo_config").get("source_table")+"_raft"

# COMMAND ----------

import dspy

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# Set up the model
lm = dspy.Databricks(model="databricks-dbrx-instruct", model_type="chat", api_key=token, 
                       api_base=url + "/serving-endpoints", max_tokens=200)
ft_lm = ...
judge = ...
dspy.settings.configure(lm=lm)

# COMMAND ----------

from dspy.retrieve.databricks_rm import DatabricksRM

rm = DatabricksRM(
    databricks_index_name=rag_config.get("vector_search_index"),
    databricks_endpoint=url,
    databricks_token=token,
    columns=[rag_config.get("chunk_column_name")],
    text_column_name=rag_config.get("chunk_column_name"),
    docs_id_column_name=rag_config.get("document_source_id"), # TODO: return doc id too, use in eval
    k=rag_config.get("vector_search_parameters").get("k")
)
rm(query="Transportation and logistic issues", query_type="text")

# COMMAND ----------

display(spark.read.table(synthetic_eval_set_table_uc_fqn+"_eval_metrics"))

# COMMAND ----------

synthetic_eval_set_table_uc_fqn = rag_config.get("demo_config").get("synthetic_eval_set_table_uc_fqn")
golden_dataset = spark.read.table(synthetic_eval_set_table_uc_fqn+"_eval_metrics").toPandas()
train_cutoff = 1 # int(len(golden_dataset) * .6)
dataset = [dspy.Example(request=row['request'], response=row['expected_response'], 
                        context=row['expected_retrieved_context']).with_inputs('request') 
           for i, row in golden_dataset.iterrows()]
trainset = dataset[:train_cutoff]
testset = dataset[train_cutoff:]

# COMMAND ----------

class QuestionAnswer(dspy.Signature):
    """Returns the answer to the question"""
    request = dspy.InputField(desc="A question about our business")
    response = dspy.OutputField(desc="A short, numbered answer to the question")

class QueryExpand(dspy.Signature):
    """Rephrases the question to increase the quality of the response"""
    question = dspy.InputField(desc="A question about our business")
    expanded_question = dspy.OutputField(desc="An expanded question")

class QuestionAnswerAgent(dspy.Module):
    def __init__(self, rm):
        super().__init__()
        self.rm = rm
        self.query_expander = dspy.ChainOfThought(QueryExpand)
        self.answer_cot = dspy.ChainOfThought(QuestionAnswer) # stitch together the RAG chain

    def forward(self, request):
        expanded_question = self.query_expander(question=request).expanded_question
        docs = " ".join(self.rm(expanded_question).docs)
        return self.answer_cot(request=request+docs)

qa_agent = QuestionAnswerAgent(rm)
pred = qa_agent(trainset[0].request)
pred

# COMMAND ----------

import mlflow
import pandas as pd

# mlflow.autolog(disable=True)
# mlflow.disable_system_metrics_logging()

# Define the metric for optimization (e.g., accuracy)
def accuracy_metric(gold, pred, trace=None):
    eval_output = [{
            "request": gold.request,
            "response": pred.response,
            "expected_response": gold.response
    }]
    eval_metrics = mlflow.evaluate(
        data=pd.DataFrame(eval_output),
        model_type="databricks-agent"
    ).metrics
    evals = [
        eval_metrics["response/llm_judged/safety/rating/percentage"],
        eval_metrics["response/llm_judged/correctness/rating/percentage"],
        eval_metrics["response/llm_judged/correctness/rating/percentage"]
    ]
    return sum(evals) / len(evals)

# COMMAND ----------

def calculate_score(predictions, testset):
    evals = []
    for i, prediction in enumerate(predictions):
        evals.append(accuracy_metric(trainset[i], prediction))
    return sum(evals) / len(evals)

unoptimized_predictions = [qa_agent.forward(example.request) for example in trainset]
unoptimized_score = calculate_score(unoptimized_predictions, testset)

# COMMAND ----------

from dspy.teleprompt import MIPRO


# Initialize the optimizer
optimizer = MIPRO(metric=accuracy_metric)

# Optimize the program
kwargs = dict(num_threads=4, display_progress=True, display_table=0)
optimized_qa_agent = optimizer.compile(
    student=qa_agent,
    trainset=trainset,
    num_trials=1,
    max_bootstrapped_demos=1,
    max_labeled_demos=1,
    eval_kwargs=kwargs,
    requires_permission_to_run=False
)

# COMMAND ----------

# Evaluate the optimized program
optimized_predictions = [optimized_qa_agent.forward(example.request) for example in testset]
optimized_score = calculate_score(optimized_predictions, testset)
print(f"Optimized Program Accuracy: {optimized_score * 100:.2f}%")

# COMMAND ----------


