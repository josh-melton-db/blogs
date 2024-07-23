# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents dspy-ai mlflow mlflow-skinny
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml

with open("rag_config.yaml", "r") as file:
    rag_config = yaml.safe_load(file)

model_fqdn = rag_config.get("demo_config").get("model_fqdn")
synthetic_eval_set_table_uc_fqn = rag_config.get("demo_config").get("synthetic_eval_set_table_uc_fqn")
ft_table = rag_config.get("demo_config").get("source_table")+"_raft"
serving_endpoint_name = rag_config.get("demo_config").get("endpoint_name") + "_ft"

# COMMAND ----------

import dspy

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# Set up the model
lm = dspy.Databricks(model="databricks-dbrx-instruct", model_type="chat", api_key=token, 
                       api_base=url + "/serving-endpoints", max_tokens=600) # Use different model as teacher/judge
ft_lm = dspy.Databricks(model=serving_endpoint_name, model_type="text", api_key=token, 
                        api_base=url + "/serving-endpoints", max_tokens=600) # Using a smaller and cheaper fine tuned model
judge = dspy.Databricks(model="databricks-meta-llama-3-70b-instruct", model_type="chat", api_key=token, 
                       api_base=url + "/serving-endpoints", max_tokens=600) # Test 70b parameter model
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

synthetic_eval_set_table_uc_fqn = rag_config.get("demo_config").get("synthetic_eval_set_table_uc_fqn")
golden_dataset = spark.read.table(synthetic_eval_set_table_uc_fqn+"_eval_metrics").toPandas()
train_cutoff = int(len(golden_dataset) * .6)
dataset = [dspy.Example(request=row['request'], response=row['expected_response'], 
                        context=row['expected_retrieved_context']).with_inputs('request') 
           for i, row in golden_dataset.iterrows()]
trainset = dataset[:train_cutoff]
testset = dataset[train_cutoff:] # TODO: get the best examples for training, dynamically set the metric for length

# COMMAND ----------

class QueryExpand(dspy.Signature):
    """Rephrases the question to increase the quality of the response"""
    question = dspy.InputField(desc="A question about our business")
    expanded_question = dspy.OutputField(desc="A slightly expanded question that provides more context to our retrieval engine")

class QuestionAnswer(dspy.Signature):
    """Returns the answer to the question"""
    request = dspy.InputField(desc="A question about our business")
    response = dspy.OutputField(desc="A short, numbered answer to the question")

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
pred = qa_agent(dataset[0].request)
pred

# COMMAND ----------

class AssessResponse(dspy.Signature):
    """Assess the quality of an outline along the specified dimension."""
    request = dspy.InputField()
    response_to_assess = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

# COMMAND ----------

# Define the metric for optimization
def metric(gold, pred, trace=None):
    request, expected_response = gold.request, gold.response # , gold.expected_retrieved_context
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
        target_len = 600
        max_len_score = 3
        length_eval = max((target_len - len(response)) / (target_len/max_len_score), 0) # Our users asked for short, numbered lists, so we'll incentivize that behavior 

        evals = [not_duplicative_eval, listed_eval]
        results = ['yes' in m.assessment_answer.lower() for m in evals]
        score = (sum(results) + length_eval) / (len(evals) + max_len_score) # Total score over total possible score
        if trace is not None: # When training, we'll only return positive signal at a high score
            print(results, length_eval, evals)
            return score > .6
        return score

# COMMAND ----------

def evaluate(testset, system):
    scores = []
    results = []
    for x in testset:
        pred = system(**x.inputs())
        score = metric(x, pred)
        scores.append(score)
        results.append(pred)
    return sum(scores) / len(testset), results

# COMMAND ----------

qa_agent = QuestionAnswerAgent(rm)
baseline_score, baseline_results = evaluate(testset, qa_agent) # TODO: testset
print(f"Baseline score:    {baseline_score * 100:.2f}%")

# COMMAND ----------

with dspy.context(lm=judge): 
    judge_score, judge_results = evaluate(testset, qa_agent)
print(f"Judge score:    {judge_score * 100:.2f}%")

# COMMAND ----------

from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(prompt_model=judge, task_model=ft_lm, metric=metric, num_candidates=3, init_temperature=0.1)

# Optimize the program
kwargs = dict(num_threads=4, display_progress=True, display_table=0)
with dspy.context(lm=ft_lm):
    optimized_qa_agent = optimizer.compile(
        student=qa_agent,
        trainset=trainset, 
        eval_kwargs=kwargs,
        requires_permission_to_run=False,
    )

# COMMAND ----------

# Evaluate the optimized program
with dspy.context(lm=ft_lm):
    optimized_score, optimized_results = evaluate(testset, optimized_qa_agent)
    print(f"Optimized score:    {optimized_score * 100:.2f}%")

# COMMAND ----------

import mlflow
import pandas as pd

eval_output = [{
        "request": [i.request for i in testset],
        "response": [prediction.response for prediction in optimized_results],
        "expected_response": [i.response for i in testset]
}]
eval_metrics = mlflow.evaluate(
    data=pd.DataFrame(eval_output),
    model_type="databricks-agent"
).metrics
eval_metrics
