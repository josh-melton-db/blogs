# Databricks notebook source
# DBTITLE 1,Install DSPy
# MAGIC %pip install dspy-ai==2.4.9 mlflow -q
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Set up DSPy
import dspy

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
serving_url = url + '/serving-endpoints'

lm = dspy.Databricks(model='databricks-dbrx-instruct', model_type='chat', api_key=token, 
                     api_base=serving_url, max_tokens=1000, temperature=0.85)
teacher = dspy.Databricks(model='databricks-meta-llama-3-70b-instruct', model_type='chat', api_key=token, 
                          api_base=serving_url, max_tokens=1000, temperature=0)
dspy.settings.configure(lm=lm) # TODO: configure caching to use a Volume

# COMMAND ----------

# DBTITLE 1,Create Retriever
from dspy.retrieve.databricks_rm import DatabricksRM

docs_rm = DatabricksRM( # This index was created via the Databricks Demo Center RAG Example
    databricks_index_name="josh_melton.blogs.databricks_documentation_vs_index",
    databricks_endpoint=url,
    databricks_token=token,
    columns=["content"],
    text_column_name="content",
    docs_id_column_name="id",
)
retrieved = docs_rm(query="Model serving API", query_type="text")
retrieved.docs

# COMMAND ----------

# DBTITLE 1,Create Outline Module
class AbstractToOutlineSig(dspy.Signature):
    """Convert an abstract to a highly structured outline"""
    abstract = dspy.InputField(desc="A high level collection of thoughts to be used as the basis of a blog post")
    outline = dspy.OutputField(desc="A highly structured outline of a topical and informative Databricks blog post with sections and subsections like 1a, 1b, 1c, 2a, ...")
    title = dspy.OutputField(desc="A short, concise title describing the main idea of the blog post in a few words")

class AbstractToOutline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(AbstractToOutlineSig)
    
    def forward(self, abstract):
        return self.prog(abstract=abstract)

# COMMAND ----------

# DBTITLE 1,Create Paragraph Module
class SectionToParagraphSig(dspy.Signature):
    """Convert one section of an outline to a paragraph"""
    section = dspy.InputField(desc="A short section of an outline describing the specific supporting idea to write about")
    abstract = dspy.InputField(desc="The high level abstract for the overall technical blog post")
    paragraph = dspy.OutputField(desc="A paragraph providing a detailed explanation of some supporting idea the topic and why it's relevant")

class SectionToParagraph(dspy.Module):
    def __init__(self, docs_rm, iterations=3):
        super().__init__()
        self.docs_rm = docs_rm
        self.prog = dspy.ChainOfThought(SectionToParagraphSig)
        self.iterations = iterations

    def get_context(self, query):
        context_list = self.docs_rm(query=query, query_type="text").docs
        return "\n".join(context_list)
    
    def forward(self, section, abstract):
        context = self.get_context(abstract + "\n" + section)
        output = self.prog(section=section, abstract=abstract, context="")
        for iteration in range(self.iterations):
            output = self.prog(section=section, abstract=abstract, context=context)
        return output

# COMMAND ----------

# DBTITLE 1,Run Unoptimized Outline Module
test_abstract = "When you use Pandas UDFs, you can't pass parameters to your function by default. It's challenging to do things like object-oriented programming or hyperparameter tuning on Pandas UDFs. As a Databricks user, I might have legacy Pandas code that I'd like to run on Databricks. How can I pass parameters to my Pandas UDFs in order to scale out their processing across a Spark cluster with dynamic parameters? I propose the cleanest solution is by using closures that accept your parameters and return the appropriately configured Pandas UDF function"
unoptimized_outliner = AbstractToOutline()
pred = unoptimized_outliner(test_abstract)
print(pred)

# COMMAND ----------

# DBTITLE 1,Run Unoptimized Outline Module
test_section = "2. Understanding the Problem\n   2a. Provide a detailed explanation of the challenge of passing parameters to Pandas UDFs in Databricks.\n   2b. Show an example of a simple Pandas UDF that does not accept parameters."
unoptimized_paragrapher = SectionToParagraph(docs_rm)
pred = unoptimized_paragrapher(test_section, test_abstract)
print(pred)

# COMMAND ----------

# DBTITLE 1,Read Outline Examples
import pandas as pd

outlines_golden_dataset = pd.read_csv('./artifacts/blog_drafter/blogs_abstracts_and_outlines.csv')
outline_train_cutoff = int(len(outlines_golden_dataset) * .6)
outline_dataset = [dspy.Example(abstract=row['Abstract'], outline=row['Outline'], title=row['Title']).with_inputs('abstract') 
                    for i, row in outlines_golden_dataset.iterrows()]
outline_trainset = outline_dataset[:outline_train_cutoff]
outline_testset = outline_dataset[outline_train_cutoff:]

# COMMAND ----------

# DBTITLE 1,Read Paragraph Examples
paragraphs_golden_dataset = pd.read_csv('./artifacts/blog_drafter/sections_and_paragraphs.csv')
paragraph_train_cutoff = int(len(paragraphs_golden_dataset) * .6)
paragraph_dataset = [dspy.Example(section=row['Section'], abstract=row['Abstract'], paragraph=row['Paragraph']).with_inputs('section', 'abstract') 
                    for i, row in paragraphs_golden_dataset.iterrows()]
paragraph_trainset = paragraph_dataset[:paragraph_train_cutoff]
paragraph_testset = paragraph_dataset[paragraph_train_cutoff:]

# COMMAND ----------

# DBTITLE 1,Create Assessment Signature
class Assess(dspy.Signature):
    """Assess the quality of a piece of text along the specified dimension."""
    text_to_assess = dspy.InputField(desc="Piece of text to assess")
    assessment_question = dspy.InputField(desc="Question to answer about the text")
    assessment_answer = dspy.OutputField(desc="Yes or No")

# Note: as we gather more data we could optimize this module too!
assessor = dspy.ChainOfThought(Assess)

# COMMAND ----------

# DBTITLE 1,Define Outline Metric
from dspy.evaluate import Evaluate


def outline_metric(gold, pred, trace=None):
    gold_outline, gold_title = gold.outline, gold.title
    outline, title = pred.outline, pred.title
    engaging = f"Does the title convey an engaging and valuable idea for a Databricks blog in a concise way, similar to the following title? \n Title: {gold_title} \n"
    structured = f"Does the outline follow a highly structured format similar to 1a, b, c, 2a, b, c, etc similar to the following Outline? \n Outline: \n {gold_outline}"
    introduction = "Does the outline start with a clear introduction section with a problem statement relevant to Databricks end users?"
    support = "Are there body sections with examples and support for the proposed solution to the problem statement?"
    conclusion = "Is there a clear conclusion section that emphasizes what the reader has learned and a solution to the problem introduced?"
    with dspy.context(lm=teacher): # Use the teacher to grade the student model
        evals =  [assessor(text_to_assess=outline, assessment_question=question) 
                  for question in [engaging, structured, introduction, support, conclusion]]
    score = sum(['yes' in e.assessment_answer.lower() for e in evals])
    if len(title) < 50: score += 1 # Incentivize shorter titles
    return score / len(evals)+1

evaluate = Evaluate(devset=outline_testset, metric=outline_metric, num_threads=4, display_progress=False, display_table=0)
outline_baseline_results = evaluate(AbstractToOutline()) / len(outline_testset)
print(outline_baseline_results)

# COMMAND ----------

def paragraph_metric(gold, pred, trace=None):
    gold_paragraph, abstract = gold.paragraph, gold.abstract
    paragraph = pred.paragraph
    clarity = f"Is the given paragraph clear and concise? Does it have continuity similar to the following paragraph? \n Paragraph: \n {gold_paragraph}"
    support = "Does the paragraph clearly articulate a supporting point about how Databricks or data more generally solves a problem?"
    example = "Is the paragraph either an introduction, a conclusion, or a supporting paragraph with a specific (often code) example to illustrate its point?"
    detailed = "Does the paragraph provide excellent detail about the overall point, rather than being generic or repetitive?"
    aligned = f"Is the paragraph supportive of and aligned to the following abstract? \n Abstract: {abstract}?"
    with dspy.context(lm=teacher): # Use the teacher to grade the student model
        evals =  [assessor(text_to_assess=paragraph, assessment_question=question) 
                  for question in [clarity, support, example, detailed, aligned]]
    score = sum(['yes' in e.assessment_answer.lower() for e in evals])
    return score / len(evals)

evaluate = Evaluate(devset=paragraph_testset, metric=paragraph_metric, num_threads=4, display_progress=False, display_table=0)
paragraph_baseline_results = evaluate(SectionToParagraph(docs_rm)) / len(paragraph_testset)
print(paragraph_baseline_results)

# COMMAND ----------

# DBTITLE 1,Optimize Outline Module
from dspy.teleprompt import BootstrapFewShot

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) to 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=1, max_labeled_demos=3, teacher_settings=dict({'lm': teacher}))

# Optimize! The metric is going to tell the optimizer how well it's doing according to our qualitative statements
optimizer = BootstrapFewShot(metric=outline_metric, **config)
optimized_outliner = optimizer.compile(AbstractToOutline(), trainset=outline_trainset)

# COMMAND ----------

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) to 4 examples of our CoT program.
config = dict(max_bootstrapped_demos=1, max_labeled_demos=3, teacher_settings=dict({'lm': teacher}))

# Optimize! The metric is going to tell the optimizer how well it's doing according to our qualitative statements
optimizer = BootstrapFewShot(metric=paragraph_metric, **config)
optimized_paragrapher = optimizer.compile(SectionToParagraph(docs_rm), trainset=paragraph_trainset)

# COMMAND ----------

# DBTITLE 1,Evaluate Optimized Outline Module
evaluate = Evaluate(devset=outline_testset, metric=outline_metric, num_threads=4, display_progress=False, display_table=0)
optimized_outline_results = evaluate(optimized_outliner) / len(outline_testset)
print(optimized_outline_results)
improvement = (optimized_outline_results / outline_baseline_results) - 1
print(f"% improvement: {improvement * 100}")

# COMMAND ----------

evaluate = Evaluate(devset=paragraph_testset, metric=paragraph_metric, num_threads=4, display_progress=False, display_table=0)
optimized_paragraph_results = evaluate(optimized_paragrapher) / len(paragraph_testset)
print(optimized_paragraph_results)
improvement = (optimized_paragraph_results / paragraph_baseline_results) - 1
print(f"% improvement: {improvement * 100}")

# COMMAND ----------

# DBTITLE 1,Create MLflow Pyfunc
import mlflow
import os


os.environ['token'] = token
os.environ['url'] = serving_url

class DSPyWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, outliner, paragrapher):
        self.outliner = outliner
        self.paragrapher = paragrapher

    def load_context(self, context):
        self.dspy_setup()

    def dspy_setup(self):
        import dspy
        from dspy.retrieve.databricks_rm import DatabricksRM
        import os
        url = os.environ['url']
        token = os.environ['token']
        lm = dspy.Databricks(model='databricks-dbrx-instruct', model_type='chat', api_key=token, api_base=url, max_tokens=1000)
        dspy.settings.configure(lm=lm)

    def parse_outline(self, outline):
        import re
        output = re.split(r'\d+[a-zA-Z]?\.', outline)
        return [line.strip() for line in output if line.strip()]
    
    def draft_blog(self, row):
        outline_pred = self.outliner(row['abstract'])
        outline, title = outline_pred.outline, outline_pred.title
        outline_sections = self.parse_outline(outline)
        paragraphs = [self.paragrapher(section=section, abstract=row['abstract']).paragraph 
                      for section in outline_sections 
                      if len(section.strip()) > 5]
        return pd.Series([outline, title, paragraphs])

    def predict(self, context, input_df):
        output = input_df.apply(self.draft_blog, axis=1, result_type='expand')
        output.columns = ['outline', 'title', 'paragraphs']
        return output

mlflow_dspy_model = DSPyWrapper(optimized_outliner, optimized_paragrapher)
input_data = pd.DataFrame({'abstract': [test_abstract]})
pred = mlflow_dspy_model.predict(None, input_data)
display(pred)

# COMMAND ----------

# DBTITLE 1,Log to MLflow
from mlflow.models import infer_signature
import pkg_resources

dspy_version = pkg_resources.get_distribution("dspy-ai").version
signature = infer_signature(input_data, pred)
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(artifact_path="model", python_model=mlflow_dspy_model, 
                            signature=signature, input_example=input_data, 
                            extra_pip_requirements=[f"dspy-ai=={dspy_version}"])
    mlflow.log_metric("outline_metric", optimized_outline_results)
    mlflow.log_metric("paragraph_metric", optimized_paragraph_results)

# COMMAND ----------

# DBTITLE 1,Register to UC
mlflow.set_registry_uri("databricks-uc")
model_path = "josh_melton.blogs.blog_post_drafter"
latest_model = mlflow.register_model(f"runs:/{run.info.run_id}/model", name=model_path)
client = mlflow.client.MlflowClient()
client.set_registered_model_alias(name=model_path, alias="Production", version=latest_model.version)

# COMMAND ----------

# DBTITLE 1,Inference With UC Model
model_path = "josh_melton.blogs.blog_post_drafter"
registered_model_uri = f"models:/{model_path}@Production"
model = mlflow.pyfunc.load_model(registered_model_uri)
display(model.predict(input_data))

# COMMAND ----------


