# Databricks notebook source
# DBTITLE 1,Install DSPy
# MAGIC %pip install dspy-ai mlflow --upgrade -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Set up DSPy
import dspy

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

lm = dspy.Databricks(model='databricks-dbrx-instruct', model_type='chat', api_key=token, api_base=url + '/serving-endpoints', max_tokens=1000)
dspy.settings.configure(lm=lm)

# COMMAND ----------

# DBTITLE 1,Read Example Dataset
import pandas as pd

golden_dataset = pd.read_csv('./artifacts/blog_drafter/blogs_abstracts_and_outlines.csv')
train_cutoff = int(len(golden_dataset) * .6)
trainset = [dspy.Example(abstract=row['Abstract'], topic=row['Topic'], 
                        outline=row['Outline'], paragraphs=row['Blog'].split("\n"),
                        code_examples=' ').with_inputs('abstract') 
            for i, row in golden_dataset[:train_cutoff].iterrows()]
testset = [dspy.Example(abstract=row['Abstract'], topic=row['Topic'], 
                        outline=row['Outline'], paragraphs=row['Blog'].split("\n"),
                        code_examples=' ').with_inputs('abstract') 
            for i, row in golden_dataset[train_cutoff:].iterrows()]

# COMMAND ----------

# DBTITLE 1,Create Blogger Signatures
class AbstractToOutline(dspy.Signature):
    """Convert an abstract to a highly structured outline"""
    abstract = dspy.InputField(desc="A high level collection of thoughts to be used as the basis of a blog post")
    outline = dspy.OutputField(desc="A highly structured outline with sections and subsections like 1a, 1b, 1c, 2a, ...")
    topic = dspy.OutputField(desc="A short, concise description of the main idea of the blog post")

class SectionToParagraph(dspy.Signature):
    """Convert one section of an outline to a paragraph"""
    section = dspy.InputField(desc="A short section of an outline describing some supporting idea for the intended topic")
    topic = dspy.InputField(desc="The overall topic for a technical blog post")
    paragraph = dspy.OutputField(desc="A paragraph providing a detailed explanation of some supporting idea the topic and why it's relevant")

class ParagraphToCodeExample(dspy.Signature):
    """Write a short, concise code example"""
    paragraph = dspy.InputField(desc="A paragraph providing a detailed explanation of some supporting idea the topic and why it's relevant")
    code_example = dspy.OutputField(desc="A short, concise code example to make the concepts illustrated in the paragraph more concrete")

# COMMAND ----------

# DBTITLE 1,Create Blogger Module
import re

class AbstractToBlog(dspy.Module):
    """Converts an abstract to a blog post."""
    def __init__(self):
        super().__init__()
        self.outliner = dspy.ChainOfThought(AbstractToOutline)
        self.section_writer = dspy.ChainOfThought(SectionToParagraph)
        self.code_exampler = dspy.ChainOfThought(ParagraphToCodeExample)
    
    def parse_outline(self, outline):
        output = re.split(r'\d+[a-zA-Z]?\.', outline)
        return [line.strip() for line in output if line.strip()]
    
    def forward(self, abstract):
        outliner_output = self.outliner(abstract=abstract)
        outline, topic = outliner_output.outline, outliner_output.topic
        outline_sections = self.parse_outline(outline)
        paragraphs = [self.section_writer(section=section, topic=topic).paragraph 
                      for section in outline_sections 
                      if len(section.strip()) > 5]
        code_examples = [self.code_exampler(paragraph=paragraph).code_example 
                         for paragraph in paragraphs[1:-1]
                         if len(paragraph) > 10] # only body paragraphs with actual content
        return dspy.Prediction(outline=outline, topic=topic, paragraphs=paragraphs, code_examples=code_examples)

# COMMAND ----------

# DBTITLE 1,Run Unoptimized Module
test_abstract = "When you use Pandas UDFs, you can't pass parameters to your function by default. It's challenging to do things like object-oriented programming or hyperparameter tuning on Pandas UDFs. As a Databricks user, I might have legacy Pandas code that I'd like to run on Databricks. How can I pass parameters to my Pandas UDFs in order to scale out their processing across a Spark cluster with dynamic parameters? I propose the cleanest solution is by using closures that accept your parameters and return the appropriately configured Pandas UDF function"
uncompiled_blogger = AbstractToBlog()
pred = uncompiled_blogger(test_abstract)
print(pred.keys(), "\n\n", pred.outline)

# COMMAND ----------

# DBTITLE 1,Create Assessment Signature
class Assess(dspy.Signature):
    """Assess the quality of a piece of text along the specified dimension."""
    text_to_assess = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

assessor = dspy.ChainOfThought(Assess)

# COMMAND ----------

# DBTITLE 1,Create Metric
def blog_metric(gold, pred, trace=None):
    abstract, gold_paragraphs = gold.abstract, gold.paragraphs
    outline, topic, paragraphs, code_examples = pred.outline, pred.topic, pred.paragraphs, pred.code_examples

    structured = "Does the outline follow a highly structured format similar to 1a., b., c., 2a., b., c., etc?"
    introduction = "Does the outline start with a clear introduction section containing a problem statement and proposed solution?"
    support = "Are there supporting sections that can be used to provide examples and support for the introduction's proposed solution to the problem statement?"
    conclusion = "Does the outline end with a conclusion section that can be used to summarize what the reader has learned?"
    outline_evals =  [assessor(text_to_assess=outline, assessment_question=question) 
                      for question in [structured, introduction, support, conclusion]]
    outline_score = sum(['yes' in e.assessment_answer.lower() for e in outline_evals])

    clarity = "Is the paragraph clear, concise, and does it have continuity?"
    intention = "Does the paragraph have an obvious intention and showcase how Databricks, or data more generally, solves some real world problem?"
    detailed = f"Does the paragraph provide insightful and interesting details, rather than being generic or repetitive?"
    aligned = f"Is the paragraph aligned to the target topic, {topic}?"
    paragraph_evals = []
    for paragraph in paragraphs:
        for question in [clarity, intention, detailed, aligned]:
            evaluation = assessor(text_to_assess=paragraph, assessment_question=question)
            paragraph_evals.append(evaluation)
    paragraph_positives = ['yes' in e.assessment_answer.lower() for e in paragraph_evals]
    paragraph_score = sum(paragraph_positives) / max(len(paragraphs), 1)

    commented = "Is the code example commented appropriately?"
    relevant = f"Is the code example directly related to some section of this outline: {outline}"
    code_example_evals = []
    for paragraph in paragraphs:
        for question in [clarity, intention, detailed, aligned]:
            evluation = assessor(text_to_assess=paragraph, assessment_question=question)
            code_example_evals.append(evluation)
    code_example_score = sum(['yes' in e.assessment_answer.lower() for e in code_example_evals])

    return outline_score + paragraph_score + code_example_score

# COMMAND ----------

# DBTITLE 1,Evaluate the Baseline Module
from dspy.evaluate import Evaluate

evaluate = Evaluate(devset=testset, metric=blog_metric, num_threads=4, display_progress=False, display_table=0)
baseline_results = evaluate(AbstractToBlog())
print(baseline_results)

# COMMAND ----------

# DBTITLE 1,Optimize the Module
from dspy.teleprompt import BootstrapFewShot

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) to 3-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=1, max_labeled_demos=2)

# Optimize! In general, the metric is going to tell the optimizer how well it's doing.
optimizer = BootstrapFewShot(metric=blog_metric, **config)
optimized_blogger = optimizer.compile(AbstractToBlog(), trainset=trainset)

# COMMAND ----------

# DBTITLE 1,Evaluate the Optimized Module
evaluate = Evaluate(devset=testset, metric=blog_metric, num_threads=4, display_progress=False, display_table=0)
optimized_results = evaluate(optimized_blogger)
print(optimized_results)

# COMMAND ----------

# DBTITLE 1,Measure Improvement
# lm.inspect_history(n=2)
improvement = (optimized_results / baseline_results) - 1
print(f"% improvement: {improvement * 100}")

# COMMAND ----------

# DBTITLE 1,Create MLflow Pyfunc
import mlflow
import pandas as pd
import os


os.environ['url'] = url
os.environ['token'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

class DSPyWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, blogger):
        self.blogger = blogger

    def load_context(self, context):
        self.dspy_setup()

    def dspy_setup(self):
        import dspy
        import os
        url = os.environ['url']
        token = os.environ['token']
        lm = dspy.Databricks(model='databricks-dbrx-instruct', model_type='chat', api_key=token, api_base=url, max_tokens=1000)
        dspy.settings.configure(lm=lm)
    
    def draft_blog(self, row):
        pred = self.blogger(row['abstract'])
        outline, topic, paragraphs, code_examples = pred.outline, pred.topic, pred.paragraphs, pred.code_examples
        return pd.Series([outline, topic, paragraphs, code_examples])

    def predict(self, context, input_df):
        output = input_df.apply(self.draft_blog, axis=1, result_type='expand')
        output.columns = ['outline', 'topic', 'paragraphs', 'code_examples']
        return output

mlflow_dspy_model = DSPyWrapper(optimized_blogger)
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
    mlflow.pyfunc.log_model(artifact_path="model", python_model=DSPyWrapper(optimized_blogger), signature=signature, 
                            input_example=input_data, extra_pip_requirements=["dspy=={dspy_version}"])
    mlflow.log_metric("blog_metric", optimized_results)

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


