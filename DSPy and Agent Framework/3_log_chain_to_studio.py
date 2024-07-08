# Databricks notebook source
# DBTITLE 1,Install Libraries
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries
import os
import mlflow
import time
from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks.sdk.errors import NotFound, ResourceDoesNotExist
from utils.demo import parse_deployment_info, _flatten_nested_params
import yaml
w = WorkspaceClient()

# COMMAND ----------

# DBTITLE 1,Setup
# Specify the full path to the chain notebook & config YAML
# Assuming your chain notebook is in the current directory, this helper line grabs the current path, prepending /Workspace/
# Limitation: RAG Studio does not support logging chains stored in Repos
current_path = '/Workspace' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

chain_notebook_file = "2_set_rag_chain"
chain_config_file = "rag_config.yaml"
chain_notebook_path = f"{current_path}/{chain_notebook_file}"
chain_config_path = f"{current_path}/{chain_config_file}"

with open("rag_config.yaml", "r") as file:
    rag_config = yaml.safe_load(file)
model_fqdn = rag_config.get("demo_config").get("model_fqdn")
mlflow_run_name = rag_config.get("demo_config").get("mlflow_run_name")
rag_app_name = rag_config.get("demo_config").get("rag_app_name")

print(f"Saving chain from: {chain_notebook_path}, config from: {chain_config_path}")

# COMMAND ----------

# DBTITLE 1,Log the chain
model_input_sample = {
    "messages": [
        {
            "role": "user",
            "content": "What is causing delivery delays?",
        }
    ]
}

with mlflow.start_run(run_name=mlflow_run_name): 
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=chain_notebook_path,  # Chain code file e.g., /path/to/the/chain.py
        model_config=rag_config,  # Chain configuration set in 00_config
        artifact_path="chain",  # Required by MLflow
        input_example=model_input_sample,  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
    )

# COMMAND ----------

# DBTITLE 1,Local Model Testing
# Run the model to see the output
loaded_model = mlflow.langchain.load_model(logged_chain_info.model_uri)
loaded_model.invoke(model_input_sample)

# COMMAND ----------

instructions_to_reviewer = f"""## Instructions for Testing the {rag_app_name}'s Initial Proof of Concept (PoC)

Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement.

1. **Variety of Questions**:
   - Please try a wide range of questions that you anticipate the end users of the application will ask. This helps us ensure the application can handle the expected queries effectively.

2. **Feedback on Answers**:
   - After asking each question, use the feedback widgets provided to review the answer given by the application.
   - If you think the answer is incorrect or could be improved, please use "Edit Answer" to correct it. Your corrections will enable our team to refine the application's accuracy.

3. **Review of Returned Documents**:
   - Carefully review each document that the system returns in response to your question.
   - Use the thumbs up/down feature to indicate whether the document was relevant to the question asked. A thumbs up signifies relevance, while a thumbs down indicates the document was not useful.

Thank you for your time and effort in testing {rag_app_name}. Your contributions are essential to delivering a high-quality product to our end users."""

print(instructions_to_reviewer)

# COMMAND ----------

# DBTITLE 1,Register the model
from mlflow.client import MlflowClient
# To deploy the model, first register the chain from the MLflow Run as a Unity Catalog model.
mlflow.set_registry_uri('databricks-uc')
uc_registered_model_info = mlflow.register_model(logged_chain_info.model_uri, model_fqdn)
MlflowClient().set_registered_model_alias(model_fqdn, "Challenger", uc_registered_model_info.version)

deployment_info = agents.deploy(model_name=model_fqdn, model_version=uc_registered_model_info.version)

# COMMAND ----------

agents.set_review_instructions(model_fqdn, instructions_to_reviewer)
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")
while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    print(".", end="")
    time.sleep(30)

print(f"\n\nReview App: {deployment_info.review_app_url}")

# COMMAND ----------

chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(model_input_sample)

# COMMAND ----------

# user_list = ["josh.melton@databricks.com"]

# # Set the permissions.  If successful, there will be no return value.
# agents.set_permissions(model_name=model_fqdn, users=user_list, permission_level=agents.PermissionLevel.CAN_QUERY)

# COMMAND ----------

# active_deployments = agents.list_deployments()
# active_deployment = next((item for item in active_deployments if item.model_name == model_fqdn), None)
# print(f"Review App URL: {active_deployment.review_app_url}")

# COMMAND ----------

# agents.get_deployments(model_fqdn)
# agents.delete_deployment(model_name=model_fqdn, model_version=3)
# from databricks.agents.client.rest_client import delete_chain
# delete_chain(model_fqdn, 5)

# COMMAND ----------


