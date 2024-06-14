# Databricks notebook source
# DBTITLE 1,Imports
import pandas as pd
import re
from langchain.chat_models import ChatDatabricks

chat_model = ChatDatabricks(endpoint='databricks-meta-llama-3-70b-instruct', max_tokens = 1000)

# COMMAND ----------

# DBTITLE 1,Parse Paragraphs
blogs_df = pd.read_csv('../../artifacts/blog_drafter/blogs_abstracts_and_outlines.csv')
blogs = blogs_df.to_dict(orient='records')

def parse_outline(outline):
    output = re.split(r'\d+[a-zA-Z]?\.', outline)
    return ['a. ' + line.strip() for line in output if line.strip()]

paragraphs_dict = {}
for row in blogs[:5]:
    parsed_outline = parse_outline(row['Outline'])
    for section in parsed_outline:
        paragraph = chat_model.predict('Pick out the whole paragraph, and only that paragraph, from the provided blog which corresponds to the provided section of the outline. Only return the entirety of the relevant paragraph, no filler, editing, or extra description of what you are doing and no mention of here is the relevant paragraph. Include the following code example if relevant.\n\nBlog:\n' + row['Blog'] + '\n\Outline Section:\n' + section)
        paragraphs_dict[section] = paragraph

data = [{'outline': outline, 'paragraph': paragraph} for outline, paragraph in paragraphs_dict.items()]
df = pd.DataFrame(data)
df['paragraph'] = df['paragraph'].str.replace('Here is the relevant paragraph:', '')
display(df)

# COMMAND ----------

# DBTITLE 1,Parse topics
blogs_df = pd.read_csv('../../artifacts/blog_drafter/blogs_abstracts_and_outlines.csv')
blogs = blogs_df.to_dict(orient='records')
topics_dict = {}
for row in blogs[:5]:
    topic = chat_model.predict('Describe the overall topic from the provided section of the outline. Limit the topic to one sentence maximum, keep it very concise, short, and to the point.\n\Blog:\n' + row['Blog'])
    topics_dict[row['Blog']] = topic
topics_dict

# COMMAND ----------

for item in topics_dict.keys():
    print(topics_dict[item], '\n\n')

# COMMAND ----------


