#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import ast
import joblib
import warnings
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from collections import Counter
from collections import defaultdict
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm



# In[2]:


classifier_name = "microsoft/phi-4"

tokenizer = AutoTokenizer.from_pretrained(classifier_name,trust_remote_code = True)
model = AutoModelForCausalLM.from_pretrained(classifier_name,trust_remote_code = True,device_map='auto')


# In[3]:


## generate response code
def generate_response(prompt,llm = "microsoft/phi-4"):
    
    
#     weakness = row['Review Segment']
#     input_prompt = prompt.replace('{{weakness}}', weakness)
    input_prompt = prompt
    messages = [
    {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)

    # Decode response
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response # remove prompt from response


# In[4]:


# extract feature vector code
def yes_no_list_to_int(lst):
    result = []
    for x in lst:
        x_lower = x.lower()
        if "yes" in x_lower:
            result.append(1)
        elif "no" in x_lower:
            result.append(-1)
        else:
            result.append(0)  # or raise an error
    return result


# In[5]:


# Feature vector generation prompt
feature_vector_prompt = """
You will be given a review segment. Your task is to evaluate its quality by answering a series of Yes/No questions.

Review Segment: {review}

Question: {question}

Respond strictly with either:

[[Yes]] if the answer is Yes

[[No]] if the answer is No

[[Other]] if the question is irrelevant

Do not provide any explanation or extra text.
"""


# In[6]:





# In[7]:


def generate_feature_vectors(review_segment, question_dict, prompt_template=feature_vector_prompt,llm="microsoft/phi-4"):
    """
    Run a set of questions through an LLM for a single review segment.

    Args:
        review_segment (str): The review text to analyze.
        question_dict (dict): Dictionary where keys are categories, values are lists of questions.
        prompt_template (str): Template prompt with placeholders `{review}` and `{question}`.

    Returns:
        dict: Same keys as question_dict, values are lists of LLM outputs for each question.
    """
    results = []
    ## load phi4

    

    for category, questions in tqdm(question_dict.items(),desc="Generating Feature Vectors"):
        category_results = []
        for q in questions:
            prompt_filled = prompt_template.format(review=review_segment, question=q)
            response = generate_response(prompt_filled,llm=llm)  # Replace with your LLM call
            category_results.append(response)
        results.append(yes_no_list_to_int(category_results))

    return results


# In[5]:


full = pd.read_csv("final_data/full_dataset_bio_combined.tsv",sep="\t")
def safe_str(s):
    """
    Convert a string into a safe version for filenames.
    Replaces any character that is not a letter, number, underscore, or hyphen with '_'.
    """
    return re.sub(r'[^a-zA-Z0-9_-]', '_', s)
llms = []
template_pkl = "full-{}-advanced_features.pkl"
template_csv = "full-{}-advanced_features.csv"
for llm in llms:
    feature_questions_file = template_pkl.format(safe_str(llm))
    feature_questions = pd.read_pickle(feature_questions_file)
    csv_file = template_csv.format(safe_str(llm))
    feature_vectors = []
    for index,row in tqdm(full.iterrows(),desc="Generating feature vectors"):

        review_segment = row["Review Segment"]
        feature_vectors.append(generate_feature_vectors(review_segment,feature_questions))
    full['feature_vectors'] = feature_vectors
    full.to_csv(csv_file)





