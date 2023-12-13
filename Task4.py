import numpy as np
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import argparse
import pickle


tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
model = AutoModel.from_pretrained('facebook/contriever-msmarco')


import json
# Open the JSON file for reading
file_path = '/work/mrastikerdar_umass_edu/ir_project/'
task_num = 4
file_name = f'task{task_num}_test_questions.json'
with open(file_path+file_name, 'r') as file:
    # Parse the file and convert JSON to a Python object
    data = json.load(file)
    
import re
phi_q = []
  
for user_num in range(len(data)):

    # Example text
    text = data[user_num]['input']

    # Regular expression to extract [TITLE], [REF1], and [REF2]
    match = re.search(r'article: (.+)', text)
    phi_q.append(match.group(1))

        
    
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


score_p = {}
for input_num in range(len(data)):
    score_p.update({data[input_num]['id']:[]})
    print(input_num)
    query = phi_q[input_num]
    input_q = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
    output_q = model(**input_q)
    embed_q = mean_pooling(output_q[0], input_q['attention_mask'])
    for doc_num in range(len(data[input_num]['profile'])):
      doc = data[input_num]['profile'][doc_num]['text']
      input_d = tokenizer(doc, padding=True, truncation=True, return_tensors='pt')
      output_d = model(**input_d)
      embed_d = mean_pooling(output_d[0], input_d['attention_mask'])
      rel_score = embed_q[0] @ embed_d[0]
      score_p[data[input_num]['id']].append(rel_score.item())
       
      
        
      
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = model.to(device)

        

#### Contriever Top-K ####


    
score_list = list(score_p.values())
ids = list(score_p.keys())

k = 4
prompts = []
for user in range(len(data)):
    prompt = {}
    prompt['id'] = data[user]['id']
    prompt['text'] = ''
    sorted_score_user_idx = np.flip(np.argsort(score_list[user]))

    for i in range(k):
        retrieved_idx = sorted_score_user_idx[i]
        title = data[user]['profile'][i]['title']
        text = data[user]['profile'][i]['text']
        prompt['text'] += f'{title} is the title for {text}'

        if i < k-1:
            prompt['text'] += ', and '
        else:
            prompt['text'] += '. '
            prompt['text'] += data[user]['input']
    prompts.append(prompt)
    
    
device = "cuda:0" if torch.cuda.is_available() else "cpu"



prediction_task4 = []
for i in range(len(data)):
    
    print(i)
    inputs = tokenizer(prompts[i]['text'], return_tensors="pt").to(device)
    model = model.to(device)
    outputs = model.generate(**inputs)
    prediction_task4.append(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
