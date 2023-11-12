from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import os
import json
import pathlib
import ast
from tqdm import tqdm

model_name = 'google/flan-t5-base'
model_dir = f"models/{model_name.split('/')[-1]}/w_neutral_CCKG+IAM"
data_dir = 'data/argumentation'

model = T5ForConditionalGeneration.from_pretrained(model_dir)
tokenizer = T5Tokenizer.from_pretrained(model_dir)

def generate_arg(topic, arg_type='supporting'):
    prefix = f"Given the following topic, generate a good {arg_type} argument. Topic="
    inputs = tokenizer(prefix + topic, return_tensors='pt', padding=True, truncation=True)
    outputs = model.generate(**inputs, min_length=25, max_length=50)
    
    argument = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    result = {
        'argument_type': arg_type,
        'argument': argument
    }
    
    return json.dumps(result, ensure_ascii=True)

df_iam = pd.read_csv(os.path.join(data_dir, 'train_iam.tsv'), sep='\t')
df_cckg = pd.read_csv(os.path.join(data_dir, 'train_cckg.tsv'), sep='\t')
topics = pd.concat([df_iam['topic'].sample(2), df_cckg['topic'].sample(2)]).values

arg_types=['supporting', 'counter', 'neutral']

js = {}
for topic in tqdm(topics):
    by_topic = [] 
    for arg_type in arg_types:    
        by_topic.append(ast.literal_eval(generate_arg(topic, arg_type=arg_type)))
    js[topic] = by_topic


output_dir = 'data/argumentation/T5Infer_'+ model_dir.split('/')[-1]
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
output_path = os.path.join(output_dir, f"{model_name.split('/')[-1]}.json")
with open(output_path, 'w', encoding='utf-8') as json_file:
    json.dump(js, json_file, indent=4, ensure_ascii=False, sort_keys=False)
