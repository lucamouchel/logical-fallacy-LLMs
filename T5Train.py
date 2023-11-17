from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import os
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import nltk
import warnings
from data_loader import load_data
warnings.filterwarnings("ignore")


DATA_SOURCE = ['cckg']
AUGMENT_WITH_NEUTRAL_ARGS = False
data_dir = "data/argumentation"


if len(DATA_SOURCE) == 1:    
    train_df, dev_df, test_df = load_data(DATA_SOURCE[0])
else:     
    train_iam, dev_iam, test_iam = load_data(DATA_SOURCE[0]) 
    train_cckg, dev_cckg, test_cckg = load_data(DATA_SOURCE[1])

    train_df = pd.concat([train_iam, train_cckg]).sample(frac=1) #shuffle these bad boys
    dev_df = pd.concat([dev_iam, dev_cckg]).sample(frac=1)
    test_df = pd.concat([test_iam, test_cckg]).sample(frac=1)

all_claims = pd.read_csv(os.path.join(data_dir, 'claims.tsv'), sep='\t')

np.random.seed(42)

if AUGMENT_WITH_NEUTRAL_ARGS:
    neutral_claims = all_claims[(all_claims.type=='O') & (all_claims.label==0)] 
    lower_bound = 0
    min_train_label = min(train_df['label'].value_counts())
    train_sample = neutral_claims.iloc[:min_train_label]
    train_sample = train_sample[['topic', 'argument', 'label']]
    train_df = pd.concat([train_df, train_sample]).sample(frac=1)
    lower_bound = min_train_label
    
    min_dev_label = min(dev_df['label'].value_counts())
    dev_sample = neutral_claims.iloc[lower_bound: lower_bound + min_dev_label]  
    dev_sample = dev_sample[['topic', 'argument', 'label']]  
    dev_df = pd.concat([dev_df, dev_sample]).sample(frac=1)
    lower_bound = lower_bound + min_dev_label
    
    min_test_label = min(test_df['label'].value_counts())
    test_sample = neutral_claims.iloc[lower_bound: lower_bound + min_test_label]    
    test_sample = test_sample[['topic', 'argument', 'label']]
    test_df = pd.concat([test_df, test_sample]).sample(frac=1)

label_encoder = LabelEncoder()
label_encoder.fit(train_df['label'])
train_df['label'] = label_encoder.transform(train_df['label'])
dev_df['label'] = label_encoder.transform(dev_df['label'])
test_df['label'] = label_encoder.transform(test_df['label'])

dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'validation': Dataset.from_pandas(dev_df),
    'test': Dataset.from_pandas(test_df)
})

from transformers import AutoTokenizer
model_name = 'google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
metric = evaluate.load('bleu')

"""    
need to be careful cause we cant give the same prefix to every type of 
arguments otherwise the model generalises and considers counter arguments as 
supporting arguments if we give the prefix : 'generate a supporting argument'   
"""
def preprocess_function(sample):
    def process_by_type(label_type):
        if AUGMENT_WITH_NEUTRAL_ARGS:
            if label_type==2:
                label = 'supporting'
            elif label_type == 1:
                label = 'neutral'
            elif label_type==0:
                label = 'counter'
            else:
                raise ValueError
        else:
            if label_type == 1:
                label = 'supporting'
            elif label_type == 0:
                label = 'counter'
            else: 
                raise ValueError
        
        label_indices = [i for i, label in enumerate(sample['label']) if label == label_type]        
        prefix = f"Given the following topic, generate a good {label} argument. Topic="
        labeled_samples = {key: [sample[key][i] for i in label_indices] for key in sample.keys()}
        inputs = [prefix + doc for doc in labeled_samples['topic']]

        model_inputs = tokenizer(inputs, max_length=4096, truncation=True, padding=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(labeled_samples['argument'], max_length=4096, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    if AUGMENT_WITH_NEUTRAL_ARGS:
        model_inputs_supporting=process_by_type(label_type=2) ## supporting
        model_inputs_neutral=process_by_type(label_type=1) ## neutral
        model_inputs_counter=process_by_type(label_type=0) ## counter
        
        combined_model_inputs = {
        'input_ids': model_inputs_supporting['input_ids'] + model_inputs_neutral['input_ids'] + model_inputs_counter['input_ids'],
        'attention_mask': model_inputs_supporting['attention_mask'] + model_inputs_neutral['attention_mask'] + model_inputs_counter['attention_mask'],
        'labels': model_inputs_supporting['labels'] + model_inputs_neutral['labels'] + model_inputs_counter['labels']
        }
        
    else:
        model_inputs_supporting=process_by_type(label_type=1) ## supporting
        model_inputs_counter=process_by_type(label_type=0) ## counter
        
        combined_model_inputs = {
        'input_ids': model_inputs_supporting['input_ids'] + model_inputs_counter['input_ids'],
        'attention_mask': model_inputs_supporting['attention_mask'] + model_inputs_counter['attention_mask'],
        'labels': model_inputs_supporting['labels'] + model_inputs_counter['labels']
        }
    
    return combined_model_inputs
        
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['label'])

batch_size = 8
training_arguments = Seq2SeqTrainingArguments(
    output_dir='results/',
    evaluation_strategy='epoch',
    learning_rate=3e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    load_best_model_at_end=True,
    save_strategy="epoch",
    logging_strategy="steps",
    fp16=True, # for cuda
    push_to_hub=False,
    logging_steps=50,
    eval_steps=50,
    save_steps=50,
)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

trainer = Seq2SeqTrainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
saved_model = f"models/{model_name.split('/')[-1]}/without_neutral_CCKG/"
trainer.save_model(saved_model)
predictions = trainer.predict(tokenized_dataset["test"])

print(predictions.metrics)



