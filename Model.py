from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import pandas as pd
import numpy as np

class Model:
    def __init__(self, model_name, train_data = pd.read_csv('data/edu_train.csv'), dev_data = pd.read_csv('data/edu_dev.csv'), test_data = pd.read_csv('data/edu_test.csv')) -> None:
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=13)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def tokenize_text(self, text, padding=True, truncation=True, add_special_tokens=True, return_tensors='pt'):
        return self.tokenizer(text, padding=padding, truncation=truncation, add_special_tokens=add_special_tokens, return_tensors=return_tensors)

    def encode(self, data, padding=True, truncation=True, return_tensors='pt'):
        texts = data['source_article'].tolist()
        return self.tokenizer(texts, padding=padding, truncation=truncation, return_tensors=return_tensors)
    
    def label_encoder(self, data):
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(data['updated_label'])
        
    def data_tensor(self, data):
        encoded = self.encode(data)
        return torch.tensor(encoded.input_ids), torch.tensor(encoded.attention_mask), torch.tensor(self.label_encoder(data))
    
    def get_model(self):
        return self.model
    
    