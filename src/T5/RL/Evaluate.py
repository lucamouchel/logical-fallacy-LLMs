from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5EncoderModel
import torch

class Eval:
    def __init__(self, ckpt_path, score_model_name, classify_model_name) -> None:
        self.model = T5EncoderModel.from_pretrained(score_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(score_model_name)        
        self.model.D = self.model.shared.embedding_dim
        self.linear = torch.nn.Linear(self.model.D, 1)
        self.checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(self.checkpoint['model'])
        self.linear.load_state_dict(self.checkpoint['linear'])
        self.checkpoint.clear()


        self.fallacy_classifier = AutoModelForSequenceClassification.from_pretrained(classify_model_name)
        self.fallacy_tokenizer = AutoTokenizer.from_pretrained(classify_model_name)
        
    def score(self, sentence):
            #model = T5EncoderModel.from_pretrained('google/t5-v1_1-small')
            #model.D = model.shared.embedding_dim
            #linear = torch.nn.Linear(model.D, 1)
            #checkpoint = torch.load('runs/train/model/ckp_25000.pth', map_location='cpu')
            #model.load_state_dict(checkpoint['model'])
            #linear.load_state_dict(checkpoint['linear'])
            #checkpoint.clear()

            input_ids = self.tokenizer(sentence, return_tensors='pt')['input_ids']
            attention_mask = (input_ids != self.tokenizer.pad_token_id).float()            
            flattened_sourcess_input_ids = input_ids.view(-1, input_ids.size(-1)) # (B * C, L)
            flattened_sourcess_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) # (B * C, L)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=flattened_sourcess_input_ids,
                    attention_mask=flattened_sourcess_attention_mask,
                )
                
                last_indices = flattened_sourcess_attention_mask.sum(dim=1, keepdim=True) - 1 # (B * C, 1)
                last_indices = last_indices.unsqueeze(-1).expand(-1, -1, self.model.D)
                last_indices = last_indices.to(torch.int64)

                last_hidden_state = outputs.last_hidden_state.to('cpu')
                hidden = last_hidden_state.gather(dim=1, index=last_indices).squeeze(1) # (B * C, D)
                flattened_logitss = self.linear(hidden).squeeze(-1) # (B * C)
                logitss = flattened_logitss.view(input_ids.size(0), -1) # (B, C)
                # WARNING: This only works on T5 and Llama tokenizers!
                mask = (attention_mask.sum(dim=-1) > 1) # (B, C)
                logitss[~mask] = -1e9 # If the first token is [EOS], then the source is empty
                return torch.sigmoid(logitss)
            
    def classify_as_fallacy(self, text, classes=['ad hominem', 'ad populum', 'appeal to emotion', 'circular reasoning', 'equivocation', 'fallacy of credibility',
       'fallacy of extension', 'fallacy of logic', 'fallacy of relevance', 'false causality', 'false dilemma', 'faulty generalization','intentional', 'no fallacy']):
        inputs = self.fallacy_tokenizer(text, return_tensors='pt')
        inputs.to(device='cpu')
        self.fallacy_classifier.eval()
        with torch.no_grad():
            outputs = self.fallacy_c(**inputs)
        logits = outputs.logits
        probas = torch.nn.functional.softmax(logits, dim=1)
        if classes:
            return classes[torch.argmax(probas, dim=1).item()]