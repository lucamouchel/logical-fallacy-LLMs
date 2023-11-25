from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5EncoderModel
import torch
def evaluate(sentence):
        model = T5EncoderModel.from_pretrained('google/t5-v1_1-small')
        model.D = model.shared.embedding_dim
        linear = torch.nn.Linear(model.D, 1)
        checkpoint = torch.load('runs/train/model/ckp_25000.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        linear.load_state_dict(checkpoint['linear'])
        step = checkpoint['step']
        checkpoint.clear()

        tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-small')
        
        input_ids = tokenizer(sentence, return_tensors='pt')['input_ids']
        attention_mask = (input_ids != tokenizer.pad_token_id).float()

        
        flattened_sourcess_input_ids = input_ids.view(-1, input_ids.size(-1)) # (B * C, L)
        flattened_sourcess_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) # (B * C, L)
        
        with torch.no_grad():
            outputs = model(
                input_ids=flattened_sourcess_input_ids,
                attention_mask=flattened_sourcess_attention_mask,
            )
            
            last_indices = flattened_sourcess_attention_mask.sum(dim=1, keepdim=True) - 1 # (B * C, 1)
            last_indices = last_indices.unsqueeze(-1).expand(-1, -1, model.D)
            last_indices = last_indices.to(torch.int64)

            last_hidden_state = outputs.last_hidden_state.to('cpu')
            hidden = last_hidden_state.gather(dim=1, index=last_indices).squeeze(1) # (B * C, D)
            flattened_logitss = linear(hidden).squeeze(-1) # (B * C)
            logitss = flattened_logitss.view(input_ids.size(0), -1) # (B, C)
            # WARNING: This only works on T5 and Llama tokenizers!
            mask = (attention_mask.sum(dim=-1) > 1) # (B, C)
            logitss[~mask] = -1e9 # If the first token is [EOS], then the source is empty
            return torch.sigmoid(logitss)
               
print(evaluate(
            "Claiming that journalism does not need subsidies implies that all journalists are financially stable and independent, which is not the case.",
    ))