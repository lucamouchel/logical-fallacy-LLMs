import torch

logits = torch.tensor([[ 3.3003, -2.8276],
        [-0.0659,  0.2377]])

probas = torch.sigmoid(logits)
print(probas)
preds = torch.argmax(probas, dim=1).tolist()
print(preds)