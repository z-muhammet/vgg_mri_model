import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.weight = weight

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        n_classes = pred.size(-1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        if self.weight is not None:
            weight = self.weight[target].unsqueeze(1)
            loss = - (true_dist * log_probs) * weight
        else:
            loss = - true_dist * log_probs
        return loss.sum(dim=1).mean() 