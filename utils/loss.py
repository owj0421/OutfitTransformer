import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

def info_nce(anchor, positive, negatives, temparature=0.07, reduction='mean'):
    anc, pos, negs = anchor.unsqueeze(1), positive, negatives

    anc = F.normalize(anc, dim=-1)
    pos = F.normalize(pos, dim=-1)
    negs = F.normalize(negs, dim=-1)

    pos_logit = torch.sum(anc * pos, dim=-1, keepdim=False)
    neg_logits = torch.sum(anc * negs, dim=-1, keepdim=False)

    logits = torch.cat([pos_logit, neg_logits], dim=-1)
    logits = torch.softmax(logits / temparature, dim=-1)
    labels = torch.zeros(len(logits), dtype=torch.long, device=logits.device)

    loss = F.cross_entropy(logits, labels, reduction=reduction)
    return loss


class TripletMarginLossWithMultipleNegatives(nn.Module):
    def __init__(self, margin=2.0):
        super(TripletMarginLossWithMultipleNegatives, self).__init__()
        self.margin = margin
        self.loss_func = nn.TripletMarginLoss(margin=self.margin, reduction='mean')

    def forward(self, anchor, positive, negatives):
        anchors = anchor.unsqueeze(1).expand_as(negatives)
        positives = positive.expand_as(negatives)
        loss = self.loss_func(anchors, positives, negatives)
        
        return loss 