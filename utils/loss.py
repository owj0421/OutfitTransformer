import torch
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
    

def triplet_margin_loss_with_multiple_negatives(
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        margin: int = 2.0,
        reduction: str = 'mean'
        ) -> torch.Tensor:
    anchors = anchor.unsqueeze(1).expand_as(negatives)
    positives = positive.expand_as(negatives)
    loss = nn.TripletMarginLoss(margin=margin, reduction=reduction)(anchors, positives, negatives)

    return loss
    

def focal_loss(
        y_prob: torch.Tensor,
        y_true: torch.Tensor,
        alpha: float = 0.5,
        gamma: float = 2,
        reduction: str = "mean",
        ) -> torch.Tensor:
    ce_loss = F.binary_cross_entropy_with_logits(y_prob, y_true, reduction="none")
    p_t = y_prob * y_true + (1 - y_prob) * (1 - y_true)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = alpha_t * loss

    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'")
    
    return loss