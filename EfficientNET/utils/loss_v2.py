from utils.utils import convert_categorial

import torch.nn.functional as F
import torch.nn as nn
import torch

class BCELabelSmoothing(nn.BCELoss):
    def __init__(self, smoothing=0.1):
        super(BCELabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, pred, target):
        with torch.no_grad():
            true_dist: torch.Tensor = target.clone()
            true_dist *= self.confidence
            true_dist += (0.5 * self.smoothing)
        return F.binary_cross_entropy(pred, true_dist)


class CrossEntropyLabelSmoothing(nn.Module):
    """
    This loss class works like CrossEntropyLoss from Pytorch with smoothing factor
    It accepts target as a number and pred is logits
    Label smoothing from Inception-v2 paper
    Explain:
        label' = (1-smoothing) * label + smoothing/num_labels
    """
    def __init__(self, smoothing=0.1, dim=-1):
        super(CrossEntropyLabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim) # apply log(softmax(x)) on the logits
        with torch.no_grad():
            true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(self.dim)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


  
# Loss for type clothes
class Loss_typ():
  def __init__(self, lsCE=False):
      super(Loss_typ, self).__init__()
      if lsCE:
        self.type_criterion = CrossEntropyLabelSmoothing()
      else:
        self.type_criterion = nn.CrossEntropyLoss()

  
  def __call__(self, pred, targets):
    # pred: output after forward by EfficientNet
    # targets: real output 
    targets = convert_categorial(targets)
    type_loss = self.type_criterion(pred, targets)
    return type_loss


# Loss for color clothes
class Loss_color():
  def __init__(self, lsBCE=False):
    super(Loss_color, self).__init__()
    if lsBCE:
        self.color_criterion = BCELabelSmoothing()
    else:
        self.color_criterion = nn.BCELoss()
  
  def __call__(self, pred, targets):
    # pred: output after forward by EfficientNet
    # targes: real output 
    color_loss = self.color_criterion(pred, targets)
    return color_loss