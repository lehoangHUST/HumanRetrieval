from utils.utils import convert_categorial
from utils.dataset import ClothesClassificationDataset
import torch
import torch.nn.functional as F
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# accuracy for type clothes
def accuracy_type(output, target, dataset: ClothesClassificationDataset):
    """Computes the accuracy for clothes type"""
    with torch.no_grad():
        batch_size = target.size(0)

        # split target
        type_target = target[:, :] # torch.Tensor(batch, dataset.type_len)
        
        #type_target = convert_categorial(type_target) # torch.Tensor(batch)

        # convert output
        type_output = output # torch.Tensor(batch, dataset.type_len)
        type_output = torch.softmax(type_output, dim=1)
        type_idx_output = torch.argmax(type_output, dim=1) # torch.Tensor(batch)

        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)
        type_output = F.one_hot(type_idx_output, dataset.type_len)
        confusion_type_acc = type_output / type_target
        
        type_matching = torch.tensor([0] * dataset.type_len)
        for i in range(dataset.type_len):
            type_matching[i] = torch.sum(confusion_type_acc[:, i] == 1) # torch.Tensor(type_len)
        
    return type_matching # torch.Tensor(dataset.type_len)


# accuarcy for type color clothes
def accuracy_color(output, target, dataset: ClothesClassificationDataset):
    """Computes the accuracy for clothes type"""
    with torch.no_grad():
        batch_size = target.size(0)

        # split target
        color_target = target[:, :] # torch.Tensor(batch, dataset.color_len)
        color_output = output
        color_output = (color_output > 0.5).type(torch.int) # torch.Tensor(batch, dataset.color_len)


        # compute acc for color
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)
        confusion_vector = color_output / color_target
        color_matching = torch.tensor([0] * dataset.color_len)
        for i in range(dataset.color_len):
            color_matching[i] = torch.sum(confusion_vector[:, i] == 1) # torch.Tensor(color_len)
    return color_matching # torch.Tensor(color_len)

def fitness_type(metrics, weights=None, useloss=True):
    """ Model fitness as weighted combination of metrics
    weight for total_loss, type_loss, color_loss, type_acc, avg_color_acc
    default to use loss for monitoring
    if use loss to monitor, please set acc weights to zeros vice versa
    """
    if weights is not None:
        assert metrics.shape == weights.shape, "Metrics and weights combination must have same shape"
        if not isinstance(weights, np.ndarray):
            weights = np.asarray(weights)
    if useloss:
        if weights is None:
            weights = np.array([0.5, 0.0]) # default use loss not acc
            # the smaller the loss the better
        else:
            weights[1] = 0.0 # zeros out the acc weights
        return 100 - np.dot(metrics, weights) # -> the larger the value the better
    elif not useloss:
        if weights is None:
            weights = np.array([0.0, 0.5]) # the larger the acc the better
        else:
            weights[0] = 0.0 # zeros out the loss weights
        return np.dot(metrics, weights) # -> the larger the value the better

if __name__ == '__main__':
    pass