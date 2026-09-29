import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from deepaneseg.utils import get_logger

logger = get_logger('Model Configuration')
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        logger.info("Dice Loss is used")

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        logger.info("BCEDiceLoss is used")

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        dice_bce = BCE + dice_loss

        return dice_bce

class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        logger.info("WeightedCrossEntropyLoss is used")


    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        logger.info("IoU Loss is used")

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + smooth)/(union + smooth)
        return 1 - IoU

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        logger.info("Focal Loss is used")

    def forward(self, inputs, targets, alpha=0.8, gamma=2):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        #first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        bce_exp = torch.exp(-BCE)
        focal_loss = alpha * (1-bce_exp)**gamma * BCE
        return focal_loss

class TverskyLoss(nn.Module):
    def __init__(self):
        super(TverskyLoss, self).__init__()
        logger.info("TverskyLoss is used")

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        return 1 - Tversky

class FocalTverskyLoss(nn.Module):
    def __init__(self):
        super(FocalTverskyLoss, self).__init__()
        logger.info("FocalTverskyLoss is used")

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        focal_tversky = (1 - Tversky)**gamma

        return focal_tversky

class KappaLoss(nn.Module):
    def __init__(self):
        super(KappaLoss, self).__init__()
        logger.info("Kappa Loss is used")

    def forward(self, inputs, targets):
        N = torch.numel(inputs)
        numerator = 2 * (inputs * targets).sum() - (targets.sum() * inputs.sum()) / N
        disc = targets.sum() + inputs.sum() - 2 * (inputs* targets).sum() / N
        kappa = numerator / disc
        return 1 - kappa

#######################################################################################################################

def get_loss_criterion(name):
    assert name is not None, 'Could not find loss function'

    if name == "BCELoss":
        logger.info("BCELoss is used")
        return nn.BCELoss()
    elif name == "FocalLoss":
        return FocalLoss()
    elif name == 'DiceLoss':
        return DiceLoss()
    elif name == 'BCEDiceLoss':
        return BCEDiceLoss()
    elif name == 'TverskyLoss':
        return TverskyLoss()
    elif name == 'FocalTverskyLoss':
        return FocalTverskyLoss()
    elif name == 'BCEWithLogitsLoss':
        logger.info("BCEWithLogitsLoss is used")
        return nn.BCEWithLogitsLoss()
    elif name == 'WeightedCrossEntropyLoss':
        return WeightedCrossEntropyLoss()
    elif name == 'KappaLoss':
        return KappaLoss()
    elif name == 'MSELoss':
        logger.info("MSELoss is used")
        return nn.MSELoss()
    elif name == "IoULoss":
        return IoULoss()
    elif name == 'SmoothL1Loss':
        logger.info("SmoothL1Loss is used")
        return nn.SmoothL1Loss()
    elif name == 'L1Loss':
        logger.info("L1Loss is used")
        return nn.L1Loss()
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")

