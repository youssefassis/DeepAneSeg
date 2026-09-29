import torch
from deepaneseg.utils import get_logger

logger = get_logger('Model Configuration')


class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """
    def __init__(self, **kwargs):
        logger.info("DiceCoefficient metric is used")

    @torch.no_grad()
    def __call__(self, inputs, targets, smooth=1e-8):
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return dice

class IoUCoefficient:
    def __init__(self, **kwargs):
        logger.info("IoUCoefficient metric is used")

    @torch.no_grad()
    def __call__(self, inputs, targets, smooth=1e-8):
        """
        Computes and return IoU for a given inputs and targets tensors
        """
        intersection = (inputs * targets).sum()
        iou = (intersection + smooth)/(inputs.sum() + targets.sum() - intersection + smooth)
        return iou

class Kappa:
    """
    Computes Kappa score
    """
    def __init__(self):
        logger.info("Cohen's Kappa metric is used")

    @torch.no_grad()
    def __call__(self, inputs, targets, smooth=1e-8):
        N = torch.numel(inputs)
        numerator = 2 * (inputs * targets).sum() - (targets.sum() * inputs.sum()) / N
        disc = targets.sum() + inputs.sum() - 2 * (inputs * targets).sum() / N
        kappa = (numerator+smooth) / (disc+smooth)
        return kappa

def cm(y, t): #, num_classes=1):
    '''Compute TP, FP, FN, TN.
    Args:
      y: (tensor) model outputs sized [N,].
      t: (tensor) labels targets sized [N,].
    Returns:
      (tensor): TP, FP, FN, TN, sized [num_classes,].
    '''
    tp = ((y==1) & (t==1)).sum()
    fp = ((y==1) & (t==0)).sum()
    fn = ((y==0) & (t==1)).sum()
    tn = ((y==0) & (t==0)).sum()
    return tp, fp, fn, tn

class Tversky:
    """
    Computes Tversky loss
    """
    def __init__(self, **kwargs):
        logger.info("Tversky metric is used")
        pass
    @torch.no_grad()
    def __call__(self, inputs, targets, alpha=0.7, beta=0.5, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        return Tversky

#################################################################################################
def get_metric(name):
    """
    Returns the evaluation metric function based on provided name
    :return: an instance of the evaluation metric
    """
    if isinstance(name, list):
        metrics = []
        if "Dice" in name:
            metrics.append(DiceCoefficient())
        if "Kappa" in name:
            metrics.append(Kappa())
        return metrics
    else:
        if name == "Dice":
            return DiceCoefficient()
        elif name == "IoU":
            return IoUCoefficient()
        elif name == "Kappa":
            return Kappa()
        elif name == "Tversky":
            return Tversky()
        else:
            raise RuntimeError(f"Unsupported Metric: '{name}'")
