import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


def similiarity(x1, x2):
    # refer: https://www.kaggle.com/code/hengck23/mvccl-model-for-admani-dataset

    p12 = (x1 * x2).sum(-1)
    p1 = torch.sqrt((x1 * x1).sum(-1))
    p2 = torch.sqrt((x2 * x2).sum(-1))
    s = p12 / (p1 * p2 + 1e-6)
    return s


def criterion_global_consistency(x1, x1_projection, x2, x2_projection, alpha=-0.5):
    # refer: https://www.kaggle.com/code/hengck23/mvccl-model-for-admani-dataset

    loss = alpha * (similiarity(x1, x1_projection) + similiarity(x2, x2_projection))
    return loss


class BFWithLogitsLoss(nn.Module):
    """Binary Focal Loss"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(input, target, self.alpha, self.gamma, self.reduction)


class ConsistencyLoss(nn.Module):

    def __init__(self, alpha: float = -0.5, reduction: str = 'mean'):
        super().__init__()
        self._alpha = alpha
        self._reduction = reduction

    def forward(self,
                x1: torch.Tensor,
                x1_projection: torch.Tensor,
                x2: torch.Tensor,
                x2_projection: torch.Tensor) -> torch.Tensor:

        loss = criterion_global_consistency(x1, x1_projection, x2, x2_projection, self._alpha)
        if self._reduction == 'mean':
            return torch.mean(loss)
        elif self._reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


def _neg_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    epsilon = 1e-6
    pred = pred.sigmoid()
    pos_loss = torch.log(pred + epsilon) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred + epsilon) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    batch_size = len(pred)
    return loss / batch_size


class CenterNetFocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super().__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)
