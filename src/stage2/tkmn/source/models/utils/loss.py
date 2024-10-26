from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class FocalLoss(nn.Module):
    # reference: https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (torch.Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class RSNA2024Loss(nn.Module):
    def __init__(
        self,
        conditions: list[str] = ['spinal_canal_stenosis', 'left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing',
                                 'left_subarticular_stenosis', 'right_subarticular_stenosis'],
        levels: list[str] = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1'],
        ce_loss: dict = dict(name='CrossEntropyLoss', weight=[1.0, 2.0, 4.0]),
        condition_weight: Optional[list[float]] = None,
        sevear_loss: bool = False,
        overall_loss_weight: float = 1.0,
        sevear_loss_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.conditions = conditions
        self.levels = levels
        self.sevear_loss = sevear_loss
        self.overall_loss_weight = overall_loss_weight
        self.sevear_loss_weight = sevear_loss_weight

        ce_loss_name = ce_loss.pop('name')
        if ce_loss_name == 'CrossEntropyLoss':
            if 'weight' in ce_loss:
                weight = ce_loss.pop('weight')
                ce_loss['weight'] = torch.tensor(weight)
            self.ce_loss = nn.CrossEntropyLoss(**ce_loss)
        elif ce_loss_name == 'FocalLoss':
            if 'alpha' in ce_loss:
                alpha = ce_loss.pop('alpha')
                ce_loss['alpha'] = torch.tensor(alpha)
            self.ce_loss = FocalLoss(**ce_loss)
        else:
            raise ValueError(f'{ce_loss_name} is not supported.')

        if condition_weight is None:
            condition_weight = [1.0] * len(conditions)
        self.condition_weight = condition_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = dict()
        partial_losses = []
        for cond_idx, cond in enumerate(self.conditions):
            for level_idx, level in enumerate(self.levels):
                logit = logits[:, cond_idx, level_idx]
                target = targets[:, cond_idx, level_idx]
                if torch.any(target != -100):
                    partial_loss = self.ce_loss(logit, target) * self.condition_weight[cond_idx]
                    partial_losses.append(partial_loss)

        overall_loss = torch.mean(torch.stack(partial_losses))
        losses['overall_loss'] = overall_loss.item()

        if self.sevear_loss:
            # 仮実装（脊柱管狭窄の重症予測を1に近づけるようなロスを追加する）
            # 各部位の重度の確率を抽出
            scs_idx = self.conditions.index('spinal_canal_stenosis')
            preds = logits.softmax(dim=-1)  # (batch_size, num_degenerative_conditions, num_segments, num_classes)
            severe_probs = preds[:, scs_idx, :, 2]  # (batch_size, num_segments)

            # 教師ラベルで重度とされている部位を特定
            has_severe_segment = (targets[:, scs_idx, :] == 2).float()  # (batch_size, num_segments)

            # 重度とされる部位における予測確率の平均を計算
            severe_prob_mean = (severe_probs * has_severe_segment).sum(dim=1) / has_severe_segment.sum(dim=1).clamp(min=1e-8)  # (batch_size)
            # all_severe_prob_mean = severe_probs.mean(dim=1)  # (batch_size)

            any_severe = has_severe_segment.sum(dim=1).gt(0).float()  # (batch_size)

            # log lossの計算
            sevear_loss = - (any_severe * torch.log(severe_prob_mean + 1e-8)).mean()
            losses['sevear_loss'] = sevear_loss.item()

            # weighted sum
            losses['loss'] = self.overall_loss_weight * overall_loss + self.sevear_loss_weight * sevear_loss
        else:
            losses['loss'] = overall_loss
        return losses


if __name__ == '__main__':
    rsna2024_loss = RSNA2024Loss(ce_loss={"name": "CrossEntropyLoss", "weight": [1.0, 2.0, 4.0]},
                                 conditions=['spinal_canal_stenosis', 'left_neural_foraminal_narrowing',
                                             'right_neural_foraminal_narrowing', 'left_subarticular_stenosis', 'right_subarticular_stenosis'],
                                 condition_weight=[1.0, 0.25, 0.25, 0.25, 0.25],
                                 levels=['any'],
                                 sevear_loss=True)
    rsna2024_loss(torch.randn(2, 5, 1, 3), torch.randint(0, 3, (2, 5, 1)))
