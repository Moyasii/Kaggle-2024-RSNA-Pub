import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False, flatten=True):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps
        self.flatten = flatten

    def _gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def forward(self, x):
        ret = self._gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            return ret[:, :, 0, 0]
        else:
            return ret

    def __repr__(self):
        if not isinstance(self.p, int):
            return (self.__class__.__name__ + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")
        else:
            return (self.__class__.__name__ + f"(p={self.p:.4f},eps={self.eps})")
