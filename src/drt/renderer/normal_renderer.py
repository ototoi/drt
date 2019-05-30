import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable

from .base_renderer import BaseRenderer
from ..utils import make_parameter as  MP

class NormalRenderer(BaseRenderer):
    def __init__(self):
        pass

    def render(self, info: dict):
        b = info['b']
        n = info['n']
        B, _, H, W = n.shape[:4]
        b = F.transpose(b, (0, 2, 3, 1))
        mask = F.where(b, np.ones((B, H, W, 1), n.dtype), np.zeros((B, H, W, 1), n.dtype))
        n = F.transpose(n, (0, 2, 3, 1))
        n = 0.5 * (n + 1)
        img = n * mask #

        return img