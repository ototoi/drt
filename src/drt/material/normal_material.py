import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable

from .base_material import BaseMaterial

class NormalMaterial(BaseMaterial):
    def __init__(self):
        pass

    def render(self, info: dict):
        b = info['b']
        n = info['n']

        bb = b
        nn = n
        _, H, W = nn.shape[:3]
        bb = F.transpose(bb, (1, 2, 0))
        #mask = F.cast(bb, "float32")
        mask = F.where(bb, np.ones((H, W, 1), nn.dtype), np.zeros((H, W, 1), nn.dtype))
        nn = F.transpose(nn, (1, 2, 0))
        nn = 0.5 * (nn + 1)
        img = F.clip(nn * mask, 0.0, 1.0)

        return img