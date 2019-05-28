import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable

from .base_renderer import BaseRenderer
from ..utils import make_parameter as  MP
from ..vec import vdot, vnorm


class DiffuseRenderer(BaseRenderer):
    def __init__(self):
        self.albedo = MP([1,1,1])
        self.zero_ = MP([0])

    def render(self, info: dict):
        b = info['b']
        n = info['n']
        rd = info['rd']
        ll = info['ll']
        albedo = info['albedo']
        _, H, W = n.shape[:3]

        bb = b
        nn = n
        _, H, W = nn.shape[:3]
        bb = F.transpose(bb, (1, 2, 0))
        #mask = F.cast(bb, "float32")
        mask = F.where(bb, np.ones((H, W, 1), nn.dtype), np.zeros((H, W, 1), nn.dtype))
        nn = F.transpose(nn, (1, 2, 0))
        albedo = F.transpose(albedo, (1, 2, 0))
        for l in ll:
            di, lc = l.illuminate(info)
            di = vnorm(di)
            di = F.transpose(di, (1, 2, 0))
            lc = F.transpose(lc, (1, 2, 0))
            dd = nn * -di
            dd = F.sum(dd, axis=2).reshape((H, W, 1))
            dd = F.maximum(dd, self.zero_)

        img = F.clip(lc * albedo * dd * mask, 0.0, 1.0)

        return img