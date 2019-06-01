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

    def render(self, info: dict):
        b = info['b']
        n = info['n']
        # rd = info['rd']
        ll = info['ll']
        albedo = info['albedo']

        xp = chainer.backend.get_array_module(albedo)

        B, _, H, W = n.shape[:4]

        n = F.transpose(vnorm(n), (0, 2, 3, 1))
        b = F.transpose(b, (0, 2, 3, 1))
        albedo = F.transpose(albedo, (0, 2, 3, 1))
        #mask = F.cast(bb, "float32")
        mask = F.where(b, xp.ones((B, H, W, 1), n.dtype), xp.zeros((B, H, W, 1), n.dtype))
        
        for l in ll:
            di, lc = l.illuminate(info)

            di = vnorm(di)
            di = F.transpose(di, (0, 2, 3, 1))
            lc = F.transpose(lc, (0, 2, 3, 1))
            dd = n * -di
            dd = F.sum(dd, axis=3).reshape((B, H, W, 1))
            dd = F.relu(dd)

        img = albedo * dd * mask
        img = F.transpose(img, (0, 3, 1, 2))

        return img