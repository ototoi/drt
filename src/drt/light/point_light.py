from .base_light import BaseLight

import numpy as np
import chainer
import chainer.functions as F
from ..utils import make_parameter as  MP


class PointLight(BaseLight):
    def __init__(self, origin, color):
        super(PointLight, self).__init__()
        with self.init_scope():
            self.origin = MP(origin)
            self.color = MP(color)

    
    def illuminate(self, info):
        p = info['p']
        B, C, H, W = p.shape[:4]

        xp = chainer.backend.get_array_module(p)
        ll = xp.zeros((1,), np.float32)
        print(type(ll))

        lo = self.origin
        lc = self.color
        #print(lo.shape)
        lo = lo.reshape((1, 3, 1, 1))
        lc = lc.reshape((1, 3, 1, 1))
        lo = F.broadcast_to(lo, (B, C, H, W))
        lc = F.broadcast_to(lc, (B, C, H, W))
        di = p - lo
        #dl = F.sqrt(vdot(di, di))
        return di, lc