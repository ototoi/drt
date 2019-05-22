from .base_light import BaseLight

import numpy as np
import chainer
import chainer.functions as F 


class PointLight(BaseLight):
    def __init__(self, origin, color):
        self.origin = origin
        self.color = color

    
    def illuminate(self, info):
        p = info['p']
        C, H, W = p.shape[:3]
        lo = self.origin
        lo = F.broadcast_to(lo.reshape((3, 1, 1)), (C, H, W))
        lc = self.color
        lc = F.broadcast_to(lc.reshape((3, 1, 1)), (C, H, W))
        di = p - lo
        #dl = F.sqrt(vdot(di, di))
        return di, lc