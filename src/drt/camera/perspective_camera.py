import numpy as np
import math


import numpy as np
import chainer.functions as F
import chainer.backend
from chainer import Variable
from ..utils import make_parameter as MP
from ..vec import vdot, vnorm


from .base_camera import BaseCamera


"""
for y in range(H):
    yy = 1 - 2 * (y + 0.5) / H
    for x in range(W):
        xx = 2 * (x + 0.5) / W - 1
        yyy = yy * HH
        xxx = xx * HH
        r = xp.array([xxx, yyy, 1], np.float32)
        r = r / xp.linalg.norm(r)
        #r = np.array([0, 0, 1], np.float32)
        rd[y, x, :] = r
"""


class PerspectiveCamera(BaseCamera):
    def __init__(self, width, height, fov, P):
        super(PerspectiveCamera, self).__init__()
        with self.init_scope():
            self.width = width
            self.height = height
            self.fov = fov
            self.P = MP(P)
            self.t0 = MP([0.01])
            self.t1 = MP([10000])


    def shoot(self):
        W = self.width
        H = self.height
        P = self.P
        t0 = self.t0
        t1 = self.t1
        xp = chainer.backend.get_array_module(P)

        angle = self.fov
        angle = (angle / 2) * math.pi / 180.0
        HH = math.tan(angle)
        ro = P
        ro = F.tile(ro, (H, W, 1))
        rd = xp.ones((H, W, 3), dtype=np.float32)
        
        yy = xp.tile(xp.arange(H, dtype=np.float32).reshape((H, 1, 1)), (1, W, 1))
        xx = xp.tile(xp.arange(W, dtype=np.float32).reshape((1, W, 1)), (H, 1, 1))
        yy = (1 - 2 * (yy + 0.5) / H) * HH
        xx = (2 * (xx + 0.5) / W - 1) * HH 
        #print(rd.shape, xx.shape, yy.shape)
        rd[:, :, 0] = xx[:, :, 0]
        rd[:, :, 1] = yy[:, :, 0]
        
        ro = F.transpose(ro, (2, 0, 1))
        rd = F.transpose(rd, (2, 0, 1))
        rd = rd.reshape((1, 3, H, W))
        rd = vnorm(rd)
        rd = rd.reshape((3, H, W))

        t0 = F.broadcast_to(t0.reshape((1, 1, 1)), (1, H, W))
        t1 = F.broadcast_to(t1.reshape((1, 1, 1)), (1, H, W))
 
        #print(ro.shape, ro.dtype)
        #print(rd.shape, rd.dtype)
        return ro, rd, t0, t1
