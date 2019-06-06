import numpy as np
import math


import numpy as np
import chainer.functions as F
import chainer.backend
from chainer import Variable
from ..utils import make_parameter as MP
from ..vec import vdot, vnorm, vcross


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

def create_axies(direction, up):
    zaxis = direction
    yaxis = up

    zaxis = zaxis.reshape((1, 3, 1, 1))
    yaxis = yaxis.reshape((1, 3, 1, 1))
    xaxis = vnorm(vcross(yaxis, zaxis))
    yaxis = vnorm(vcross(zaxis, xaxis))

    xaxis = xaxis.reshape((3,))
    yaxis = yaxis.reshape((3,))
    zaxis = zaxis.reshape((3,))
    return xaxis, yaxis, zaxis


class PerspectiveCamera(BaseCamera):
    def __init__(self, width, height, fov, origin, direction = [0, 0, 1], up = [0, 1, 0]):
        super(PerspectiveCamera, self).__init__()
        xaxis, yaxis, zaxis = create_axies(MP(direction), MP(up))
        with self.init_scope():
            self.width = width
            self.height = height
            self.fov = MP(fov)
            self.origin = MP(origin)
            self.xaxis = MP(xaxis.data)
            self.yaxis = MP(yaxis.data)
            self.zaxis = MP(zaxis.data)
            self.t0 = MP([0.01])
            self.t1 = MP([10000])


    def shoot(self):
        W = self.width
        H = self.height
        origin = self.origin
        xaxis = self.xaxis
        zaxis = self.zaxis
        yaxis = self.yaxis
        angle = self.fov
        t0 = self.t0
        t1 = self.t1

        xp = chainer.backend.get_array_module(origin)

        #zaxis = zaxis.reshape((1, 3, 1, 1))
        #yaxis = yaxis.reshape((1, 3, 1, 1))
        #xaxis = vnorm(vcross(yaxis, zaxis))
        #yaxis = vnorm(vcross(zaxis, xaxis))

        xaxis = vnorm(xaxis.reshape((1, 3, 1, 1))).reshape((1, 1, 3))
        yaxis = vnorm(yaxis.reshape((1, 3, 1, 1))).reshape((1, 1, 3))
        zaxis = vnorm(zaxis.reshape((1, 3, 1, 1))).reshape((1, 1, 3))
        
        #print(angle.shape)
        angle = (angle / 2) * math.pi / 180.0
        HH = F.tan(angle)
        ro = origin
        ro = F.tile(ro, (H, W, 1))
        ro = F.transpose(ro, (2, 0, 1))
        
        yy = xp.tile(xp.arange(H, dtype=np.float32).reshape((H, 1, 1)), (1, W, 1))  #(H, W, 1)
        xx = xp.tile(xp.arange(W, dtype=np.float32).reshape((1, W, 1)), (H, 1, 1))  #(H, W, 1)
        yy = (1 - 2 * (yy + 0.5) / H) * HH
        xx = (2 * (xx + 0.5) / W - 1) * HH
        rd = xx * xaxis + yy * yaxis + F.broadcast_to(zaxis, (H, W, 3))
        rd = F.transpose(rd, (2, 0, 1))
        rd = rd.reshape((1, 3, H, W))
        rd = vnorm(rd)
        rd = rd.reshape((3, H, W))

        t0 = F.broadcast_to(t0.reshape((1, 1, 1)), (1, H, W))
        t1 = F.broadcast_to(t1.reshape((1, 1, 1)), (1, H, W))
 
        #print(ro.shape, ro.dtype)
        #print(rd.shape, rd.dtype)
        return ro, rd, t0, t1
