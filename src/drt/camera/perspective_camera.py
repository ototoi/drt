import numpy as np
import math


import numpy as np
import chainer.functions as F
import chainer.backend
from chainer import Variable
from ..utils import make_parameter as MP


from .base_camera import BaseCamera


class PerspectiveCamera(BaseCamera):
    def __init__(self, width, height, fov, P):
        super(PerspectiveCamera, self).__init__()
        self.width = width
        self.height = height
        self.fov = fov
        self.P = MP(P)

    def shoot(self):
        W = self.width
        H = self.height
        P = self.P
        xp = chainer.backend.get_array_module(P)

        angle = self.fov
        angle = (angle / 2) * math.pi / 180.0
        HH = math.tan(angle)
        ro = P
        ro = F.tile(ro, (H, W, 1))
        rd = xp.zeros((H, W, 3), np.float32)
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

        ro = F.transpose(ro, (2, 0, 1))
        rd = F.transpose(rd, (2, 0, 1))
        #print(ro.shape, ro.dtype)
        #print(rd.shape, rd.dtype)
        return ro, rd
