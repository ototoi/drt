import numpy as np
import math


import numpy as np
import chainer.functions as F
import chainer.backend
from chainer import Variable


from .base_camera import BaseCamera


class PerspectiveCamera(BaseCamera):
    def __init__(self, width, height, fov, P):
        self.width = width
        self.height = height
        self.fov = fov
        self.P = P

    def shoot(self):
        W = self.width
        H = self.height
        P = self.P
        angle = self.fov
        angle = (angle / 2) * math.pi / 180.0
        HH = math.tan(angle)
        ro = np.array([P[0], P[1], P[2]], dtype=np.float32)
        ro = np.tile(ro, (H, W, 1))
        rd = np.zeros((H, W, 3), np.float32)
        for y in range(H):
            yy = 1 - 2 * (y + 0.5) / H
            for x in range(W):
                xx = 2 * (x + 0.5) / W - 1
                yyy = yy * HH
                xxx = xx * HH
                r = np.array([xxx, yyy, 1], np.float32)
                r = r / np.linalg.norm(r)
                #r = np.array([0, 0, 1], np.float32)
                rd[y, x, :] = r

        ro = np.transpose(ro, (2, 0, 1))
        rd = np.transpose(rd, (2, 0, 1))

        return Variable(ro), Variable(rd)
