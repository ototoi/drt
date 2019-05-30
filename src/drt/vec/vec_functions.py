import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable


def vdot(a, b):
    B, _, H, W = a.shape[:4] 
    m = a * b
    return F.sum(m, axis=1).reshape((B, 1, H, W)) #C


def vnorm(a):
    l = F.sqrt(vdot(a, a))
    return a / l


def vcross(a, b):
    B, _, H, W = a.shape[:4]
    x = a[:, 1, :, :]*b[:, 2, :, :] - a[:, 2, :, :]*b[:, 1, :, :]
    y = a[:, 2, :, :]*b[:, 0, :, :] - a[:, 0, :, :]*b[:, 2, :, :]
    z = a[:, 0, :, :]*b[:, 1, :, :] - a[:, 1, :, :]*b[:, 0, :, :]
    return F.concat([x, y, z], axis=1).reshape((B, 3, H, W))
