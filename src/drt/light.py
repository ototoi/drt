import numpy as np
import chainer
import chainer.functions as F 


def vdot(a, b):
    m = a * b
    return F.sum(m, axis=0)


class Light(object):
    def __init__(self):
        pass

    def illuminate(self, info):
        pass


class PointLight(Light):
    def __init__(self, origin):
        self.origin = origin
    
    def illuminate(self, info):
        p = info['p']
        C, H, W = p.shape[:3]
        lo = self.origin
        lo = F.broadcast_to(lo.reshape((3, 1, 1)), (C, H, W))
        di = p - lo
        dl = F.sqrt(vdot(di, di))

