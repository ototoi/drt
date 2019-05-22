import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable


from .base_shape import BaseShape
from ..vec import vdot, vnorm


class CompositeShape(BaseShape):
    def __init__(self, shapes):
        self.shapes = shapes

    def intersect(self, ro, rd, t0, t1):
        s = self.shapes[0]
        t = t1
        b, t, p, n = s.intersect(ro, rd, t0, t)
        for s in self.shapes[1:]:
            bb, t, pp, nn = s.intersect(ro, rd, t0, t)
            p = F.where(bb, pp, p)
            n = F.where(bb, nn, n)
            b = b + bb
        return b, t, p, n