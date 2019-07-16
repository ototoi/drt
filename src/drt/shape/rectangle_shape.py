import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable

from .base_shape import BaseShape
from ..vec import vdot, vnorm
from ..utils import make_parameter as MP

from .triangle_shape import TriangleShape


class RectangleShape(BaseShape):
    """
    RectangleShape:
    """

    def __init__(self, p0, p1, p2, p3):
        super(RectangleShape, self).__init__()
        with self.init_scope():
            self.p0 = MP(p0)
            self.p1 = MP(p1)
            self.p2 = MP(p2)
            self.p3 = MP(p3)
            self.tri0 = TriangleShape(p0, p1, p2, [0])
            self.tri1 = TriangleShape(p0, p2, p3, [1])

    def intersect(self, ro, rd, t0, t1):
        t = t1
        info = self.tri0.intersect(ro, rd, t0, t)

        b = info['b']
        t = info['t']
        iinfo = self.tri1.intersect(ro, rd, t0, t)
        bb = iinfo['b']
        tt = iinfo['t']
        b = b + bb
        t = tt
        for k in iinfo.keys():
            info[k] = F.where(bb, iinfo[k], info[k])
        info['b'] = b
        info['t'] = t

        return info

    def to_gpu(self):
        self.tri0.to_gpu()
        self.tri1.to_gpu()
