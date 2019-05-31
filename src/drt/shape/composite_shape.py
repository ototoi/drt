import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable

from .base_shape import BaseShape
from ..vec import vdot, vnorm


class CompositeShape(BaseShape):
    def __init__(self, shapes):
        super(CompositeShape, self).__init__()
        self.shapes = shapes

    def intersect(self, ro, rd, t0, t1):
        s = self.shapes[0]
        t = t1
        info = s.intersect(ro, rd, t0, t)
        
        b = info['b']
        t = info['t']
        for s in self.shapes[1:]:
            iinfo = s.intersect(ro, rd, t0, t)
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
        for s in self.shapes:
            s.to_gpu()
    
