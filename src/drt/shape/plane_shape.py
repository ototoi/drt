import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable


from .base_shape import BaseShape
from ..vec import vdot, vnorm


def is_positive(a):
    return F.relu(F.sign(a))


class PlaneShape(BaseShape):
    def __init__(self, origin, normal):
        self.origin = origin
        self.normal = normal
    
    def intersect(self, ro, rd, t0, t1):
        """
        dot(so - p, sn) = 0
        ro + t * rd = p
        """
        # dot(so, sn) - dot(p, sn) = 0
        # dot(so, sn) - dot((ro + t * rd), sn) = 0
        # dot(so, sn) - dot(ro, sn) - dot(rd, sn) * t = 0
        # t = ((so, sn) - (ro, sn)) / (rd, n)
        # t = (so - ro, sn) / (rd, sn)
        C, H, W = ro.shape[:3]
        so = self.origin
        so = F.broadcast_to(so.reshape((3, 1, 1)), (C, H, W))
        sn = self.normal
        sn = F.broadcast_to(sn.reshape((3, 1, 1)), (C, H, W))
        A = vdot(so - ro, sn)
        B = vdot(rd, sn)
        tx = A / B
        MASK_B = is_positive(F.absolute(B)).reshape((1, H, W))
        MASK_T0 = is_positive(tx - t0).reshape((1, H, W))
        MASK_T1 = is_positive(t1 - tx).reshape((1, H, W))

        b = F.cast(MASK_B * MASK_T0 * MASK_T1, 'bool')
        #print("MASK_B", MASK_B.shape)
        #print("b", b.shape)
        t = F.where(b, tx, t1)

        p = ro + tx * rd
        #print(p.shape, p.dtype)
        bn = F.cast(is_positive(vdot(rd, sn)).reshape((1, H, W)), 'bool')
        n = F.where(bn, -sn, sn)
        #print(n.shape, n.dtype)
        return b, t, p, n