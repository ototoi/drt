import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable


from .base_shape import BaseShape
from ..vec import vdot, vnorm
from ..utils import make_parameter as MP


def is_positive(a):
    B, _, H, W = a.shape[:4]
    return F.relu(F.sign(a)).reshape((B, 1, H, W))


class PlaneShape(BaseShape):
    def __init__(self, origin, normal):
        self.origin = MP(origin)
        self.normal = MP(normal)

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
        B, C, H, W = ro.shape[:4]
        so = self.origin
        so = F.broadcast_to(so.reshape((1, 3, 1, 1)), (B, C, H, W))
        sn = self.normal
        sn = F.broadcast_to(sn.reshape((1, 3, 1, 1)), (B, C, H, W))
        A = vdot(so - ro, sn)
        B = vdot(rd, sn)
        tx = A / B
        MASK_B = is_positive(F.absolute(B)).reshape((B, 1, H, W))
        MASK_T0 = is_positive(tx - t0).reshape((B, 1, H, W))
        MASK_T1 = is_positive(t1 - tx).reshape((B, 1, H, W))

        b = F.cast(MASK_B * MASK_T0 * MASK_T1, 'bool')
        #print("MASK_B", MASK_B.shape)
        #print("b", b.shape)
        t = F.where(b, tx, t1)

        p = ro + tx * rd
        #print(p.shape, p.dtype)
        bn = F.cast(is_positive(vdot(rd, sn)).reshape((B, 1, H, W)), 'bool')
        n = F.where(bn, -sn, sn).reshape((B, 3, H, W))
        #print(n.shape, n.dtype)
        return {'b': b, 't': t, 'p': p, 'n': n}
