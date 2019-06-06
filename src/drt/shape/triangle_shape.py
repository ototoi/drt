import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable


from .base_shape import BaseShape
from ..vec import vdot, vnorm, vcross
from ..utils import make_parameter as MP


def is_positive(a):
    B, _, H, W = a.shape[:4]
    return F.relu(F.sign(a)).reshape((B, 1, H, W))


class TriangleShape(BaseShape):
    def __init__(self, p0, p1, p2):
        super(TriangleShape, self).__init__()
        with self.init_scope():
            self.p0 = MP(p0)
            self.p1 = MP(p1)
            self.p2 = MP(p2)
            self.eps = MP([1e-8])


    def intersect(self, ro, rd, t0, t1):
        B, _, H, W = ro.shape[:4]
        p0 = F.broadcast_to(self.p0.reshape((1, 3, 1, 1)), (B, 3, H, W))
        p1 = F.broadcast_to(self.p1.reshape((1, 3, 1, 1)), (B, 3, H, W))
        p2 = F.broadcast_to(self.p2.reshape((1, 3, 1, 1)), (B, 3, H, W))
        eps = self.eps.reshape((1, 1, 1, 1))

        so = p0
        sn = vcross(p1 - p0, p2 - p0)

        # print(p0.shape)
        # print(sn.shape)
        aa = so - ro

        A = vdot(aa, sn)
        B = vdot(rd, sn) + eps
        #print(A.shape, B.shape)
        tx = A / B
        p = ro + tx * rd
        n01 = vcross(p0 - p, p1 - p)
        n12 = vcross(p1 - p, p2 - p)
        n20 = vcross(p2 - p, p0 - p)

        MASK_P = is_positive(vdot(n01, n12))
        MASK_Q = is_positive(vdot(n12, n20))
        MASK_R = is_positive(vdot(n20, n01))

        # is_positive(F.absolute(B).reshape((B, 1, H, W)))
        MASK_B = is_positive(F.absolute(B))
        # print(MASK_B.shape)

        MASK_T0 = is_positive(tx - t0)
        MASK_T1 = is_positive(t1 - tx)

        b = F.cast(MASK_P * MASK_Q * MASK_R *
                   MASK_B * MASK_T0 * MASK_T1, 'bool')
        #print("MASK_B", MASK_B.shape)
        #print("b", b.shape)
        t = F.where(b, tx, t1)
        p = ro + t * rd

        #print(p.shape, p.dtype)
        bn = F.cast(is_positive(vdot(rd, sn)), 'bool')
        n = F.where(bn, -sn, sn)
        #print(n.shape, n.dtype)
        return {'b': b, 't': t, 'p': p, 'n': n}
