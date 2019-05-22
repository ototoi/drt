import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable


from .base_shape import BaseShape
from ..vec import vdot, vnorm


def is_positive(a):
    return F.relu(F.sign(a))


class SphereShape(BaseShape):
    def __init__(self, origin, radius):
        self.origin = origin
        self.radius = radius

    def intersect(self, ro, rd, t0, t1):
        """
        r^2 = (x-x0)^2 + (y-y0)^2 + (z-z0)^2
        ro + t * rd = p
        """
        C, H, W = ro.shape[:3]
        so = self.origin
        so = so.reshape((3, 1, 1))
        #print("so", so.shape, so.dtype)
        #print(C, H, W)
        so = F.broadcast_to(so, (C, H, W))
        sr = self.radius
        sr2 = sr * sr
        sr2 = F.broadcast_to(sr2.reshape((1, 1, 1)), (1, H, W))
        rs = ro - so
        #print("so", so.shape, so.dtype)
        #print("rs", rs.shape, rs.dtype)
        B = vdot(rs, rd)
        #print("B", B.shape, B.dtype)
        B = vdot(rs, rd).reshape((1, H, W))
        C = vdot(rs, rs).reshape((1, H, W)) - sr2
        # - xp.broadcast_to(sr2, (1, H, W))
        #print("B", B.shape, B.dtype)
        #print("C", C.shape, C.dtype)
        D = B * B - C
        #print("D", D.shape, D.dtype)
        #MASK_D = D > zero
        MASK_D = is_positive(D).reshape((1, H, W))
        #print("MASK", MASK.shape, MASK.dtype)
        #zero = np.zeros((1, H, W), np.float32)
        tx = -B - F.sqrt(F.absolute(D))
        MASK_T0 = is_positive(tx - t0).reshape((1, H, W))
        MASK_T1 = is_positive(t1 - tx).reshape((1, H, W))

        b = F.cast(MASK_D * MASK_T0 * MASK_T1, 'bool')
        t = F.where(b, tx, t1)
        #print(t.shape, t.dtype)
        p = ro + t * rd
        #print(p.shape, p.dtype)
        n = vnorm(p - so)
        #print(n.shape, n.dtype)
        return b, t, p, n