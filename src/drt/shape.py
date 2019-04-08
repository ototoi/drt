import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable


def vdot(a, b):
    m = a * b
    return F.sum(m, axis=0)


def vnorm(a):
    l = F.sqrt(vdot(a, a))
    return a / l


def is_positive(a):
    return F.relu(F.sign(a))


class Shape(object):
    def __init__(self):
        pass
    def intersect(self, ro, rd, t0, t1):
        pass


class SphereShape(Shape):
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
        so = F.broadcast_to(so.reshape((3, 1, 1)), (C, H, W))
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
        MASK_D = is_positive(D)
        #print("MASK", MASK.shape, MASK.dtype)
        #zero = np.zeros((1, H, W), np.float32)
        tx = -B - F.sqrt(F.absolute(D))
        MASK_T0 = is_positive(tx - t0)
        MASK_T1 = is_positive(t1 - tx)
        #print("tx", tx.shape)
        #print("t0", t0.shape)
        #print("t1", t1.shape)
        #print("MASK_D", MASK_D.shape)
        #print("MASK_T0", MASK_T0.shape)
        #print("MASK_T1", MASK_T1.shape)

        b = F.cast(MASK_D * MASK_T0 * MASK_T1, 'bool')
        t = F.where(b, tx, t1)
        #print(t.shape, t.dtype)
        p = ro + t * rd
        #print(p.shape, p.dtype)
        n = vnorm(p - so)
        #print(n.shape, n.dtype)
        return b, t, p, n


        





