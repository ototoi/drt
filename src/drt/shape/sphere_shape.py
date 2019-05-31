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


class SphereShape(BaseShape):
    def __init__(self, origin, radius):
        super(SphereShape, self).__init__()
        self.origin = MP(origin)
        self.radius = MP(radius)

    def intersect(self, ro, rd, t0, t1):
        """
        r^2 = (x-x0)^2 + (y-y0)^2 + (z-z0)^2
        ro + t * rd = p
        """
        B, C, H, W = ro.shape[:4]
        so = self.origin
        so = F.broadcast_to(so.reshape((1, 3, 1, 1)), (B, C, H, W))
        sr = self.radius
        sr2 = sr * sr
        sr2 = F.broadcast_to(sr2.reshape((1, 1, 1, 1)), (B, 1, H, W))
        rs = ro - so  # (B, C, H, W)
        B = vdot(rs, rd)  # (B, C, H, W)
        B = vdot(rs, rd).reshape((B, 1, H, W))  # (B, 1, H, W)
        C = vdot(rs, rs).reshape((B, 1, H, W)) - sr2  # (B, 1, H, W)

        D = B * B - C
        MASK_D = is_positive(D)
        tx = -B - F.sqrt(F.absolute(D))
        MASK_T0 = is_positive(tx - t0)
        MASK_T1 = is_positive(t1 - tx)

        b = F.cast(MASK_D * MASK_T0 * MASK_T1, 'bool')
        t = F.where(b, tx, t1)
        p = ro + t * rd
        n = vnorm(p - so)
        return {'b': b, 't': t, 'p': p, 'n': n}
