import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable

import cv2


from .base_shape import BaseShape
from ..vec import vdot, vnorm, vcross
from ..utils import make_parameter as MP


def is_positive(a):
    B, _, H, W = a.shape[:4]
    return F.relu(F.sign(a)).reshape((B, 1, H, W))

def is_positive_(a):
    return a > 0

def vdot_(a, b, xp):
    B, _, H, W = a.shape[:4] 
    m = a * b
    return xp.sum(m, axis=1).reshape((B, 1, H, W)) #C


def vnorm_(a, xp):
    l = xp.rsqrt(vdot_(a, a, xp))
    return a * l


def vcross_(a, b, xp):
    B, _, H, W = a.shape[:4]
    x = a[:, 1, :, :]*b[:, 2, :, :] - a[:, 2, :, :]*b[:, 1, :, :]
    y = a[:, 2, :, :]*b[:, 0, :, :] - a[:, 0, :, :]*b[:, 2, :, :]
    z = a[:, 0, :, :]*b[:, 1, :, :] - a[:, 1, :, :]*b[:, 0, :, :]
    return xp.concatenate([x, y, z], axis=1).reshape((B, 3, H, W))


def vdot_e(a, b):
    m = a * b
    return F.sum(m)

def vcross_e(a, b):
    x = (a[1]*b[2] - a[2]*b[1]).reshape((1,))
    y = (a[2]*b[0] - a[0]*b[2]).reshape((1,))
    z = (a[0]*b[1] - a[1]*b[0]).reshape((1,))
    return F.concat([x, y, z], axis=0)

def vnorm_e(a):
    return a * F.rsqrt(vdot_e(a, a))

def where_(mask, m0, m1):
    return m1 + (m0-m1) * mask

"""
def save_boolean_img(path, mask):
    B, C, H, W = mask.shape[:4]
    mask = F.cast(mask, 'int').data
    mask = mask.reshape((H, W))
    mask = mask.astype(np.uint8) * 255
    cv2.imwrite(path, mask)
"""


class TriangleShape(BaseShape):
    """
    TriangleShape:
    """

    def __init__(self, p0, p1, p2, id):
        super(TriangleShape, self).__init__()
        p0 = MP(p0)
        p1 = MP(p1)
        p2 = MP(p2)
        fn = vnorm_e(vcross_e(p1 - p0, p2 - p0))
        with self.init_scope():
            self.p0 = p0
            self.p1 = p1
            self.p2 = p2
            self.fn = fn
            self.id = MP(id)

    def intersect(self, ro, rd, t0, t1):
        xp = chainer.backend.get_array_module(ro)
        BB, _, H, W = ro.shape[:4]
        
        p0 = F.broadcast_to(self.p0.reshape((1, 3, 1, 1)), (BB, 3, H, W))
        p1 = F.broadcast_to(self.p1.reshape((1, 3, 1, 1)), (BB, 3, H, W))
        p2 = F.broadcast_to(self.p2.reshape((1, 3, 1, 1)), (BB, 3, H, W))
        fn = F.broadcast_to(self.fn.reshape((1, 3, 1, 1)), (BB, 3, H, W))
        face_id = F.broadcast_to(self.id, (BB, 1, H, W))

        aa = p0 - ro

        A = vdot(aa, fn)
        B = vdot(rd, fn)                                #(1, 1, H, W)
        #B = F.sign(B) * F.maximum(F.absolute(B), eps)   #

        tx = A / B
        p = ro + tx * rd
        """
        n01 = vnorm(vcross(p0 - p, p1 - p))
        n12 = vnorm(vcross(p1 - p, p2 - p))
        n20 = vnorm(vcross(p2 - p, p0 - p))

        MASK_P = is_positive(vdot(n01, n12) + eps)
        MASK_Q = is_positive(vdot(n12, n20) + eps)
        MASK_R = is_positive(vdot(n20, n01) + eps)
        """
        e0 = p0.data - p.data
        e1 = p1.data - p.data
        e2 = p2.data - p.data
        n01 = vcross_(e0, e1, xp)
        n12 = vcross_(e1, e2, xp)
        n20 = vcross_(e2, e0, xp)

        MASK_P = is_positive_(vdot_(n01, n12, xp))
        MASK_Q = is_positive_(vdot_(n12, n20, xp))
        MASK_R = is_positive_(vdot_(n20, n01, xp))

        MASK_B = is_positive_(xp.abs(B.data))

        #MASK_TN = is_positive(tx)
        MASK_T0 = is_positive_(tx.data - t0.data)
        MASK_T1 = is_positive_(t1.data - tx.data)

        """
        save_boolean_img('./MASK_TN.png', MASK_TN)
        save_boolean_img('./MASK_T0.png', MASK_T0)
        save_boolean_img('./MASK_T1.png', MASK_T1)
        save_boolean_img('./MASK_P.png', MASK_P)
        save_boolean_img('./MASK_Q.png', MASK_Q)
        save_boolean_img('./MASK_R.png', MASK_R)
        save_boolean_img('./MASK_PQR.png', MASK_Q * MASK_R)
        """

        b = MASK_P & MASK_Q & MASK_R & MASK_B & MASK_T0 & MASK_T1
        #b = F.cast(b, np.bool)

        t = F.where(b, tx, t1)
        p = ro + t * rd

        bn = is_positive_(vdot_(rd.data, fn.data, xp))
        #bn = F.cast(bn, np.bool)
        n = F.where(bn, -fn, fn)
        #print('shape', face_id.shape)
        return {'b': b, 't': t, 'p': p, 'n': n, 'face_id': face_id}
