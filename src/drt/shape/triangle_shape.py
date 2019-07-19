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


def save_boolean_img(path, mask):
    B, C, H, W = mask.shape[:4]
    mask = F.cast(mask, 'int').data
    mask = mask.reshape((H, W))
    mask = mask.astype(np.uint8) * 255
    cv2.imwrite(path, mask)



class TriangleShape(BaseShape):
    """
    TriangleShape:
    """

    def __init__(self, p0, p1, p2, id):
        super(TriangleShape, self).__init__()
        xp = chainer.backend.get_array_module(p0.data)
        with self.init_scope():
            self.p0 = MP(p0)
            self.p1 = MP(p1)
            self.p2 = MP(p2)
            self.id = MP(id)
            self.eps = MP(xp.array([1e-6], xp.float32))

    def intersect(self, ro, rd, t0, t1):
        BB, _, H, W = ro.shape[:4]
        p0 = F.broadcast_to(self.p0.reshape((1, 3, 1, 1)), (BB, 3, H, W))
        p1 = F.broadcast_to(self.p1.reshape((1, 3, 1, 1)), (BB, 3, H, W))
        p2 = F.broadcast_to(self.p2.reshape((1, 3, 1, 1)), (BB, 3, H, W))
        eps = F.broadcast_to(self.eps.reshape((1, 1, 1, 1)), (BB, 1, H, W))

        so = p0
        sn = vnorm(vcross(p1 - p0, p2 - p0))

        aa = so - ro

        A = vdot(aa, sn)
        B = vdot(rd, sn)                                #(1, 1, H, W)
        #B = F.sign(B) * F.maximum(F.absolute(B), eps)   #

        tx = A / B
        p = ro + tx * rd
        n01 = vnorm(vcross(p0 - p, p1 - p))
        n12 = vnorm(vcross(p1 - p, p2 - p))
        n20 = vnorm(vcross(p2 - p, p0 - p))

        MASK_P = is_positive(vdot(n01, n12) + eps)
        MASK_Q = is_positive(vdot(n12, n20) + eps)
        MASK_R = is_positive(vdot(n20, n01) + eps)

        MASK_B = is_positive(F.absolute(B))

        #MASK_TN = is_positive(tx)
        MASK_T0 = is_positive(tx - t0)
        MASK_T1 = is_positive(t1 - tx)

        """
        save_boolean_img('./MASK_TN.png', MASK_TN)
        save_boolean_img('./MASK_T0.png', MASK_T0)
        save_boolean_img('./MASK_T1.png', MASK_T1)
        save_boolean_img('./MASK_P.png', MASK_P)
        save_boolean_img('./MASK_Q.png', MASK_Q)
        save_boolean_img('./MASK_R.png', MASK_R)
        save_boolean_img('./MASK_PQR.png', MASK_Q * MASK_R)
        """

        b = F.cast(MASK_P * MASK_Q * MASK_R * MASK_B * MASK_T0 * MASK_T1, 'bool')
        t = F.where(b, tx, t1)
        p = ro + t * rd

        bn = F.cast(is_positive(vdot(rd, sn)), 'bool')
        n = F.where(bn, -sn, sn)
        face_id = F.broadcast_to(self.id, (BB, 1, H, W))
        #print('shape', face_id.shape)
        return {'b': b, 't': t, 'p': p, 'n': n, 'face_id': face_id}
