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
    l = xp.rsqrt(xp.mamimum(vdot_(a, a, xp), 1e-6))
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
    x = (a[:,1]*b[:,2] - a[:,2]*b[:,1]).reshape((-1, 1))
    y = (a[:,2]*b[:,0] - a[:,0]*b[:,2]).reshape((-1, 1))
    z = (a[:,0]*b[:,1] - a[:,1]*b[:,0]).reshape((-1, 1))
    return F.concat([x, y, z], axis=1)


def vnorm_e(a):
    return a * F.rsqrt(vdot_e(a, a))


def where_(mask, m0, m1):
    return m1 + (m0-m1) * mask


def batched_triangle_intersect_(p0, p1, p2, eps, fn, id, ro, rd, t0, t1):
    xp = chainer.backend.get_array_module(ro)
    BB = p0.shape[0]
    EB = p0.shape[0]
    _, _, H, W = ro.shape[:4]
    
    p0 = F.broadcast_to(p0.reshape((BB, 3, 1, 1)), (BB, 3, H, W))
    p1 = F.broadcast_to(p1.reshape((BB, 3, 1, 1)), (BB, 3, H, W))
    p2 = F.broadcast_to(p2.reshape((BB, 3, 1, 1)), (BB, 3, H, W))
    fn = F.broadcast_to(fn.reshape((BB, 3, 1, 1)), (BB, 3, H, W))
    id = F.broadcast_to(id.reshape((BB, 1, 1, 1)), (BB, 1, H, W))
    eps = F.broadcast_to(eps.reshape((EB, 1, 1, 1)), (BB, 1, H, W))
    ro = F.broadcast_to(ro.reshape((1, 3, H, W)), (BB, 3, H, W)) 
    rd = F.broadcast_to(rd.reshape((1, 3, H, W)), (BB, 3, H, W))
    t0 = F.broadcast_to(t0.reshape((1, 1, H, W)), (BB, 1, H, W))
    t1 = F.broadcast_to(t1.reshape((1, 1, H, W)), (BB, 1, H, W))

    aa = p0 - ro

    A = vdot(aa, fn)
    B = vdot(rd, fn)
    B = F.where(xp.abs(B.data) < eps.data , eps, B)

    #tx = F.where((xp.abs(A.data) < 1e-6)&(xp.abs(B.data) < 1e-6), t1, A / B)

    tx = F.maximum(t0, F.minimum(A / B, t1))
    p = ro + tx * rd

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

    b = MASK_P & MASK_Q & MASK_R & MASK_B & MASK_T0 & MASK_T1

    t = F.where(b, tx, t1)
    p = ro + t * rd

    n = - xp.sign(vdot_(rd.data, fn.data, xp)) * fn

    return b, t, p, n, id


def batched_triangle_reduce_(b, t, p, n, id):
    xp = chainer.backend.get_array_module(b)
    BB, _, H, W = b.shape[:4]
    kb = xp.sum(b, axis=0).astype(xp.bool)
    kt = F.min(t, axis=0)
    kp = p[0,:,:,:]
    kn = n[0,:,:,:]
    kid = id[0,:,:,:]
    
    for i in range(1, BB):
        bb = (kt.data >= t[i,:,:,:].data)
        kp = F.where(bb, p[i,:,:,:], kp)
        kn = F.where(bb, n[i,:,:,:], kn)
        kid = F.where(bb, id[i,:,:,:], kid)
        
    b = chainer.as_variable(kb.reshape(1, 1, H, W))
    t = kt.reshape(1, 1, H, W)
    p = kp.reshape(1, 3, H, W)
    n = kn.reshape(1, 3, H, W)
    id = kid.reshape(1, 1, H, W)
    
    return b, t, p, n, id

def batched_triangle_intersect(p0, p1, p2, eps, fn, id, ro, rd, t0, t1):
    b, t, p, n, id = batched_triangle_intersect_(p0, p1, p2, eps, fn, id, ro, rd, t0, t1)
    b, t, p, n, id = batched_triangle_reduce_(b, t, p, n, id)
    return {'b': b, 't': t, 'p': p, 'n': n, 'face_id': id}


class BatchedTriangleShape(BaseShape):
    """
    BatchedTriangleShape:
    """

    def __init__(self, p0, p1, p2, fn, id):
        super(BatchedTriangleShape, self).__init__()
        xp = chainer.backend.get_array_module(p0)
        p0 = MP(p0)
        p1 = MP(p1)
        p2 = MP(p2)
        #fn = vnorm_e(vcross_e(p1 - p0, p2 - p0))
        with self.init_scope():
            self.p0 = p0
            self.p1 = p1
            self.p2 = p2
            self.fn = fn
            self.id = MP(id)
            self.eps = MP(xp.array([1e-6],xp.float32))

    def intersect(self, ro, rd, t0, t1):
        return batched_triangle_intersect(self.p0, self.p1, self.p2, self.eps, self.fn, self.id, ro, rd, t0, t1)




