import math
import copy

import numpy as np
import chainer


def is_positive(a):
    return a > 0


def vdot(a, b):
    xp = chainer.backend.get_array_module(a)
    B, _, H, W = a.shape[:4] 
    m = a * b
    return xp.sum(m, axis=1).reshape((B, 1, H, W)) #C


def vnorm(a):
    xp = chainer.backend.get_array_module(a)
    l = xp.sqrt(vdot(a, a))
    return a / xp.maximum(l, 1e-6)


def vcross(a, b):
    xp = chainer.backend.get_array_module(a)
    B, _, H, W = a.shape[:4]
    x = a[:, 1, :, :]*b[:, 2, :, :] - a[:, 2, :, :]*b[:, 1, :, :]
    y = a[:, 2, :, :]*b[:, 0, :, :] - a[:, 0, :, :]*b[:, 2, :, :]
    z = a[:, 0, :, :]*b[:, 1, :, :] - a[:, 1, :, :]*b[:, 0, :, :]
    return xp.concatenate([x, y, z], axis=1).reshape((B, 3, H, W))


def intersect_triangle(bs, ids, p0, p1, p2, id, eps, ro, rd, t0, t1):
    xp = chainer.backend.get_array_module(ro)
    BB, _, H, W = ro.shape[:4]
    p0 = xp.broadcast_to(p0.reshape((1, 3, 1, 1)), (BB, 3, H, W))
    p1 = xp.broadcast_to(p1.reshape((1, 3, 1, 1)), (BB, 3, H, W))
    p2 = xp.broadcast_to(p2.reshape((1, 3, 1, 1)), (BB, 3, H, W))
    eps = xp.broadcast_to(eps.reshape((1, 1, 1, 1)), (BB, 1, H, W))

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

    MASK_B = is_positive(xp.abs(B))

    MASK_T0 = is_positive(tx - t0)
    MASK_T1 = is_positive(t1 - tx)

    b = MASK_P * MASK_Q * MASK_R * MASK_B * MASK_T0 * MASK_T1
    
    face_id = xp.broadcast_to(id, (BB, 1, H, W))

    bs += b
    ids = xp.where(b, face_id, ids)
    t1  = xp.where(b, tx, t1)
    return bs, ids, t0, t1
