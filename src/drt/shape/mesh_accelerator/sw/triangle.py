import math
import copy

import numpy as np
import chainer


def is_positive_(a):
    return a > 0


def vdot_(a, b, xp):
    B, _, H, W = a.shape[:4] 
    m = a * b
    return xp.sum(m, axis=1).reshape((B, 1, H, W)) #C


def vnorm_(a, xp):
    l = 1.0 / xp.sqrt(vdot_(a, a, xp))
    return a * l


def vcross_(a, b, xp):
    c = xp.zeros((a.shape), a.dtype)
    c[:, 0, :, :] = a[:, 1, :, :]*b[:, 2, :, :] - a[:, 2, :, :]*b[:, 1, :, :]
    c[:, 1, :, :] = a[:, 2, :, :]*b[:, 0, :, :] - a[:, 0, :, :]*b[:, 2, :, :]
    c[:, 2, :, :] = a[:, 0, :, :]*b[:, 1, :, :] - a[:, 1, :, :]*b[:, 0, :, :]
    return c

def vdot_e_(a, b, xp):
    m = a * b
    return xp.sum(m)

def vcross_e_(a, b, xp):
    x = (a[1]*b[2] - a[2]*b[1]).reshape((1,))
    y = (a[2]*b[0] - a[0]*b[2]).reshape((1,))
    z = (a[0]*b[1] - a[1]*b[0]).reshape((1,))
    return xp.concatenate([x, y, z], axis=0)

def vnorm_e_(a, xp):
    return a / xp.sqrt(vdot_e_(a, a, xp))


def where_(mask, m0, m1):
    return m1 + (m0-m1) * mask


def intersect_triangle(bs, ids, p0, p1, p2, id, ro, rd, t0, t1):
    xp = chainer.backend.get_array_module(ro)
    BB, _, H, W = ro.shape[:4]

    sn = xp.broadcast_to(vnorm_e_(vcross_e_(p1 - p0, p2 - p0, xp), xp).reshape((1, 3, 1, 1)), (BB, 3, H, W))

    p0 = xp.broadcast_to(p0.reshape((1, 3, 1, 1)), (BB, 3, H, W))
    p1 = xp.broadcast_to(p1.reshape((1, 3, 1, 1)), (BB, 3, H, W))
    p2 = xp.broadcast_to(p2.reshape((1, 3, 1, 1)), (BB, 3, H, W))
    #eps = xp.broadcast_to(eps.reshape((1, 1, 1, 1)), (BB, 1, H, W))

    aa = p0 - ro

    A = vdot_(aa, sn, xp)
    B = vdot_(rd, sn, xp)                                #(1, 1, H, W)
    #B = F.sign(B) * F.maximum(F.absolute(B), eps)   #

    tx = A / B
    p = ro + tx * rd
    e0 = p0 - p
    e1 = p1 - p
    e2 = p2 - p
    n01 = vcross_(e0, e1, xp)
    n12 = vcross_(e1, e2, xp)
    n20 = vcross_(e2, e0, xp)

    MASK_P = is_positive_(vdot_(n01, n12, xp))
    MASK_Q = is_positive_(vdot_(n12, n20, xp))
    MASK_R = is_positive_(vdot_(n20, n01, xp))

    MASK_B = is_positive_(xp.abs(B))

    MASK_T0 = is_positive_(tx - t0)
    MASK_T1 = is_positive_(t1 - tx)

    b = MASK_P & MASK_Q & MASK_R & MASK_B & MASK_T0 & MASK_T1
    
    face_id = xp.broadcast_to(id, (BB, 1, H, W))

    bs |= b
    ids = where_(b.astype(xp.int32), face_id, ids)
    t1  = where_(b.astype(xp.float32), tx, t1)
    return bs, ids, t0, t1
