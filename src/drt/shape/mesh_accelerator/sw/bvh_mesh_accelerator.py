
import math
import copy

import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F

from ..base_mesh_accelerator import BaseMeshAccelerator
from ...triangle_shape import TriangleShape
from ....utils.set_item import set_item

from .triangle import intersect_triangle


class Triangle(object):
    def __init__(self, p0, p1, p2, id):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.id = id
    

class BVH(object):
    def __init__(self, triangles=[], bvhs=[], plane=0, xp=None):
        self.triangles = triangles
        self.bvhs = bvhs
        self.plane = plane
        if len(triangles) > 0:
            xp = chainer.backend.get_array_module(triangles[0].p0)
            p0 = [t.p0 for t in triangles]
            p1 = [t.p1 for t in triangles]
            p2 = [t.p2 for t in triangles]
            points = xp.array([p0, p1, p2], dtype=np.float32)
            points = xp.transpose(points, (2, 0, 1))
            points = points.reshape((3, -1))

            self.min = xp.min(points, axis=1).reshape((3, )) - 1e-3
            self.max = xp.max(points, axis=1).reshape((3, )) + 1e-3
        else:
            b0 = bvhs[0]
            b1 = bvhs[1]

            self.min = xp.minimum(b0.min, b1.min).reshape((3, )) - 1e-3
            self.max = xp.maximum(b0.max, b1.max).reshape((3, )) + 1e-3


def construct_bvh_box(bvh):
    triangles = bvh.triangles
    sz = len(triangles)
    if sz <= 2:
        return bvh
    else:
        xp = chainer.backend.get_array_module(triangles[0])
        min_ = bvh.min
        max_ = bvh.max
        wid_ = max_ - min_
        plane = xp.argmax(wid_, axis=0)
        triangles = sorted(
            triangles, key=lambda t: t.p0[plane]+t.p1[plane]+t.p2[plane])
        m = sz // 2
        bvh0 = construct_bvh_box(BVH(triangles[:m]))
        bvh1 = construct_bvh_box(BVH(triangles[m:]))
        return BVH(triangles=[], bvhs=[bvh0, bvh1], plane=plane, xp=xp)


def construct_bvh(triangles):
    xp = chainer.backend.get_array_module(triangles[0])
    tmp = BVH(triangles, xp=xp)
    return construct_bvh_box(tmp)

def where_(mask, m0, m1):
    return m1 + (m0-m1) * mask

"""
def get_minmax(mask, m0, m1, xp):
    B, H, W, _ = mask.shape[:4]
    t0 = where_(mask[:,:,:,0], m0[0], m1[0]).reshape((B, H, W, 1))
    t1 = where_(mask[:,:,:,1], m0[1], m1[1]).reshape((B, H, W, 1))
    t2 = where_(mask[:,:,:,2], m0[2], m1[2]).reshape((B, H, W, 1))
    return xp.concatenate([t0, t1, t2], axis=3)
"""

def get_minmax(mask, m0, m1):
    return where_(mask, m0, m1)


def intersect_box(bvh, ro, ird, t0, t1):
    B, _, H, W = ro.shape[:4]
    xp = chainer.backend.get_array_module(ro)
    min_ = bvh.min
    max_ = bvh.max
    ro = xp.transpose(ro, (0, 2, 3, 1))  # B, H, W, 3
    ird = xp.transpose(ird, (0, 2, 3, 1))  # B, H, W, 3
    mask = (ird > 0).astype(xp.float32)  # B, H, W, 3
    #tt0 = (xp.where(mask, min_, max_) - ro) * ird
    #tt1 = (xp.where(mask, max_, min_) - ro) * ird
    tt0 = (get_minmax(mask, min_, max_) - ro) * ird
    tt1 = (get_minmax(mask, max_, min_) - ro) * ird
    #print(tt0.shape)
    #print(tt1.shape)
    tt0 = xp.transpose(xp.max(tt0, axis=3).reshape((B, H, W, 1)), (0, 3, 1, 2))  # B, 1, H, W
    tt1 = xp.transpose(xp.min(tt1, axis=3).reshape((B, H, W, 1)), (0, 3, 1, 2))  # B, 1, H, W
    tt0 = xp.maximum(tt0, t0)
    tt1 = xp.minimum(tt1, t1)
    
    mask = tt0 < tt1
    pred = xp.any(mask)
    return pred


def intersect_bvh(bs, ids, bvh, table, ro, rd, ird, t0, t1):
    #xp = chainer.backend.get_array_module(ro)
    b = intersect_box(bvh, ro, ird, t0, t1)
    if b:
        tt0 = t0
        tt1 = t1
        if len(bvh.bvhs) > 0:
            plane = bvh.plane
            if table[plane]:   # rd+
                bs, ids, tt0, tt1 = intersect_bvh(bs, ids, bvh.bvhs[0], table, ro, rd, ird, tt0, tt1)
                bs, ids, tt0, tt1 = intersect_bvh(bs, ids, bvh.bvhs[1], table, ro, rd, ird, tt0, tt1)
            else:
                bs, ids, tt0, tt1 = intersect_bvh(bs, ids, bvh.bvhs[1], table, ro, rd, ird, tt0, tt1)
                bs, ids, tt0, tt1 = intersect_bvh(bs, ids, bvh.bvhs[0], table, ro, rd, ird, tt0, tt1)
            return bs, ids, tt0, tt1
        else:
            for t in bvh.triangles:
                p0_ = t.p0
                p1_ = t.p1
                p2_ = t.p2
                id_ = t.id
                bs, ids, tt0, tt1 = intersect_triangle(bs, ids, p0_, p1_, p2_, id_, ro, rd, tt0, tt1)
            return bs, ids, tt0, tt1
    else:
        return bs, ids, t0, t1


def get_triangles(bvh):
    if len(bvh.bvhs) > 0:
        triangles = []
        for b in bvh.bvhs:
            triangles += get_triangles(b)
        return triangles
    else:
        return bvh.triangles


def get_direction_table(rd):
    xp = chainer.backend.get_array_module(rd)
    rd = xp.transpose(rd, (0, 2, 3, 1))
    rd = rd.reshape((-1, 3))
    counts = xp.count_nonzero(rd >= 0, axis=0)
    return counts >= (rd.shape[0] // 2)


class SWBVHMeshAccelerator(object):
    def __init__(self):
        self.triangles = []
        self.root = None

    def add_triangle(self, t):
        p0 = t.p0.data
        p1 = t.p1.data
        p2 = t.p2.data
        id = t.id.data

        p0 = cuda.to_cpu(p0)
        p1 = cuda.to_cpu(p1)
        p2 = cuda.to_cpu(p2)
        id = cuda.to_cpu(id)

        t = Triangle(p0, p1, p2, id)
        self.triangles.append(t)

    def construct(self):
        self.root = construct_bvh(self.triangles)
        # triangles = get_triangles(self.root)
        # print(len(triangles))
    
    
    def intersect(self, ro, rd, t0, t1):
        B, _, H, W = ro.shape[:4]
        xp = chainer.backend.get_array_module(ro)
        ro_ = ro.data
        rd_ = rd.data
        ird_ = xp.where(rd_ >= 0, xp.maximum(rd_, +1e-6), xp.minimum(rd_, -1e-6))
        ird_ = 1.0 / ird_
        t0_ = t0.data
        t1_ = t1.data
        table = get_direction_table(rd_)

        bs_  = np.zeros((B, 1, H, W), xp.bool)
        ids_ = np.zeros((B, 1, H, W), xp.int32) * -1
        table = cuda.to_cpu(table)
        ro_ = cuda.to_cpu(ro_)
        rd_ = cuda.to_cpu(rd_)
        ird_ = cuda.to_cpu(ird_)
        t0_ = cuda.to_cpu(t0_)
        t1_ = cuda.to_cpu(t1_)

        #print(table)
        bs_, ids_, _, _ = intersect_bvh(bs_, ids_, self.root, table, ro_, rd_, ird_, t0_, t1_)
        ids_ = ids_.reshape((-1))
        ids_ = ids_[ids_>=0]
        ids_ = np.unique(ids_)
        return np.any(bs_), ids_.tolist()


class BVHMeshAccelerator(BaseMeshAccelerator):
    """
    SWMeshAccelerator: Software Mesh Accelerator
    """

    def __init__(self, block_size=16):
        self.triangles = []
        self.accelerator = None
        self.block_size = block_size

    def intersect_block(self, ro, rd, t0, t1):
        B, _, H, W = ro.shape[:4]
        xp = chainer.backend.get_array_module(ro)
        bac, ids = self.accelerator.intersect(ro, rd, t0, t1)
        if bac and len(ids) > 0:
            #print(len(ids))
            s = self.triangles[ids[0]]
            t = t1
            info = s.intersect(ro, rd, t0, t)

            b = info['b']
            t = info['t']
            for i in ids[1:]:
                s = self.triangles[i]
                iinfo = s.intersect(ro, rd, t0, t)
                bb = iinfo['b']
                tt = iinfo['t']
                b = b + bb
                t = tt
                for k in iinfo.keys():
                    if k in info:
                        info[k] = F.where(bb, iinfo[k], info[k])
                    else:
                        info[k] = iinfo[k]
            info['b'] = b
            info['t'] = t

            return info 
        else:
            b = chainer.as_variable(xp.zeros((B, 1, H, W), xp.bool))
            t = t0
            p = chainer.as_variable(xp.zeros((B, 3, H, W), xp.float32))
            n = chainer.as_variable(xp.zeros((B, 3, H, W), xp.float32))
            return {'b': b, 't': t, 'p': p, 'n': n}

    def intersect(self, ro, rd, t0, t1):
        bsz = self.block_size
        B, _, H, W = ro.shape[:4]
        nH = int(math.ceil(H / bsz))
        nW = int(math.ceil(W / bsz))
        xp = chainer.backend.get_array_module(ro)
        b = chainer.as_variable(xp.zeros((B, 1, H, W), xp.bool))
        t = t0
        p = chainer.as_variable(xp.zeros((B, 3, H, W), xp.float32))
        n = chainer.as_variable(xp.zeros((B, 3, H, W), xp.float32))
        info = {'b': b, 't': t, 'p': p, 'n': n}
        
        y: int
        for y in range(nH):
            y0 = y * bsz
            y1 = min(y0+bsz, H)
            x: int
            for x in range(nW):
                x0 = x*bsz
                x1 = min(x0+bsz, W)
                cro = ro[:, :, y0:y1, x0:x1]
                crd = rd[:, :, y0:y1, x0:x1]
                ct0 = t0[:, :, y0:y1, x0:x1]
                ct1 = t1[:, :, y0:y1, x0:x1]
                cinfo = self.intersect_block(cro, crd, ct0, ct1)
                for k in cinfo.keys():
                    ## info[:, :, y0:y1, x0:x1] = cinfo
                    i_ = cinfo[k]
                    slices = [slice(B), slice(i_.shape[1]), slice(y0, y1), slice(x0, x1)]
                    if k in info:
                        o_ = info[k]
                    else:
                        o_ = chainer.as_variable(xp.zeros((B, i_.shape[1], H, W), i_.dtype))
                    o_ = set_item(o_, slices, i_, inplace=False)
                    info[k] = o_

        return info

    def clear(self):
        self.triangles = []

    def add_triangle(self, t):
        t = TriangleShape(t.p0, t.p1, t.p2, t.id)
        self.triangles.append(t)

    def construct(self):
        accelerator = SWBVHMeshAccelerator()
        for t in self.triangles:
            accelerator.add_triangle(t)
        accelerator.construct()
        self.accelerator = accelerator

