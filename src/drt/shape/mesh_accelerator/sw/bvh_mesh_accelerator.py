
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

class BVH(object):
    def __init__(self, triangles=[], bvhs=[], xp=None):
        self.triangles = triangles
        self.bvhs = bvhs
        if len(triangles) > 0:
            xp = chainer.backend.get_array_module(triangles[0])
            p0 = [t.p0.data for t in triangles]
            p1 = [t.p1.data for t in triangles]
            p2 = [t.p2.data for t in triangles]
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
            triangles, key=lambda t: t.p0.data[plane]+t.p1.data[plane]+t.p2.data[plane])
        m = sz // 2
        bvh0 = construct_bvh_box(BVH(triangles[:m]))
        bvh1 = construct_bvh_box(BVH(triangles[m:]))
        return BVH(triangles=[], bvhs=[bvh0, bvh1], xp=xp)


def construct_bvh(triangles):
    xp = chainer.backend.get_array_module(triangles[0])
    tmp = BVH(triangles, xp=xp)
    return construct_bvh_box(tmp)


def intersect_box(bvh, ro, ird, t0, t1):
    B, _, H, W = ro.shape[:4]
    xp = chainer.backend.get_array_module(ro)
    min_ = bvh.min - 1e-3 
    max_ = bvh.max + 1e-3
    ro = xp.transpose(ro, (0, 2, 3, 1))  # B, H, W, 3
    ird = xp.transpose(ird, (0, 2, 3, 1))  # B, H, W, 3
    mask = ird > 0  # B, H, W, 3
    tt0 = (xp.broadcast_to(xp.where(mask, min_, max_), (B, H, W, 3)) - ro) * ird
    tt1 = (xp.broadcast_to(xp.where(mask, max_, min_), (B, H, W, 3)) - ro) * ird
    #print(tt0.shape)
    #print(tt1.shape)
    tt0 = xp.transpose(xp.max(tt0, axis=3).reshape((B, H, W, 1)), (0, 3, 1, 2))  # B, 1, H, W
    tt1 = xp.transpose(xp.min(tt1, axis=3).reshape((B, H, W, 1)), (0, 3, 1, 2))  # B, 1, H, W
    
    mask1 = tt0 < tt1
    mask2 = t0 <= tt0
    mask3 = tt1 <= t1
    mask = mask1 * mask2 * mask3
    tt0 = xp.where(mask, tt0, t0)
    tt1 = xp.where(mask, tt1, t1)
    pred = xp.any(mask)
    return pred, tt0, tt1


def query_bvh(bvh, ro, ird, t0, t1):
    b, tt0, tt1 = intersect_box(bvh, ro, ird, t0, t1)
    if b:
        if len(bvh.bvhs) > 0:
            triangles = []
            for cb in bvh.bvhs:
                triangles += query_bvh(cb, ro, ird, tt0, tt1)
            return triangles
        else:
            return bvh.triangles
    else:
        return []


def intersect_bvh(bs, ids, bvh, ro, rd, ird, t0, t1):
    b, _, _ = intersect_box(bvh, ro, ird, t0, t1)
    tt0 = t0
    tt1 = t1
    if b:
        if len(bvh.bvhs) > 0:
            for cb in bvh.bvhs:
                bs, ids, tt0, tt1 = intersect_bvh(bs, ids, cb, ro, rd, ird, tt0, tt1)
            return bs, ids, tt0, tt1
        else:
            for t in bvh.triangles:
                p0_ = t.p0.data
                p1_ = t.p1.data
                p2_ = t.p2.data
                id_ = t.id.data
                eps_ = t.eps.data
                bs, ids, tt0, tt1 = intersect_triangle(bs, ids, p0_, p1_, p2_, id_, eps_, ro, rd, tt0, tt1)
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


class SWBVHMeshAccelerator(object):
    def __init__(self):
        self.triangles = []
        self.root = None

    def add_triangle(self, t):
        self.triangles.append(t)

    def construct(self):
        self.root = construct_bvh(self.triangles)
        # triangles = get_triangles(self.root)
        # print(len(triangles))

    def query(self, ro, rd, t0, t1):
        xp = chainer.backend.get_array_module(ro)
        ro = ro.data
        rd = rd.data
        rd = xp.where(rd >= 0, xp.maximum(rd, +1e-6), xp.minimum(rd, -1e-6))
        ird = 1.0 / rd
        t0 = t0.data
        t1 = t1.data
        triangles = query_bvh(self.root, ro, ird, t0, t1)
        return triangles
    
    def intersect(self, ro, rd, t0, t1):
        B, _, H, W = ro.shape[:4]
        xp = chainer.backend.get_array_module(ro)
        ro_ = ro.data
        rd_ = rd.data
        ird_ = xp.where(rd_ >= 0, xp.maximum(rd_, +1e-6), xp.minimum(rd_, -1e-6))
        ird_ = 1.0 / ird_
        t0_ = t0.data
        t1_ = t1.data
        bs_  = xp.zeros((B, 1, H, W), xp.bool)
        ids_ = xp.zeros((B, 1, H, W), xp.int32) * -1
        bs_, ids_, _, _ = intersect_bvh(bs_, ids_, self.root, ro_, rd_, ird_, t0_, t1_)
        ids_ = ids_.reshape((-1))
        ids_ = ids_[ids_>=0]
        ids_ = xp.unique(ids_)
        ids_ = cuda.to_cpu(ids_)
        return xp.any(bs_), ids_.tolist()


class BVHMeshAccelerator(BaseMeshAccelerator):
    """
    SWMeshAccelerator: Software Mesh Accelerator
    """

    def __init__(self, block_size=8):
        self.triangles = []
        self.accelerator = None
        self.block_size = block_size

    def intersect_block(self, ro, rd, t0, t1):
        """
        triangles = self.accelerator.query(ro, rd, t0, t1)
        if len(triangles) > 0:
            #print(len(triangles))
            s = triangles[0]
            t = t1
            info = s.intersect(ro, rd, t0, t)

            b = info['b']
            t = info['t']
            for s in triangles[1:]:
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
        """
        B, _, H, W = ro.shape[:4]
        xp = chainer.backend.get_array_module(ro)
        bb, ids = self.accelerator.intersect(ro, rd, t0, t1)
        if bb and len(ids) > 0:
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
            b = chainer.as_variable(xp.zeros((B, 1, H, W), np.bool))
            t = t0
            p = chainer.as_variable(xp.zeros((B, 3, H, W), np.float32))
            n = chainer.as_variable(xp.zeros((B, 3, H, W), np.float32))
            return {'b': b, 't': t, 'p': p, 'n': n}

    def intersect(self, ro, rd, t0, t1):
        bsz = self.block_size
        B, _, H, W = ro.shape[:4]
        nH = int(math.ceil(H / bsz))
        nW = int(math.ceil(W / bsz))
        xp = chainer.backend.get_array_module(ro)
        b = chainer.as_variable(xp.zeros((B, 1, H, W), np.bool))
        t = t0
        p = chainer.as_variable(xp.zeros((B, 3, H, W), np.float32))
        n = chainer.as_variable(xp.zeros((B, 3, H, W), np.float32))
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

    def add_triangle(self, t):
        t = TriangleShape(t.p0, t.p1, t.p2, t.id)
        self.triangles.append(t)

    def construct(self):
        accelerator = SWBVHMeshAccelerator()
        for t in self.triangles:
            accelerator.add_triangle(t)
        accelerator.construct()
        self.accelerator = accelerator

