
import math
import copy

import numpy as np
import chainer
import chainer.functions as F

from ..triangle_shape import TriangleShape
import numpy as np
import chainer
import chainer.backend
import chainer.functions as F

from ..triangle_shape import TriangleShape


class BVH(object):
    def __init__(self, triangles=[], bvhs=[]):
        self.triangles = triangles
        self.bvhs = bvhs
        if len(triangles) > 0:
            xp = chainer.backend.get_array_module(triangles[0])
            p0s = [t.p0.data for t in triangles]
            p1s = [t.p1.data for t in triangles]
            p2s = [t.p1.data for t in triangles]
            points = xp.array([p0, p1, p2], dtype=np.float32)
            points = xp.transpose(points, (2, 0, 1))
            points = points.reshape((3, -1))

            self.min = xp.min(points, axis=1)
            self.max = xp.max(points, axis=1)
        else:
            b0 = bvhs[0]
            b1 = bvhs[1]
            self.min = xp.minimum(b0.min, b1.min)
            self.max = xp.maximum(b0.max, b1.max)


def construct_bvh_box(box):
    sz = len(box.triangles)
    if sz <= 4:
        return box
    else:
        xp = chainer.backend.get_array_module(triangles[0])
        min_ = box.min
        max_ = box.max
        wid_ = max_ - min_
        plane = xp.argmax(wid_, axis=0)
        triangles = box.triangles
        triangles = sorted(triangles, key=lambda t: t.p0[plane]+t.p1[plane]+t.p2[plane])
        m = sz // 2
        bvh0 = construct_bvh_box(BVH(triangles[:m, :]))
        bvh1 = construct_bvh_box(BVH(triangles[m:, :]))
        return BVH(triangles=[], bvhs=[bvh0, bvh1])


def construct_bvh(triangles):
    tmp = BVH(triangles)
    return construct_bvh_(tmp)


def box_intersect(box, ro, rd, t0, t1):
    min_ = box.min
    max_ = box.max
    
    return False


def query_bvh(bvh, ro, rd, t0, t1):
    if len(bvh.bvhs) > 0:
        triangles = []
        for b in bvh.bvhs:
            if box_intersect(b, ro, rd, t0, t1):
                triangles += query_bvh(b, ro, rd, t0, t1)
    else:
        return bvh.triangles


class BVHMeshAccelerator(object):
    def __init__(self):
        self.triangles = []
        self.root = None


    def add_triangle(self, t):
        self.triangles.append(t)


    def construct(self):
        self.root = construct_bvh(self.triangles)

    
    def query(self, ro, rd, t0, t1)
        triangles = query_bvh(self.root, ro.data, rd.data, t0.data, t1.data)
        return triangles



class SoftMeshAccelerator(object):
    def __init__(self, block_size=16):
        self.triangles = []
        self.accelerator = None
        self.block_size = block_size

    
    def intersect_block(self, ro, rd, t0, t1):
        triangles = self.accelerator.query(ro, rd, t0, td)
        if len(triangles) > 0:
            s = triangles[0]
            t = t1
            info = s.intersect(ro, rd, t0, t)
            
            b = info['b']
            t = info['t']
            for s in self.shapes[1:]:
                iinfo = s.intersect(ro, rd, t0, t)
                bb = iinfo['b']
                tt = iinfo['t']
                b = b + bb
                t = tt
                for k in iinfo.keys():
                    info[k] = F.where(bb, iinfo[k], info[k])
            info['b'] = b
            info['t'] = t

            return info
        else:
            B, _, H, W
            xp = chainer.backend.get_array_module(ro)
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
        for y in range(nH):
            y0 = y*bsz
            y1 = min(y0+bsz, H)
            for x in range(nW):
                x0 = x*bsz
                x1 = min(x0+bsz, W)
                cro = ro[:, :, y0:y1, x0:x1]
                crd = rd[:, :, y0:y1, x0:x1]
                ct0 = t0[:, :, y0:y1, x0:x1]
                ct1 = t1[:, :, y0:y1, x0:x1]
                cinfo = self.intersect_block(cro, crd, ct0, ct1)
                for k in cinfo.keys():
                    info[k][:, :, y0:y1, x0:x1] += cinfo[k]

        return info


    def add_triangle(self, t):
        t = TriangleShape(t.p0, t.p1, t.p2)
        self.triangles.append(t)


    def construct(self):
        accelerator = BVHMeshAccelerator()
        for t in self.triangles:
            accelerator.add_triangle(t)
        accelerator.construct()
        self.accelerator = accelerator
    