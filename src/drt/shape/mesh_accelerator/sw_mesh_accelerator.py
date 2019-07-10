
import math
import copy

import numpy as np
import chainer
import chainer.functions as F

from ..triangle_shape import TriangleShape
from ...utils.set_item import set_item


class BVH(object):
    def __init__(self, triangles=[], bvhs=[], xp=None):
        self.triangles = triangles
        self.bvhs = bvhs
        if len(triangles) > 0:
            xp = chainer.backend.get_array_module(triangles[0])
            p0 = [t.p0.data for t in triangles]
            p1 = [t.p1.data for t in triangles]
            p2 = [t.p1.data for t in triangles]
            points = xp.array([p0, p1, p2], dtype=np.float32)
            points = xp.transpose(points, (2, 0, 1))
            points = points.reshape((3, -1))

            min_ = xp.min(points, axis=1).reshape((3, )) - 1e-6
            max_ = xp.max(points, axis=1).reshape((3, )) + 1e-6
            self.box = xp.array([min_, max_])
        else:
            b0 = bvhs[0]
            b1 = bvhs[1]

            min_ = xp.minimum(b0.box[0, :], b1.box[0, :]).reshape((3, )) - 1e-6
            max_ = xp.maximum(b0.box[1, :], b1.box[1, :]).reshape((3, )) + 1e-6
            self.box = xp.array([min_, max_])


def construct_bvh_box(bvh):
    triangles = bvh.triangles
    sz = len(triangles)
    if sz <= 2:
        return bvh
    else:
        xp = chainer.backend.get_array_module(triangles[0])
        min_ = bvh.box[0, :].reshape((3, ))
        max_ = bvh.box[1, :].reshape((3, ))
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


"""
int phase = r.phase();
int sign[3] = {(phase >> 0) & 1, (phase >> 1) & 1, (phase >> 2) & 1};
vector3 box[2] = {min, max};
const vector3& org = r.origin();
const vector3& idir = r.inversed_direction();

for (int i = 0; i < 3; i++)
{
    tmin = std::max<real>(tmin, (box[sign[i]][i] - org[i]) * idir[i]);
    tmax = std::min<real>(tmax, (box[1 - sign[i]][i] - org[i]) * idir[i]);
}
tmin *= real(1) - epsilon_<real>::value();
tmax *= real(1) + epsilon_<real>::value();
return tmin <= tmax;

"""


def box_intersect(box, ro, ird, t0, t1):
    B, _, H, W = ro.shape[:4]
    xp = chainer.backend.get_array_module(ro)
    box = xp.transpose(box, (1, 0))  # 3, 2
    min_ = box[:,0]
    max_ = box[:,1]
    ro = xp.transpose(ro, (0, 2, 3, 1))  # B, H, W, 3
    ird = xp.transpose(ird, (0, 2, 3, 1))  # B, H, W, 3
    mask = ird > 0  # B, H, W, 3
    t0 = xp.broadcast_to(t0.reshape((B, H, W, 1)), (B, H, W, 3))
    t1 = xp.broadcast_to(t1.reshape((B, H, W, 1)), (B, H, W, 3))
    t0 = xp.maximum(t0, (xp.where(mask, min_, max_) - ro) * ird)  # B, H, W, 3
    t1 = xp.minimum(t1, (xp.where(mask, max_, min_) - ro) * ird)  # B, H, W, 3
    t0 = xp.max(t0, axis=3).reshape((B, H, W))  # B, H, W
    t1 = xp.min(t1, axis=3).reshape((B, H, W))  # B, H, W
    pred = np.any(t0 < t1)
    return pred


def query_bvh(bvh, ro, ird, t0, t1):
    if box_intersect(bvh.box, ro, ird, t0, t1):
        if len(bvh.bvhs) > 0:
            triangles = []
            for b in bvh.bvhs:
                triangles += query_bvh(b, ro, ird, t0, t1)
            return triangles
        else:
            return bvh.triangles
    else:
        return []


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


class SWMeshAccelerator(object):
    '''
    SWMeshAccelerator: Software Mesh Accelerator
    '''

    def __init__(self, block_size=16):
        self.triangles = []
        self.accelerator = None
        self.block_size = block_size

    def intersect_block(self, ro, rd, t0, t1):
        triangles = self.accelerator.query(ro, rd, t0, t1)
        if len(triangles) > 0:
            # print(len(triangles))
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
        else:
            B, _, H, W = ro.shape[:4]
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
        t = TriangleShape(t.p0, t.p1, t.p2)
        self.triangles.append(t)

    def construct(self):
        accelerator = SWBVHMeshAccelerator()
        for t in self.triangles:
            accelerator.add_triangle(t)
        accelerator.construct()
        self.accelerator = accelerator
