import numpy as np
import chainer
import chainer.functions as F

from ..triangle_shape import TriangleShape

class BruteforceMeshAccelerator(object):
    def __init__(self):
        self.triangles = []

    def intersect(self, ro, rd, t0, t1):
        s = self.triangles[0]
        t = t1
        info = s.intersect(ro, rd, t0, t)
        
        b = info['b']
        t = info['t']
        for i in range(1, len(self.triangles)):
            s = self.triangles[i]
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


    def add_triangle(self, t):
        t = TriangleShape(t.p0, t.p1, t.p2)
        self.triangles.append(t)


    def construct(self):
        pass