import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable


from .base_shape import BaseShape
from .mesh_accelerator import BruteforceMeshAccelerator
from ..vec import vdot, vnorm
from ..utils import make_parameter as MP


class TriangleLink(chainer.Link):
    def __init__(self, p0, p1, p2):
        super(TriangleLink, self).__init__()
        with self.init_scope():
            self.p0 = p0
            self.p1 = p1
            self.p2 = p2


class MeshLink(chainer.Link):
    def __init__(self, indices, positions, uvs=None, normals=None):
        super(MeshLink, self).__init__()
        self.indices = indices
        with self.init_scope():
            self.positions = MP(positions.reshape((-1, 3)))
            if uvs is not None:
                self.uvs = MP(uvs.reshape((-1, 2)))
            else:
                self.uvs = None
            if normals is not None:
                self.normals = MP(uvs.reshape((-1, 3)))
            else:
                self.normals = None

    def __len__(self):
        return self.get_triangle_length()

    def get_triangle_length(self):
        return len(self.indices) // 3

    def get_position_length(self):
        return self.positions.shape[0]

    def get_triangle(self, i):
        i0 = self.indices[3 * i + 0]
        i1 = self.indices[3 * i + 1]
        i2 = self.indices[3 * i + 2]
        p0 = self.positions[i0, :]
        p1 = self.positions[i1, :]
        p2 = self.positions[i2, :]
        t = TriangleLink(p0, p1, p2)
        return t


class MeshShape(BaseShape):
    """
    MeshShape: Shape for Mesh
    """

    def __init__(self, mesh, accelerator=None):
        super(MeshShape, self).__init__()
        with self.init_scope():
            self.mesh = mesh
        if accelerator is None:
            accelerator = BruteforceMeshAccelerator()
        self.accelerator = accelerator
        if self.accelerator is not None:
            for i in range(len(self.mesh)):
                t = self.mesh.get_triangle(i)
                self.accelerator.add_triangle(t)
            self.accelerator.construct()

    def construct(self):
        self.accelerator.construct()

    def intersect(self, ro, rd, t0, t1):
        return self.accelerator.intersect(ro, rd, t0, t1)

    def to_gpu(self):
        self.mesh.to_gpu()
