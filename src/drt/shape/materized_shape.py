import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable

from .base_shape import BaseShape
from ..vec import vdot, vnorm


from .base_shape import BaseShape
from ..vec import vdot, vnorm


class MaterizedShape(BaseShape):
    """
    MaterizedShape: 
    """

    def __init__(self, shape, material):
        super(MaterizedShape, self).__init__()
        with self.init_scope():
            self.shape = shape
            self.material = material

    def intersect(self, ro, rd, t0, t1):
        info = self.shape.intersect(ro, rd, t0, t1)
        return self.material.set_parameters(info)

    def to_gpu(self):
        self.shape.to_gpu()
        self.material.to_gpu()

    def construct(self):
        self.shape.construct()
