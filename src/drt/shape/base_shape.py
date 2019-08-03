import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable


class BaseShape(chainer.Link):
    """
    BaseShape:
    """
    def __init__(self):
        super(BaseShape, self).__init__()

    def intersect(self, ro, rd, t0, t1):
        pass

    def construct(self):
        pass
