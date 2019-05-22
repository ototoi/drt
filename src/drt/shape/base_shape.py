import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable


class BaseShape(object):
    def __init__(self):
        pass

    def intersect(self, ro, rd, t0, t1):
        pass
