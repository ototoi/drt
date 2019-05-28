import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable


class BaseMaterial(object):
    def __init__(self):
        pass

    def set_parameters(self, info):
        pass


