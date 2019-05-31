import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable


class BaseMaterial(chainer.Link):
    def __init__(self):
        super(BaseMaterial, self).__init__()

    def set_parameters(self, info):
        pass


