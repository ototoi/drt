import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable


class BaseRenderer(object):
    def __init__(self):
        pass
    
    def render(self, info):
        pass
