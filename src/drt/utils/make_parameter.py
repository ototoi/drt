import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable


def make_parameter(x):
    if isinstance(x, Variable):
        return x
    elif isinstance(x, (np.ndarray, np.generic) ):
        return Variable(x)
    elif isinstance(x, list):
        x = np.array(x, dtype=np.float32)
        return Variable(x)
    else:
        return Variable(x)

