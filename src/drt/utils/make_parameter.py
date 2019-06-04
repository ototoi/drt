import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable
from chainer import Parameter


def make_parameter(x):
    if isinstance(x, Parameter):
        return x
    elif isinstance(x, Variable):
        return x
    elif isinstance(x, (np.ndarray, np.generic) ):
        return Parameter(x)
    elif isinstance(x, list):
        x = np.array(x, dtype=np.float32)
        return Parameter(x)
    else:
        return Parameter(x)


def add_to_model(model, name, param):
    with model.init_scope():
        setattr(model, name, param)

