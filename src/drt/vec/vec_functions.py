import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable

def vdot(a, b):
    m = a * b
    return F.sum(m, axis=0)


def vnorm(a):
    l = F.sqrt(vdot(a, a))
    return a / l

