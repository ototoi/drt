import chainer 
from chainer import Variable

import numpy as np


class ArrayLink(chainer.Link):
    def __init__(self, data, data1):
        super(ArrayLink, self).__init__()
        with self.init_scope():
            self.data = chainer.Parameter(data)

    def __call__(self):
        return self.data
