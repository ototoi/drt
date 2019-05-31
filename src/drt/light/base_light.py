import numpy as np
import chainer
import chainer.functions as F 

class BaseLight(chainer.Link):
    def __init__(self):
        super(BaseLight, self).__init__()

    def illuminate(self, info):
        pass