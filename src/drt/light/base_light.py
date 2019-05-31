import numpy as np
import chainer
import chainer.functions as F 

class BaseLight(chainer.Link):
    def __init__(self):
        pass

    def illuminate(self, info):
        pass