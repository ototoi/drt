import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable

class BaseCamera(chainer.Link):
    def __init__(self):
        super(BaseCamera, self).__init__()
    
    def shoot(self):
        pass 


