import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable

from .base_material import BaseMaterial
from ..utils import make_parameter as  MP

class DiffuseMaterial(BaseMaterial):
    def __init__(self, albedo=[1,1,1]):
        self.albedo = MP(albedo)
        self.zero = MP([0])


    def set_parameters(self, info):
        mask = info['b']
        B, _, H, W = mask.shape[:4]
        if 'albedo' in info:
            albedo_old = info['albedo']
            albedo_new = F.broadcast_to(self.albedo.reshape((1, 3, 1, 1)), (B, 3, H, W))
            info['albedo'] = F.where(mask, albedo_new, albedo_old)
        else:
            albedo_old = F.broadcast_to(self.zero.reshape((1, 1, 1, 1)), (B, 3, H, W))
            albedo_new = F.broadcast_to(self.albedo.reshape((1, 3, 1, 1)), (B, 3, H, W))
            info['albedo'] = F.where(mask, albedo_new, albedo_old)
        return info

