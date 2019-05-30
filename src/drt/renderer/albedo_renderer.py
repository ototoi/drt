import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable

from .base_renderer import BaseRenderer
from ..utils import make_parameter as  MP
from ..vec import vdot, vnorm


class AlbedoRenderer(BaseRenderer):
    def __init__(self):
        self.albedo = MP([1,1,1])
        self.zero_ = MP([0])

    def render(self, info: dict):
        b = info['b']
        albedo = info['albedo']
        B, _, H, W = b.shape[:4]

        b = F.transpose(b, (0, 2, 3, 1))
        albedo = F.transpose(albedo, (0, 2, 3, 1))
        mask = F.where(b, np.ones((B, H, W, 1), albedo.dtype), np.zeros((B, H, W, 1), albedo.dtype))
        img = F.clip(albedo * mask, 0.0, 1.0)
        
        return img