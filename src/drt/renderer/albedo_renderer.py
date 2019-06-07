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
        pass

    def render(self, info: dict):
        b = info['b']
        albedo = info['albedo']

        xp = chainer.backend.get_array_module(albedo)

        B, _, H, W = b.shape[:4]
        
        b = F.transpose(b, (0, 2, 3, 1))
        albedo = F.transpose(albedo, (0, 2, 3, 1))
        #mask = F.where(b, xp.ones((B, H, W, 1), albedo.dtype), xp.zeros((B, H, W, 1), albedo.dtype))
        img = albedo
        img = F.transpose(img, (0, 3, 1, 2))

        return img