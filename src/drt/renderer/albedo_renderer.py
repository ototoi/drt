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
        _, H, W = b.shape[:3]

        bb = b
        bb = F.transpose(bb, (1, 2, 0))
        mask = F.where(bb, np.ones((H, W, 1), albedo.dtype), np.zeros((H, W, 1), albedo.dtype))
        albedo = F.transpose(albedo, (1, 2, 0))
        img = F.clip(albedo * mask, 0.0, 1.0)

        return img