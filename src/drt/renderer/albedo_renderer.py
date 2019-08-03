import numpy as np
import chainer
import chainer.functions as F
import chainer.backend
from chainer import Variable

from .base_renderer import BaseRenderer
from ..utils import make_parameter as  MP
from ..vec import vdot, vnorm


def replace_albedo(face_id):
    albedos = np.array([[1,0,0], [0,1,0], [0,0,1]], np.float32)
    B, _, H, W = face_id.shape[:4]
    face_id = face_id.reshape((B, H, W))
    a = chainer.as_variable(albedos[face_id.data%3])
    return np.transpose(a, (0, 3, 1, 2))


class AlbedoRenderer(BaseRenderer):
    def __init__(self):
        pass

    def render(self, info: dict):
        b = info['b']
        albedo = info['albedo']
        #if 'face_id' in info:
        #    albedo = replace_albedo(info['face_id'])
        #    #print(albedo.shape)
        xp = chainer.backend.get_array_module(albedo)
        B, _, H, W = b.shape[:4]
        b = F.transpose(b, (0, 2, 3, 1))
        albedo = F.transpose(albedo, (0, 2, 3, 1))
        #mask = F.where(b, xp.ones((B, H, W, 1), albedo.dtype), xp.zeros((B, H, W, 1), albedo.dtype))
        img = albedo
        img = F.transpose(img, (0, 3, 1, 2))

        return img