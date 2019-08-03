
import os
import sys
import argparse
import glob
import math
import time

import numpy as np
import cv2


import chainer
from chainer import cuda, training, reporter, function
from chainer.training import StandardUpdater, extensions
from chainer import serializers, optimizers, functions as F
from chainer.dataset import DatasetMixin


sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..')))

from drt.io.obj_mesh_loader import ObjMeshLoader
from drt.shape.mesh_shape import MeshLink, MeshShape
from drt.shape.mesh_accelerator.sw import BVHMeshAccelerator
from drt.light import PointLight
from drt.utils import make_parameter as MP
from drt.utils import add_parameter as AP
from drt.vec import vdot, vnorm
from drt.renderer import NormalRenderer, DiffuseRenderer, AlbedoRenderer
from drt.material import DiffuseMaterial
from drt.shape import SphereShape, PlaneShape, TriangleShape, RectangleShape, CompositeShape, MaterizedShape
from drt.camera import PerspectiveCamera
from drt.shape.cornellbox_shape import create_light, create_floor, create_shortblock, create_tallblock


from drt.utils.set_item import set_item

class RaytraceFunc(object):
    def __init__(self, shape, light, camera):
        renderer = DiffuseRenderer()
        self.camera = camera
        self.shape = shape
        self.renderer = renderer
        self.ll = [light]

    def __call__(self, B):
        ro, rd, t0, t1 = self.camera.shoot()
        C, H, W = ro.shape[:3]
        ro = F.broadcast_to(ro.reshape((1, C, H, W)), (B, C, H, W))
        rd = F.broadcast_to(rd.reshape((1, C, H, W)), (B, C, H, W))
        t0 = F.broadcast_to(t0.reshape((1, 1, H, W)), (B, 1, H, W))
        t1 = F.broadcast_to(t1.reshape((1, 1, H, W)), (B, 1, H, W))

        info = self.shape.intersect(ro, rd, t0, t1)
        info['ro'] = ro
        info['rd'] = rd

        #x = x[0, :].reshape((1, 3))
        
        info['ll'] = self.ll
        img = self.renderer.render(info)
        return img

    def to_gpu(self):
        self.camera.to_gpu()
        self.shape.to_gpu()
        for l in self.ll:
            l.to_gpu()


def get_obj_shape(path, scale=1.0):
    loader = ObjMeshLoader()
    meshes, _ = loader.load(path)
    mesh_data = meshes[0]
    indices   = np.array([idx.vertex_index for idx in mesh_data.indices], dtype=np.int32)
    positions = np.array(mesh_data.positions, dtype=np.float32).reshape((-1, 3))
    positions *= scale
    #positions -= np.array([0, -1, 0], positions.dtype)
    mesh_link = MeshLink(indices, positions)
    mesh_shape = MeshShape(mesh_link, accelerator=BVHMeshAccelerator())
    #mesh_shape = MeshShape(mesh_link, accelerator=None)
    return mesh_shape


def draw_goal_cornelbox(path, output, device=-1):
    materials = {}
    materials["light"] = DiffuseMaterial([1.0, 1.0, 1.0])
    materials["white"] = DiffuseMaterial([0.5, 0.5, 0.5])
    materials["green"] = DiffuseMaterial([0.0, 1.0, 0.0])
    materials["red"] = DiffuseMaterial([1.0, 0.0, 0.0])

    #shape_floor = create_floor(materials)
    #shape_shortblock = create_shortblock(materials)
    #shape_tallblock = create_tallblock(materials)
    mesh_shape = get_obj_shape(path, scale=20.0)
    #print(mesh_shape.mesh.positions.shape)
    #print(mesh_shape.mesh.indices)
    mesh_shape = MaterizedShape(mesh_shape, materials["green"])

    shape = mesh_shape #CompositeShape([mesh_shape])

    fov = math.atan2(0.025, 0.035) * 180.0 / math.pi
    camera = PerspectiveCamera(512, 512, fov, origin=[8.0, 8.0, 8.0], direction=[-1, -1, -1])
    light = PointLight(origin=[0, 100, 0], color=[1.0, 1.0, 1.0])

    func = RaytraceFunc(shape=shape, light=light, camera=camera)

    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        func.to_gpu()
        mesh_shape.construct()
    else:
        mesh_shape.construct()

    y_data = func(1)
    y_data = y_data.data
    
    if device >= 0:
        y_data = y_data.get()
        cuda.get_device_from_id(device).synchronize()

    img = y_data[0]
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output, img)
    return 0

def process(args):
    start = time.time()
    draw_goal_cornelbox(args.input, args.output, device=args.gpu)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description='DRT')
    parser.add_argument('--input', '-i', default='./data/load_and_render_obj/bunny.obj', help='input file path')
    parser.add_argument('--output', '-o', default='./data/load_and_render_obj/goal.png', help='input file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='gpu')
    args = parser.parse_args()
    return process(args)


if __name__ == '__main__':
    exit(main())

