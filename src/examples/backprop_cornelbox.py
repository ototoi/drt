import os
import sys
import argparse
import math
import random

import cv2
import numpy as np
import chainer
from chainer import Variable

import chainer
from chainer import cuda, training, reporter, function
from chainer.training import StandardUpdater, extensions
from chainer import serializers, optimizers, functions as F
from chainer.dataset import DatasetMixin

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..')))

from drt.light import PointLight
from drt.utils import make_parameter as MP
from drt.vec import vdot, vnorm
from drt.renderer import NormalRenderer, DiffuseRenderer, AlbedoRenderer
from drt.material import DiffuseMaterial
from drt.shape import SphereShape, PlaneShape, TriangleShape, RectangleShape, CompositeShape, MaterizedShape
from drt.camera import PerspectiveCamera
from drt.net import ArrayLink


def create_light(materials):
    mat_l = materials["light"]
    light = MaterizedShape(RectangleShape(
        [343.0, 548.8, 227.0],
        [343.0, 548.8, 332.0],
        [213.0, 548.8, 332.0],
        [213.0, 548.8, 227.0]), mat_l)

    return light

def create_floor(materials):
    mat_w = materials["white"]
    mat_r = materials["red"]
    mat_g = materials["green"]
    plane_top = MaterizedShape(RectangleShape(
        [556.0, 548.8, 0.0],
        [556.0, 548.8, 559.2],
        [0.0,   548.8, 559.2],
        [0.0,   548.8, 0.0]), mat_w)
    plane_bottom = MaterizedShape(RectangleShape(
        [556.0, 0.0,   0.0],
        [0.0,   0.0,   0.0],
        [0.0,   0.0, 559.2],
        [556.0, 0.0, 559.2]), mat_w)
    plane_back = MaterizedShape(RectangleShape(
        [556.0,   0.0, 559.2],
        [0.0,     0.0, 559.2],
        [0.0,   548.8, 559.2],
        [556.0, 548.8, 559.2]), mat_w)
    plane_left = MaterizedShape(RectangleShape(
        [556.0,   0.0,   0.0],
        [556.0,   0.0, 559.2],
        [556.0, 548.8, 559.2],
        [556.0, 548.8,   0.0]), mat_r)
    plane_right = MaterizedShape(RectangleShape(
        [0.0,     0.0, 559.2],
        [0.0,     0.0,   0.0],
        [0.0,   548.8,   0.0],
        [0.0,   548.8, 559.2]), mat_g)

    cmps = CompositeShape([plane_top, plane_bottom, plane_back, plane_left, plane_right])
    return cmps


def create_shortblock(materials):
    mat_w = materials["white"]
    a = RectangleShape([130.0, 165.0,  65.0], [ 82.0, 165.0, 225.0], [240.0, 165.0, 272.0], [290.0, 165.0, 114.0])
    b = RectangleShape([290.0, 0.0, 114.0], [290.0, 165.0, 114.0], [240.0, 165.0, 272.0], [240.0, 0.0, 272.0])
    c = RectangleShape([130.0, 0.0, 65.0], [130.0, 165.0, 65.0], [290.0, 165.0, 114.0], [290.0, 0.0, 114.0])
    d = RectangleShape([82.0, 0.0, 225.0], [82.0, 165.0, 225.0], [130.0, 165.0, 65.0], [130.0, 0.0, 65.0])
    e = RectangleShape([240.0, 0.0, 272.0], [240.0, 165.0, 272.0], [82.0, 165.0, 225.0], [82.0, 0.0, 225.0])
    cmps = MaterizedShape(CompositeShape([a, b, c, d, e]), mat_w)
    return cmps

def create_tallblock(materials):
    mat_w = materials["white"]
    a = RectangleShape([423.0, 330.0, 247.0], [265.0, 330.0, 296.0], [314.0, 330.0, 456.0], [472.0, 330.0, 406.0])
    b = RectangleShape([423.0, 0.0, 247.0], [423.0, 330.0, 247.0], [472.0, 330.0, 406.0], [472.0, 0.0, 406.0])
    c = RectangleShape([472.0, 0.0, 406.0], [472.0, 330.0, 406.0], [314.0, 330.0, 456.0], [314.0, 0.0, 456.0])
    d = RectangleShape([314.0, 0.0, 456.0], [314.0, 330.0, 456.0], [265.0, 330.0, 296.0], [265.0, 0.0, 296.0])
    e = RectangleShape([265.0, 0.0, 296.0], [265.0, 330.0, 296.0], [423.0, 330.0, 247.0], [423.0, 0.0, 247.0])
    cmps = MaterizedShape(CompositeShape([a, b, c, d, e]), mat_w)
    return cmps

"""
def draw_cornelbox_0(output):
    materials = {}
    materials["light"] = DiffuseMaterial([1.0, 1.0, 1.0])
    materials["white"] = DiffuseMaterial([0.5, 0.5, 0.5])
    materials["green"] = DiffuseMaterial([0.0, 1.0, 0.0])
    materials["red"] = DiffuseMaterial([1.0, 0.0, 0.0])
    shape_light = create_light(materials)
    shape_floor = create_floor(materials)
    shape_shortblock = create_shortblock(materials)
    shape_tallblock = create_tallblock(materials)

    cmps = CompositeShape([shape_light, shape_floor, shape_shortblock, shape_tallblock])
    
    fov = math.atan2(0.025, 0.035) * 180.0 / math.pi
    #fov = math.atan2(0.035, 0.025) * 180.0 / math.pi
    cam = PerspectiveCamera(512, 512, fov, [278.0, 273.0, -800.0])
    ro, rd = cam.shoot()
    C, H, W = ro.shape[:3]
    t0 = MP(np.broadcast_to(
        np.array([0.01], np.float32).reshape((1, 1, 1)), (1, H, W)))
    t1 = MP(np.broadcast_to(
        np.array([10000], np.float32).reshape((1, 1, 1)), (1, H, W)))
    #print("t0", t0.shape)
    #print("t1", t1.shape)
    info = cmps.intersect(ro, rd, t0, t1)
    info['ro'] = ro
    info['rd'] = rd
    l = PointLight(origin=[300, 548, 300], color=[0.1, 0.1, 0.1])
    info['ll'] = [l]
    #print("b", b.shape)

    renderer = DiffuseRenderer()
    #renderer = AlbedoRenderer()
    img = renderer.render(info)

    img = img.data
    img = (img * 255).astype(np.uint8)
    if output is not None:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output, img)
    return 0
"""

class RaytraceFunc(object):
    def __init__(self):
        materials = {}
        materials["light"] = DiffuseMaterial([1.0, 1.0, 1.0])
        materials["white"] = DiffuseMaterial([0.5, 0.5, 0.5])
        materials["green"] = DiffuseMaterial([0.0, 1.0, 0.0])
        materials["red"] = DiffuseMaterial([1.0, 0.0, 0.0])
        shape_light = create_light(materials)
        shape_floor = create_floor(materials)
        shape_shortblock = create_shortblock(materials)
        shape_tallblock = create_tallblock(materials)
        shape = CompositeShape([shape_light, shape_floor, shape_shortblock, shape_tallblock])

        fov = math.atan2(0.025, 0.035) * 180.0 / math.pi
        cam = PerspectiveCamera(512, 512, fov, [278.0, 273.0, -800.0])

        renderer = DiffuseRenderer()

        self.cam = cam
        self.shape = shape
        self.renderer = renderer

    def __call__(self, x):
        B = x.shape[0]
        ro, rd = self.cam.shoot()
        C, H, W = ro.shape[:3]
        ro = F.broadcast_to(ro.reshape((1, C, H, W)), (B, C, H, W))
        rd = F.broadcast_to(rd.reshape((1, C, H, W)), (B, C, H, W))
        t0 = MP(np.broadcast_to(
            np.array([0.01], np.float32).reshape((1, 1, 1, 1)), (B, 1, H, W)))
        t1 = MP(np.broadcast_to(
            np.array([10000], np.float32).reshape((1, 1, 1, 1)), (B, 1, H, W)))

        info = self.shape.intersect(ro, rd, t0, t1)
        info['ro'] = ro
        info['rd'] = rd

        #x = x[0, :].reshape((1, 3))
        l = PointLight(origin=x, color=[0.1, 0.1, 0.1])
        info['ll'] = [l]
        img = self.renderer.render(info)
        return img




def compute_loss(data1, data2):
    return F.mean_squared_error(data1, data2)

class RaytraceUpdater(StandardUpdater):
    def __init__(self, iterator, model, func, optimizer, device=None):
        super(RaytraceUpdater, self).__init__(iterator, optimizer, device=device)
        self.model = model
        self.func = func

    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer  = self.get_optimizer('main')

        batch = train_iter.next()
        t_data = self.converter(batch, self.device)
        x_data = self.model.data.reshape((1, 3))
        y_data = self.func(x_data)
        B, C, H, W = t_data.shape[:4]
        y_data = F.broadcast_to(y_data, (B, C, H, W))
        loss = compute_loss(y_data, t_data)

        reporter.report({
            'main/loss': loss,
            'pos/light_x': self.model.data[0],
            'pos/light_y': self.model.data[1],
            'pos/light_z': self.model.data[2]
        })

        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()
        #print(self.model.data.grad)

class RaytraceDataset(DatasetMixin):
    def __init__(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        img = np.transpose(img, (2, 0, 1))
        self.img = img
    
    def __len__(self):
        return 1000

    def get_example(self, i):
        return self.img
    

def draw_cornelbox_1(output):
    epoch = 1000

    outdir = os.path.dirname(output)
    light = np.array([300, 545, 300], dtype=np.float32)
    model = ArrayLink(light)
    func = RaytraceFunc()

    chainer.config.autotune = True
    chainer.cudnn_fast_batch_normalization = True

    optimizer = optimizers.Adam(alpha=1e-1, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)

    #dataset
    train_dataset = RaytraceDataset(output)
    train_iter = chainer.iterators.SerialIterator(train_dataset, 1, shuffle=True)

    #updator
    updater = RaytraceUpdater(train_iter, model, func, optimizer)

    #trainer
    trainer = training.Trainer(updater, (epoch, 'epoch'), outdir)

    log_interval = (1, 'iteration')

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(
        extensions.PrintReport([
            'epoch',
            'iteration',
            'main/loss',
            'pos/light_x',
            'pos/light_y',
            'pos/light_z',
        ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=1))

    trainer.run()

    return 0


def process(args):
    output = args.output
    #ret = draw_cornelbox_0(output)
    ret = draw_cornelbox_1(output)

    return ret


def main() -> int:
    parser = argparse.ArgumentParser(description='DRT')
    parser.add_argument(
        '--output', '-o', default='./data/backprop_cornelbox/first.png', help='output file directory path')
    args = parser.parse_args()
    return process(args)


if __name__ == '__main__':
    exit(main())
