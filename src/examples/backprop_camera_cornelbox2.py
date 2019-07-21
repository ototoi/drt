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
from chainer import serializers, optimizers as O, functions as F
from chainer.dataset import DatasetMixin

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..')))

from drt.light import PointLight
from drt.utils import make_parameter as MP
from drt.utils import add_parameter as AP
from drt.vec import vdot, vnorm
from drt.renderer import NormalRenderer, DiffuseRenderer, AlbedoRenderer
from drt.material import DiffuseMaterial
from drt.shape import SphereShape, PlaneShape, TriangleShape, RectangleShape, CompositeShape, MaterizedShape
from drt.camera import PerspectiveCamera
from drt.shape.cornellbox_shape import create_light, create_floor, create_shortblock, create_tallblock

START_POS = [300, 545, 300]
GOAL_POS  = [300, 545, 300]

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


def compute_loss(data1, data2):
    _, _, H, W = data1.shape[:4]
    return F.sum(F.absolute(data1-data2)) * (H * W) / (1024 * 1024)


def save_progress_image(odir, i, img):
    path = os.path.join(odir, '{0:08d}.png'.format(i))
    cv2.imwrite(path, img)

class RaytraceUpdater(StandardUpdater):
    def __init__(self, iterator, models, func, optimizers, odir, device=-1):
        optimizer = list(optimizers.values())[0]
        super(RaytraceUpdater, self).__init__(iterator, optimizer, device=device)
        self.models = models
        self.optimizers = optimizers
        self.func = func
        self.odir = odir
        self.count = 0
        self.device = device

    def update_core(self):
        train_iter = self.get_iterator('main')

        batch = train_iter.next()
        t_data = self.converter(batch, self.device)
        B = t_data.shape[0]
        y_data = self.func(B)
        B, C, H, W = t_data.shape[:4]
        y_data = F.broadcast_to(y_data, (B, C, H, W))
        loss = compute_loss(y_data, t_data)

        pos_model = self.models['position']
        dir_model = self.models['direction']

        reporter.report({
            'main/loss': loss,
            'camera_position/x': pos_model.camera_position[0],
            'camera_position/y': pos_model.camera_position[1],
            'camera_position/z': pos_model.camera_position[2],
            'camera_direction/x': dir_model.camera_zaxis[0],
            'camera_direction/y': dir_model.camera_zaxis[1],
            'camera_direction/z': dir_model.camera_zaxis[2]
        })

        y_data = y_data.data
        if self.device >= 0:
            y_data = y_data.get()
            cuda.get_device_from_id(self.device).synchronize()

        img = y_data[0]
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        save_progress_image(self.odir, self.count, img)

        for o in self.optimizers.values():
            o.target.cleargrads()
        
        loss.backward()

        for o in self.optimizers.values():
            o.update()
        self.count += 1
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



def draw_goal_cornelbox(output, device=-1):
    materials = {}
    materials["light"] = DiffuseMaterial([1.0, 1.0, 1.0])
    materials["white"] = DiffuseMaterial([0.5, 0.5, 0.5])
    materials["green"] = DiffuseMaterial([0.0, 1.0, 0.0])
    materials["red"] = DiffuseMaterial([1.0, 0.0, 0.0])

    shape_floor = create_floor(materials)
    shape_shortblock = create_shortblock(materials)
    shape_tallblock = create_tallblock(materials)
    shape = CompositeShape([shape_floor, shape_shortblock, shape_tallblock])

    fov = math.atan2(0.025, 0.035) * 180.0 / math.pi
    camera = PerspectiveCamera(512, 512, fov, origin=[278.0, 273.0, -800.0], direction=[0, 0, 1])
    light = PointLight(origin=START_POS, color=[0.1, 0.1, 0.1])

    func = RaytraceFunc(shape=shape, light=light, camera=camera)

    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        func.to_gpu()

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

def norm(v):
    v = np.array(v, dtype=np.float32)
    v = v / np.sqrt(np.sum(v*v))
    return v.tolist()

def draw_start_cornelbox(output, device=-1):
    materials = {}
    materials["light"] = DiffuseMaterial([1.0, 1.0, 1.0])
    materials["white"] = DiffuseMaterial([0.5, 0.5, 0.5])
    materials["green"] = DiffuseMaterial([0.0, 1.0, 0.0])
    materials["red"] = DiffuseMaterial([1.0, 0.0, 0.0])

    shape_floor = create_floor(materials)
    shape_shortblock = create_shortblock(materials)
    shape_tallblock = create_tallblock(materials)
    shape = CompositeShape([shape_floor, shape_shortblock, shape_tallblock])

    fov = math.atan2(0.025, 0.035) * 180.0 / math.pi
    camera = PerspectiveCamera(512, 512, fov, origin=[400.0, 300, -800.0], direction=norm([-0.1, 0, 1]))
    light = PointLight(origin=START_POS, color=[0.1, 0.1, 0.1])

    func = RaytraceFunc(shape=shape, light=light, camera=camera)

    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        func.to_gpu()

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


def calc_goal_cornelbox(output, device=-1):
    epoch = 100
    outdir = os.path.dirname(output)

    materials = {}
    materials["light"] = DiffuseMaterial([1.0, 1.0, 1.0])
    materials["white"] = DiffuseMaterial([0.5, 0.5, 0.5])
    materials["green"] = DiffuseMaterial([0.0, 1.0, 0.0])
    materials["red"] = DiffuseMaterial([1.0, 0.01, 0.01])

    shape_floor = create_floor(materials)
    shape_shortblock = create_shortblock(materials)
    shape_tallblock = create_tallblock(materials)
    shape = CompositeShape([shape_floor, shape_shortblock, shape_tallblock])

    # origin=[278.0, 273.0, -800.0], direction=[0, 0, 1])
    # origin=[400.0, 300, -800.0], direction=norm([-0.1, 0, 1])
    fov = math.atan2(0.025, 0.035) * 180.0 / math.pi
    camera = PerspectiveCamera(512, 512, fov, origin=[270.0, 273.0, -800.0], direction=norm([-0.1, 0, 0.8]))
    light = PointLight(origin=START_POS, color=[0.1, 0.1, 0.1])

    func = RaytraceFunc(shape=shape, light=light, camera=camera)

    model1 = chainer.Link()
    AP(model1, 'camera_position', camera.origin)

    model2 = chainer.Link()
    AP(model2, 'camera_zaxis', camera.zaxis)
    AP(model2, 'camera_xaxis', camera.xaxis)
    AP(model2, 'camera_yaxis', camera.yaxis)
    
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        model1.to_gpu()
        model2.to_gpu()
        func.to_gpu()

    chainer.config.autotune = True
    chainer.cudnn_fast_batch_normalization = True

    optimizer1 = O.SGD(lr=1e-3)
    optimizer1.setup(model1)

    optimizer2 = O.SGD(lr=1e-7)
    optimizer2.setup(model2)

    #dataset
    train_dataset = RaytraceDataset(output)
    train_iter = chainer.iterators.SerialIterator(train_dataset, 1, shuffle=True)

    dict_models = {'position': model1, 'direction': model2}
    dict_optimizers = {'position': optimizer1, 'direction': optimizer2}

    #updator
    updater = RaytraceUpdater(train_iter, dict_models, func, dict_optimizers, outdir, device=device)

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
            'camera_position/x',
            'camera_position/y',
            'camera_position/z',
            'camera_direction/x',
            'camera_direction/y',
            'camera_direction/z'
        ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=1))

    trainer.run()

    return 0


def process(args):
    start = args.start
    goal = args.goal
    gpu = args.gpu
    os.makedirs(os.path.dirname(start), exist_ok=True)
    os.makedirs(os.path.dirname(goal), exist_ok=True)
    ret = draw_start_cornelbox(start, gpu)
    ret = draw_goal_cornelbox(goal, gpu)
    ret = calc_goal_cornelbox(goal, gpu)

    return ret


def main() -> int:
    parser = argparse.ArgumentParser(description='DRT')
    parser.add_argument(
        '--goal', default='./data/backprop_camera_cornelbox2/goal.png', help='output file directory path')
    parser.add_argument(
        '--start', default='./data/backprop_camera_cornelbox2/start.png', help='output file directory path')
    parser.add_argument(
        '--gpu', '-g', type=int, default=-1, help='GPU')
    args = parser.parse_args()
    return process(args)


if __name__ == '__main__':
    exit(main())
