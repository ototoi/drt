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
from drt.utils import add_to_model as AM
from drt.vec import vdot, vnorm
from drt.renderer import NormalRenderer, DiffuseRenderer, AlbedoRenderer
from drt.material import DiffuseMaterial
from drt.shape import SphereShape, PlaneShape, TriangleShape, RectangleShape, CompositeShape, MaterizedShape
from drt.camera import PerspectiveCamera
from drt.shape.cornellbox_shape import create_light, create_floor, create_shortblock, create_tallblock


class RaytraceFunc(object):
    def __init__(self, shape, light, camera):
        renderer = DiffuseRenderer()
        self.camera = camera
        self.shape = shape
        self.renderer = renderer
        self.ll = [light]

    def __call__(self, B):
        #print("C----")
        ro, rd, t0, t1 = self.camera.shoot()
        #print("D----")
        C, H, W = ro.shape[:3]
        ro = F.broadcast_to(ro.reshape((1, C, H, W)), (B, C, H, W))
        rd = F.broadcast_to(rd.reshape((1, C, H, W)), (B, C, H, W))
        t0 = F.broadcast_to(t0.reshape((1, 1, H, W)), (B, 1, H, W))
        t1 = F.broadcast_to(t1.reshape((1, 1, H, W)), (B, 1, H, W))

        #print("E----")
        info = self.shape.intersect(ro, rd, t0, t1)
        info['ro'] = ro
        info['rd'] = rd
        #print("F----")

        #x = x[0, :].reshape((1, 3))
        
        info['ll'] = self.ll
        img = self.renderer.render(info)
        #print("G----")
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
    def __init__(self, iterator, model, func, optimizer, odir, device=None):
        super(RaytraceUpdater, self).__init__(iterator, optimizer, device=device)
        self.model = model
        self.func = func
        self.odir = odir
        self.count = 0


    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer  = self.get_optimizer('main')

        batch = train_iter.next()
        t_data = self.converter(batch, self.device)
        B = t_data.shape[0]
        y_data = self.func(B)
        B, C, H, W = t_data.shape[:4]
        y_data = F.broadcast_to(y_data, (B, C, H, W))
        loss = compute_loss(y_data, t_data)

        reporter.report({
            'main/loss': loss,
            'pos/light_x': self.model.data[0],
            'pos/light_y': self.model.data[1],
            'pos/light_z': self.model.data[2]
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

        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()
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


START_POS = [300, 545, 300]
GOAL_POS  = [300, 500, 300]


def draw_goal_cornelbox(output, device=-1):
    materials = {}
    materials["light"] = DiffuseMaterial([1.0, 1.0, 1.0])
    materials["white"] = DiffuseMaterial([0.5, 0.5, 0.5])
    materials["green"] = DiffuseMaterial([0.0, 1.0, 0.0])
    materials["red"] = DiffuseMaterial([1.0, 0.0, 0.0])
    #shape_light = create_light(materials)
    shape_floor = create_floor(materials)
    shape_shortblock = create_shortblock(materials)
    shape_tallblock = create_tallblock(materials)
    shape = CompositeShape([shape_floor, shape_shortblock, shape_tallblock])

    """
    light = np.array(GOAL_POS, dtype=np.float32)
    model = ArrayLink(light)
    light = PointLight(origin=model.data, color=[1, 1, 1])
    """
    
    light = PointLight(origin=GOAL_POS, color=[1, 1, 1])
    model = chainer.Link()
    AM(model, 'data', light.origin)

    fov = math.atan2(0.025, 0.035) * 180.0 / math.pi
    camera = PerspectiveCamera(512, 512, fov, [278.0, 273.0, -800.0])


    func = RaytraceFunc(shape=shape, light=light, camera=camera)

    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()
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


def draw_start_cornelbox(output, device=-1):
    materials = {}
    materials["light"] = DiffuseMaterial([1.0, 1.0, 1.0])
    materials["white"] = DiffuseMaterial([0.5, 0.5, 0.5])
    materials["green"] = DiffuseMaterial([0.0, 1.0, 0.0])
    materials["red"] = DiffuseMaterial([1.0, 0.0, 0.0])
    #shape_light = create_light(materials)
    shape_floor = create_floor(materials)
    shape_shortblock = create_shortblock(materials)
    shape_tallblock = create_tallblock(materials)
    shape = CompositeShape([shape_floor, shape_shortblock, shape_tallblock])

    light = PointLight(origin=START_POS, color=[1, 1, 1])

    fov = math.atan2(0.025, 0.035) * 180.0 / math.pi
    camera = PerspectiveCamera(512, 512, fov, [278.0, 273.0, -800.0])

    func = RaytraceFunc(shape=shape, light=light, camera=camera)

    model = chainer.Link()
    AM(model, 'data', light.origin)

    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()
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
    epoch = 50

    outdir = os.path.dirname(output)

    materials = {}
    materials["light"] = DiffuseMaterial([1.0, 1.0, 1.0])
    materials["white"] = DiffuseMaterial([0.5, 0.5, 0.5])
    materials["green"] = DiffuseMaterial([0.0, 1.0, 0.0])
    materials["red"] = DiffuseMaterial([1.0, 0.0, 0.0])
    #shape_light = create_light(materials)
    shape_floor = create_floor(materials)
    shape_shortblock = create_shortblock(materials)
    shape_tallblock = create_tallblock(materials)
    shape = CompositeShape([shape_floor, shape_shortblock, shape_tallblock])

    light = PointLight(origin=START_POS, color=[1, 1, 1])
    model = chainer.Link()
    AM(model, 'data', light.origin)

    fov = math.atan2(0.025, 0.035) * 180.0 / math.pi
    camera = PerspectiveCamera(512, 512, fov, [278.0, 273.0, -800.0])

    func = RaytraceFunc(shape=shape, light=light, camera=camera)

    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()
        func.to_gpu()

    chainer.config.autotune = True
    chainer.cudnn_fast_batch_normalization = True

    #optimizer = optimizers.Adam(alpha=1e-1, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer = optimizers.SGD(lr=0.001)
    optimizer.setup(model)

    #dataset
    train_dataset = RaytraceDataset(output)
    train_iter = chainer.iterators.SerialIterator(train_dataset, 1, shuffle=True)

    #updator
    updater = RaytraceUpdater(train_iter, model, func, optimizer, outdir, device=device)

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
        '--goal', default='./data/backprop_light_cornelbox/goal.png', help='output file directory path')
    parser.add_argument(
        '--start', default='./data/backprop_light_cornelbox/start.png', help='output file directory path')
    parser.add_argument(
        '--gpu', '-g', type=int, default=-1, help='GPU')
    args = parser.parse_args()
    return process(args)


if __name__ == '__main__':
    exit(main())
