import os
import sys
import argparse
import math

import cv2
import numpy as np
import chainer
from chainer import Variable
import chainer.functions as F

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../..')))

from drt.camera import PerspectiveCamera
from drt.shape import SphereShape, PlaneShape, CompositeShape
from drt.material import NormalMaterial
from drt.vec import vdot, vnorm


def process1(output):
    origin = Variable(np.array([0, 0, 0], np.float32))
    radius = Variable(np.array([0.5], np.float32))
    s = SphereShape(origin, radius)

    cam = PerspectiveCamera(256, 256, 60, [0,0,-3])
    ro, rd = cam.shoot()
    C, H, W = ro.shape[:3]
    #print(C, H, W)
    t0 = Variable(np.broadcast_to(
        np.array(
        [0.01], np.float32).reshape((1, 1, 1)), (1, H, W)))
    t1 = Variable(np.broadcast_to(
        np.array([10000], np.float32).reshape((1, 1, 1)), (1, H, W)))
    #print("t0", t0.shape)
    #print("t1", t1.shape)
    b, t, p, n = s.intersect(ro, rd, t0, t1)
    #print("b", b.shape)
    mat = NormalMaterial()
    info = {'b': b, 't': t, 'p': p, 'n': n}
    img = mat.render(info)

    img = img.data
    img = (img * 255).astype(np.uint8)
    if output is not None:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        cv2.imwrite(output, img)


def process2(output):
    pt = PlaneShape(Variable(np.array([0, 1, 0], np.float32)), Variable(
        vnorm(np.array([0, -1, 0], np.float32)).data))
    pb = PlaneShape(Variable(np.array(
        [0, -1, 0], np.float32)), Variable(vnorm(np.array([0, 1, 0], np.float32)).data))
    pd = PlaneShape(Variable(np.array([0, 0, 1], np.float32)), Variable(
        vnorm(np.array([0, 0, -1], np.float32)).data))
    pl = PlaneShape(Variable(np.array(
        [-1, 0, 0], np.float32)), Variable(vnorm(np.array([1, 0, 0], np.float32)).data))
    pr = PlaneShape(Variable(np.array([1, 0, 0], np.float32)), Variable(
        vnorm(np.array([-1, 0, 0], np.float32)).data))
    s = SphereShape(Variable(np.array([0, 0, 0], np.float32)), Variable(
        np.array([0.45], np.float32)))
    obj = CompositeShape([pt, pb, pd, pl, pr, s])
    cam = PerspectiveCamera(256, 256, 60, [0,0,-3])
    ro, rd = cam.shoot()
    C, H, W = ro.shape[:3]
    t0 = Variable(np.broadcast_to(
        np.array([0.01], np.float32).reshape((1, 1, 1)), (1, H, W)))
    t1 = Variable(np.broadcast_to(
        np.array([10000], np.float32).reshape((1, 1, 1)), (1, H, W)))
    #print("t0", t0.shape)
    #print("t1", t1.shape)
    b, t, p, n = obj.intersect(ro, rd, t0, t1)
    #print("b", b.shape)
    mat = NormalMaterial()
    info = {'b': b, 't': t, 'p': p, 'n': n}
    img = mat.render(info)

    img = img.data
    img = (img * 255).astype(np.uint8)
    if output is not None:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        cv2.imwrite(output, img)


def process(args):
    output = args.output
    dirname = os.path.dirname(output)
    filename, ext = os.path.splitext(os.path.basename(output))
    path0 = output
    process1(path0)
    path1 = '{0}/{1}{2}{3}'.format(dirname, filename, 0, ext)
    process1(path1)
    path2 = '{0}/{1}{2}{3}'.format(dirname, filename, 1, ext)
    process2(path2)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='DRT')
    parser.add_argument('--input', '-i', default='data',
                        help='input file directory path')
    parser.add_argument('--output', '-o', default='data/test0/test.png',
                        help='output file directory path')
    args = parser.parse_args()
    return process(args)


if __name__ == '__main__':
    exit(main())
