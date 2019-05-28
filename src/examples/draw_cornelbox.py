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
    os.path.dirname(os.path.abspath(__file__)), '..')))

from drt.camera import PerspectiveCamera
from drt.shape import SphereShape, PlaneShape, TriangleShape, CompositeShape, MaterizedShape
from drt.material import DiffuseMaterial
from drt.renderer import NormalRenderer, DiffuseRenderer
from drt.vec import vdot, vnorm
from drt.utils import make_parameter as MP
from drt.light import PointLight


def draw_cornelbox(output):
    pt = PlaneShape([0, 1, 0], [0, -1, 0])
    pt = MaterizedShape(pt, DiffuseMaterial([1, 0, 0]))
    pb = PlaneShape([0, -1, 0], [0, 1, 0])
    pb = MaterizedShape(pb, DiffuseMaterial([0, 1, 0]))
    pd = PlaneShape([0, 0, 1], [0, 0, -1])
    pd = MaterizedShape(pd, DiffuseMaterial([1, 1, 0]))
    pl = PlaneShape([-1, 0, 0], [1, 0, 0])
    pl = MaterizedShape(pl, DiffuseMaterial([0, 1, 1]))
    pr = PlaneShape([1, 0, 0], [-1, 0, 0])
    pr = MaterizedShape(pr, DiffuseMaterial([1, 0, 1]))
    s = SphereShape([0, 0, 0], [0.45])
    s = MaterizedShape(s, DiffuseMaterial([1, 1, 1]))

    cmps = CompositeShape([pt, pb, pd, pl, pr, s])
    cam = PerspectiveCamera(512, 512, 60, [0,0,-3])
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
    l = PointLight(origin=[0.9, 0.9, 0.0], color=[1,1,1])
    info['ll'] = [l]
    #print("b", b.shape)

    renderer = DiffuseRenderer()
    img = renderer.render(info)

    img = img.data
    img = (img * 255).astype(np.uint8)
    if output is not None:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output, img)
    return 0


def process(args):
    output = args.output
    ret = draw_cornelbox(output)
    return ret


def main() -> int:
    parser = argparse.ArgumentParser(description='DRT')
    parser.add_argument('--output', '-o', default='./data/draw_cornelbox/cornelbox.png', help='output file directory path')
    args = parser.parse_args()
    return process(args)


if __name__ == '__main__':
    exit(main())
