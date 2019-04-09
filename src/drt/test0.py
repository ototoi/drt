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

from drt.material import NormalMaterial
from drt.shape import SphereShape, PlaneShape, CompositeShape


def cam():
    W = 256
    H = 256
    angle = 60
    ang = (angle/2) * math.pi / 180.0
    HH = math.tan(ang)
    ro = np.zeros((H, W, 3), np.float32)
    ro[:, :, 2] = -3.0
    rd = np.zeros((H, W, 3), np.float32)
    for y in range(H):
        yy = 1 - 2*(y + 0.5)/H
        for x in range(W):
            xx = 2*(x+0.5)/W - 1
            yyy = yy * HH
            xxx = xx * HH
            r = np.array([xxx, yyy, 1], np.float32)
            r = r / np.linalg.norm(r)
            #r = np.array([0, 0, 1], np.float32)
            rd[y, x, :] = r

    ro = np.transpose(ro, (2, 0, 1))
    rd = np.transpose(rd, (2, 0, 1))

    return ro, rd


def vdot(a, b):
    m = a * b
    return F.sum(m, axis=0)

def vnorm(a):
    l = F.sqrt(vdot(a, a))
    return a / l

def process1(output):
    origin = Variable(np.array([0, 0, 0], np.float32))
    radius = Variable(np.array([0.5], np.float32))
    s = SphereShape(origin, radius)

    ro, rd = cam()
    ro = Variable(ro)
    rd = Variable(rd)
    C, H, W = ro.shape[:3]
    t0 = Variable(np.broadcast_to(
        np.array([0.01], np.float32).reshape((1, 1, 1)), (1, H, W)))
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
    pt = PlaneShape(Variable(np.array([0, 1, 0], np.float32)), Variable(vnorm(np.array([0, -1, 0], np.float32)).data))
    pb = PlaneShape(Variable(np.array([0, -1, 0], np.float32)), Variable(vnorm(np.array([0, 1, 0], np.float32)).data))
    pd = PlaneShape(Variable(np.array([0, 0, 1], np.float32)), Variable(vnorm(np.array([0, 0, -1], np.float32)).data))
    pl = PlaneShape(Variable(np.array([-1, 0, 0], np.float32)), Variable(vnorm(np.array([1, 0, 0], np.float32)).data))
    pr = PlaneShape(Variable(np.array([1, 0, 0], np.float32)), Variable(vnorm(np.array([-1, 0, 0], np.float32)).data))
    s = SphereShape(Variable(np.array([0, 0, 0], np.float32)), Variable(np.array([0.45], np.float32)))
    obj = CompositeShape([pt, pb, pd, pl, pr, s])
    ro, rd = cam()
    ro = Variable(ro)
    rd = Variable(rd)
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
    parser = argparse.ArgumentParser(description='predict pose')
    parser.add_argument('--input', '-i', default=None,
                        help='input file directory path')
    parser.add_argument('--output', '-o', default=None,
                        help='output file directory path')
    args = parser.parse_args()
    return process(args)


if __name__ == '__main__':
    exit(main())
