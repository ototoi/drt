
import os
import sys
import argparse
import glob

import numpy as np

from chainer import functions as F


sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..')))

from drt.io.obj_mesh_loader import ObjMeshLoader
from drt.shape.mesh_shape import MeshLink, MeshShape
from drt.shape.mesh_accelerator import SWMeshAccelerator
from drt.camera import PerspectiveCamera

from drt.utils.set_item import set_item

def process(args):
    path = args.input
    loader = ObjMeshLoader()
    meshes, _ = loader.load(path)
    mesh_data = meshes[0]
    indices   = np.array([idx.vertex_index for idx in mesh_data.indices], dtype=np.int32)
    positions = np.array(mesh_data.positions, dtype=np.float32).reshape((-1, 3))
    mesh_link = MeshLink(indices, positions)
    print(mesh_data)
    print(mesh_link)

    """
    a = np.array([[0, 1, 2], [3, 4, 5]])
    b = np.array([[10, 20], [40, 50]])
    indices = [slice(0, 2), slice(0, 2)] 
    print(indices)
    c = set_item(a, indices, b, inplace=False)
    print (a, b, c)
    """


    for i in range(len(mesh_link)):
        t = mesh_link.get_triangle(i)
        #print(t)
    #print(len(mesh_link))
    mesh_shape = MeshShape(mesh_link, accelerator=SWMeshAccelerator())
    print(mesh_shape)
    fov = 45.0
    camera = PerspectiveCamera(512, 512, fov, origin=[0, 0, -10], direction=[0, 0, 1])

    B = 1
    ro, rd, t0, t1 = camera.shoot()
    C, H, W = ro.shape[:3]
    ro = F.broadcast_to(ro.reshape((1, C, H, W)), (B, C, H, W))
    rd = F.broadcast_to(rd.reshape((1, C, H, W)), (B, C, H, W))
    t0 = F.broadcast_to(t0.reshape((1, 1, H, W)), (B, 1, H, W))
    t1 = F.broadcast_to(t1.reshape((1, 1, H, W)), (B, 1, H, W))

    info = mesh_shape.intersect(ro, rd, t0, t1)
    





def main() -> int:
    parser = argparse.ArgumentParser(description='DRT')
    parser.add_argument('--input', '-i', default='./data/load_and_render_obj/bunny.obj', help='input file path')
    args = parser.parse_args()
    return process(args)


if __name__ == '__main__':
    exit(main())

