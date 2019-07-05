
import os
import sys
import argparse
import glob

import numpy as np


sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..')))

from drt.io.obj_mesh_loader import ObjMeshLoader
from drt.shape.mesh_shape import MeshLink, MeshShape


def process(args):
    path = args.input
    loader = ObjMeshLoader()
    meshes, materials = loader.load(path)
    mesh_data = meshes[0]
    indices   = np.array([idx.vertex_index for idx in mesh_data.indices], dtype=np.int32)
    positions = np.array(mesh_data.positions, dtype=np.float32).reshape((-1, 3))
    mesh_link = MeshLink(indices, positions)
    print(mesh_data)
    print(mesh_link)
    for i in range(len(mesh_link)):
        t = mesh_link.get_triangle(i)
        print(t)
    mesh_shape = MeshShape(mesh_link)
    print(mesh_shape)

    





def main() -> int:
    parser = argparse.ArgumentParser(description='DRT')
    parser.add_argument('--input', '-i', default='./data/load_and_render_obj/bunny.obj', help='input file path')
    args = parser.parse_args()
    return process(args)


if __name__ == '__main__':
    exit(main())

