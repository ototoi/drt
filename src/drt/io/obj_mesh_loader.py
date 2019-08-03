import sys
import tinyobjloader


class Mesh(object):
    """
        Mesh: a mesh
    """
    def __init__(self, name, indices, positions, uvs=None, normals=None, material_ids=None):
        self.name = name
        self.indices = indices
        self.positions = positions
        self.uvs = uvs
        self.normals = normals
        self.material_ids = material_ids


class ObjMeshLoader(object):
    """
        ObjMeshLoader: a loader class which loads meshes from "*.obj" file.
    """
    def __init__(self):
        self.reader = tinyobjloader.ObjReader()

    def load(self, path):
        reader = self.reader
        option = tinyobjloader.ObjReaderConfig()
        option.triangulate = True
        ret = reader.ParseFromFile(path, option=option)
        if ret == False:
            return None
        attrib = reader.GetAttrib()
        #print("attrib.vertices = ", len(attrib.vertices))
        #print("attrib.normals = ", len(attrib.normals))
        #print("attrib.texcoords = ", len(attrib.texcoords))

        materials = reader.GetMaterials()
        #print("Num materials: ", len(materials))
        #for m in materials:
        #    print(m.name)
        #    print(m.diffuse)

        meshs = []
        shapes = reader.GetShapes()
        #print("Num shapes: ", len(shapes))
        for shape in shapes:
            #print(shape.name)
            #print("num_indices = {}".format(len(shape.mesh.indices)))
            mesh = Mesh(shape.name, shape.mesh.indices, attrib.vertices, attrib.texcoords, attrib.normals, shape.mesh.material_ids)
            meshs.append(mesh)

        return meshs, materials
