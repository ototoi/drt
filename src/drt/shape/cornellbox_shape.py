from . import MaterizedShape, RectangleShape, CompositeShape


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

    cmps = CompositeShape(
        [plane_top, plane_bottom, plane_back, plane_left, plane_right])
    return cmps


def create_shortblock(materials):
    mat_w = materials["white"]
    a = RectangleShape([130.0, 165.0,  65.0], [82.0, 165.0, 225.0], [
                       240.0, 165.0, 272.0], [290.0, 165.0, 114.0])
    b = RectangleShape([290.0, 0.0, 114.0], [290.0, 165.0, 114.0], [
                       240.0, 165.0, 272.0], [240.0, 0.0, 272.0])
    c = RectangleShape([130.0, 0.0, 65.0], [130.0, 165.0, 65.0], [
                       290.0, 165.0, 114.0], [290.0, 0.0, 114.0])
    d = RectangleShape([82.0, 0.0, 225.0], [82.0, 165.0, 225.0], [
                       130.0, 165.0, 65.0], [130.0, 0.0, 65.0])
    e = RectangleShape([240.0, 0.0, 272.0], [240.0, 165.0, 272.0], [
                       82.0, 165.0, 225.0], [82.0, 0.0, 225.0])
    cmps = MaterizedShape(CompositeShape([a, b, c, d, e]), mat_w)
    return cmps


def create_tallblock(materials):
    mat_w = materials["white"]
    a = RectangleShape([423.0, 330.0, 247.0], [265.0, 330.0, 296.0], [
                       314.0, 330.0, 456.0], [472.0, 330.0, 406.0])
    b = RectangleShape([423.0, 0.0, 247.0], [423.0, 330.0, 247.0], [
                       472.0, 330.0, 406.0], [472.0, 0.0, 406.0])
    c = RectangleShape([472.0, 0.0, 406.0], [472.0, 330.0, 406.0], [
                       314.0, 330.0, 456.0], [314.0, 0.0, 456.0])
    d = RectangleShape([314.0, 0.0, 456.0], [314.0, 330.0, 456.0], [
                       265.0, 330.0, 296.0], [265.0, 0.0, 296.0])
    e = RectangleShape([265.0, 0.0, 296.0], [265.0, 330.0, 296.0], [
                       423.0, 330.0, 247.0], [423.0, 0.0, 247.0])
    cmps = MaterizedShape(CompositeShape([a, b, c, d, e]), mat_w)
    return cmps
