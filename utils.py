import taichi as ti
import numpy as np
import random

PI = 3.141592653589793
EPSILON = 1e-5
FMIN = np.finfo(np.float32).min
FMAX = np.finfo(np.float32).max


def deg_to_rad(deg):
    return deg * PI / 180.0


def rand3(lo=0.0, hi=1.0):
    return ti.Vector([random.uniform(lo, hi), random.uniform(lo, hi), random.uniform(lo, hi)])


def clip(x, lo, hi):
    return min(max(x, lo), hi)


def minimum(vec1, vec2):
    return ti.Vector([min(vec1[0], vec2[0]),
                      min(vec1[1], vec2[1]),
                      min(vec1[2], vec2[2])])


def maximum(vec1, vec2):
    return ti.Vector([max(vec1[0], vec2[0]),
                      max(vec1[1], vec2[1]),
                      max(vec1[2], vec2[2])])


def get_rot_mat(theta):
    sin_theta = np.sin(theta / 180.0 * PI)
    cos_theta = np.cos(theta / 180.0 * PI)
    rot_mat = np.array([[cos_theta, 0.0, -sin_theta], [0.0, 1.0, 0.0], [sin_theta, 0.0, cos_theta]])
    return rot_mat


@ti.func
def rand_ti(lo=0.0, hi=1.0):
    x = ti.random()
    return x * (hi - lo) + lo


@ti.func
def rand3_ti(lo=0.0, hi=1.0):
    return ti.Vector([rand_ti(lo, hi), rand_ti(lo, hi), rand_ti(lo, hi)])


@ti.func
def random_in_unit_sphere():
    """
    Uniformly sample a point from the sphere surface.
    Returns:
        coords
    """
    p = 2.0 * ti.Vector([ti.random(), ti.random(), ti.random()]) - ti.Vector([1.0, 1.0, 1.0])
    while p.norm() > 1.0:
        p = 2.0 * ti.Vector([ti.random(), ti.random(), ti.random()]) - ti.Vector([1.0, 1.0, 1.0])
    return p


@ti.func
def random_unit_vector():
    return random_in_unit_sphere().normalized()


@ti.func
def near_zero(p):
    return (ti.abs(p) < 1e-8).all()
