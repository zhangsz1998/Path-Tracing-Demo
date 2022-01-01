import taichi as ti
import numpy as np
import material
from material import Material
from ray import ray_at
from utils import minimum, maximum, EPSILON
from bounds3 import Bounds3


Triangle = ti.types.struct(
    v0=ti.types.vector(3, ti.f32),
    v1=ti.types.vector(3, ti.f32),
    v2=ti.types.vector(3, ti.f32),
    e1=ti.types.vector(3, ti.f32),
    e2=ti.types.vector(3, ti.f32),
    t0=ti.types.vector(2, ti.f32),
    t1=ti.types.vector(2, ti.f32),
    t2=ti.types.vector(2, ti.f32),
    normal=ti.types.vector(3, ti.f32),
    m=Material
)


def init(v0=ti.Vector([0.0, 0.0, 0.0]), v1=ti.Vector([0.0, 0.0, 0.0]), v2=ti.Vector([0.0, 0.0, 0.0]),
         t0=ti.Vector([0.0, 0.0]), t1=ti.Vector([0.0, 0.0]), t2=ti.Vector([0.0, 0.0]),
         n0=ti.Vector([0.0, 0.0, 0.0]), n1=ti.Vector([0.0, 0.0, 0.0]), n2=ti.Vector([0.0, 0.0, 0.0]),
         m=material.init()):
    e1 = v1 - v0
    e2 = v2 - v0
    normal = e1.cross(e2).normalized()
    interp_normal = (n0 + n1 + n2)
    if interp_normal.norm() > EPSILON and interp_normal.dot(normal) < 0.0:
        normal = -normal
    return Triangle(v0=v0, v1=v1, v2=v2, e1=e1, e2=e2, t0=t0, t1=t1, t2=t2, normal=normal, m=m)


def get_bounds(tri):
    p_min = minimum(minimum(tri.v0, tri.v1), tri.v2)
    p_max = maximum(maximum(tri.v0, tri.v1), tri.v2)
    return Bounds3(p_min=p_min, p_max=p_max)


def get_area(tri):
    return tri.e1.cross(tri.e2).norm() * 0.5


def get_uv(tri, coords):
    inv_S = 0.5 / get_area(tri)
    alpha = (tri.v1 - coords).cross(tri.v2 - coords).norm() * inv_S
    beta = (tri.v0 - coords).cross(tri.v2 - coords).norm() * inv_S
    return tri.t0 * alpha + tri.t1 * beta + tri.t2 * (1.0 - alpha - beta)


@ti.func
def get_normal(tri):
    return tri.normal


@ti.func
def intersect_p(tri, ray, t_min=1e-3, t_max=1e9):
    is_hit = False
    t = np.inf
    coords = ti.Vector([0.0, 0.0, 0.0])
    front_face = True
    pvec = ray.direction.cross(tri.e2)
    det = tri.e1.dot(pvec)
    normal = tri.normal
    if det != 0.0:
        tvec = ray.origin - tri.v0
        u = tvec.dot(pvec) / det
        if 0.0 <= u <= 1.0:
            qvec = tvec.cross(tri.e1)
            v = ray.direction.dot(qvec) / det
            if v >= 0.0 and u + v <= 1.0:
                t = tri.e2.dot(qvec) / det
                if t_min <= t <= t_max:
                    is_hit = True
                    coords = ray_at(ray, t)
                    if normal.dot(ray.direction) >= 0.0:
                        front_face = False
                        normal = -normal
    return is_hit, t, coords, normal, front_face, tri.m
