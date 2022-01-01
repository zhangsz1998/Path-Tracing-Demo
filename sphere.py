import taichi as ti
import numpy as np
from bounds3 import Bounds3
from ray import Ray, ray_at
import material
from material import Material
from utils import PI, EPSILON

Sphere = ti.types.struct(
    center=ti.types.vector(3, ti.f32),
    radius=ti.f32,
    m=Material,
)


def init(center=ti.Vector([0.0, 0.0, 0.0]), radius=1.0, m=material.init()):
    return Sphere(center=center, radius=radius, m=m)


def get_bounds(sphere):
    p_min = sphere.center - ti.Vector([sphere.radius, sphere.radius, sphere.radius])
    p_max = sphere.center + ti.Vector([sphere.radius, sphere.radius, sphere.radius])
    return Bounds3(p_min=p_min, p_max=p_max)


def get_area(sphere):
    return 4 * PI * sphere.radius ** 2


@ti.func
def intersect_p(sphere, ray, t_min=1e-3, t_max=1e9):
    is_hit = False
    t = np.inf
    coords = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 0.0, 0.0])
    front_face = True
    L = ray.origin - sphere.center
    a = ray.direction.dot(ray.direction)
    b = 2.0 * ray.direction.dot(L)
    c = L.dot(L) - sphere.radius * sphere.radius
    delta = b ** 2 - 4.0 * a * c
    if delta > 0.0:
        r_delta = ti.sqrt(delta)
        t0 = (-b - r_delta) / (2.0 * a)
        t1 = (-b + r_delta) / (2.0 * a)
        if t0 < t_min or t0 > t_max:
            t0 = t1
        if t_min <= t0 <= t_max:
            is_hit = True
            coords = ray_at(ray, t0)
            normal = (coords - sphere.center).normalized()
            t = t0
            if normal.dot(ray.direction) >= 0.0:
                front_face = False
                normal = -normal
    return is_hit, t, coords, normal, front_face, sphere.m


@ti.func
def get_uv(sphere, p):
    p_normed = (p - sphere.center) / sphere.radius
    theta = ti.acos(-p_normed[1])
    phi = ti.atan2(-p_normed[2], p_normed[0]) + PI
    u = phi / (2 * PI)
    v = theta / PI
    return u, v
