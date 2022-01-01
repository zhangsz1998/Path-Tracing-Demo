import taichi as ti
from utils import FMIN, FMAX, minimum, maximum


Bounds3 = ti.types.struct(
    p_min=ti.types.vector(3, ti.f32),
    p_max=ti.types.vector(3, ti.f32)
)


def init(p_min=ti.Vector([FMAX, FMAX, FMAX]), p_max=ti.Vector([FMIN, FMIN, FMIN])):
    return Bounds3(p_min=p_min, p_max=p_max)


def get_centroid(box):
    return 0.5 * (box.p_min + box.p_max)


def get_diag(box):
    return box.p_max - box.p_min


def bounds_union(box1, box2):
    pp_min = minimum(box1.p_min, box2.p_min)
    pp_max = maximum(box1.p_max, box2.p_max)
    return Bounds3(p_min=pp_min, p_max=pp_max)


def bounds_vec_union(box, vec):
    return Bounds3(p_min=minimum(box.p_min, vec), p_max=maximum(box.p_max, vec))


def max_extent(box):
    d = get_diag(box)
    dim = 2
    if d[0] > d[1] and d[0] > d[2]:
        dim = 0
    elif d[1] > d[2]:
        dim = 1
    return dim


@ti.func
def bounds_intersect(box, ray, t_min=1e-3, t_max=1e9):
    insect = True
    for i in ti.static(range(3)):
        # if abs(ray.direction[i]) > 1e-6:
        inv = 1.0 / ray.direction[i]
        t0 = (box.p_min[i] - ray.origin[i]) * inv
        t1 = (box.p_max[i] - ray.origin[i]) * inv
        if inv < 0.0:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_max < t_min or t_max < 0:
            insect = False
    return insect
