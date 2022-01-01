import taichi as ti

Ray = ti.types.struct(
    origin=ti.types.vector(3, ti.f32),
    direction=ti.types.vector(3, ti.f32)
)


def init(origin=ti.Vector([0.0, 0.0, 0.0]), direction=ti.Vector([1.0, 0.0, 0.0])):
    return Ray(origin=origin, direction=direction)


@ti.func
def ray_at(ray, t):
    return ray.origin + t * ray.direction
