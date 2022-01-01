import taichi as ti
from BVH import BVHAccel


@ti.data_oriented
class Scene:

    def __init__(self, spheres=list(), triangles=list(), stack_width=512, stack_height=512, stack_depth=20):
        self.bvh = BVHAccel(spheres, triangles, 1, stack_width, stack_height, stack_depth)

    @ti.func
    def intersect_p(self, ray, si, sj, t_min=1e-3, t_max=1e9):
        return self.bvh.intersect_p(ray, si, sj, t_min, t_max)
