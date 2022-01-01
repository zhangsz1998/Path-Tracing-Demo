import taichi as ti
import numpy as np
from material import get_texture_value, scatter


@ti.data_oriented
class Renderer:

    def __init__(self, image_width, image_height, scene, camera, spp=16, max_depth=10, p_RR=0.8):
        self.image_width = image_width
        self.image_height = image_height
        self.canvas = ti.Vector.field(3, ti.f32, shape=(image_width, image_height))
        self.scene = scene
        self.camera = camera
        self.spp = spp
        self.cnt = ti.field(ti.i32, shape=())
        self.reset()
        self.max_depth = max_depth
        self.p_RR = p_RR

    @ti.func
    def ray_color(self, ray, si, sj, max_depth=10, p_RR=0.8):
        color = ti.Vector([0.0, 0.0, 0.0])
        attenuation = ti.Vector([1.0, 1.0, 1.0])
        for i in range(max_depth):
            if ti.random() > p_RR:
                break
            is_hit, t, p, u, v, n, f, m = self.scene.intersect_p(ray, si, sj)
            if is_hit:
                if m.m_type == 0:
                    color = get_texture_value(m, p, u, v) * attenuation
                    break
                else:
                    is_scatter, scattered, attenuation_tmp = scatter(ray, p, u, v, n, f, m)
                    if not is_scatter:
                        break
                    attenuation = attenuation * attenuation_tmp
                    ray = scattered
            else:
                t = 0.5 * (ray.direction[1] + 1.0)
                attenuation_tmp = (1.0 - t) * ti.Vector([1.0, 1.0, 1.0]) + t * ti.Vector([0.5, 0.7, 1.0])
                color = attenuation * attenuation_tmp / p_RR
                break
            attenuation /= p_RR
        return color

    # @ti.func
    # def ray_color(self, ray, si, sj, max_depth=10, p_RR=0.8):
    #     """
    #     Color Only.
    #     """
    #     is_hit, t, p, u, v, n, f, m = self.scene.intersect_p(ray, si, sj)
    #     color = ti.Vector([0.0, 0.0, 0.0])
    #     if is_hit:
    #         color = get_texture_value(m, p, u, v)
    #     else:
    #         t = 0.5 * (ray.direction[1] + 1.0)
    #         color = (1.0 - t) * ti.Vector([1.0, 1.0, 1.0]) + t * ti.Vector([0.5, 0.7, 1.0])
    #     return color

    @ti.kernel
    def render(self):
        for i, j in self.canvas:
            u = (i + ti.random()) / self.image_width
            v = (j + ti.random()) / self.image_height
            ray = self.camera.get_ray(u, v)
            color = ti.Vector([0.0, 0.0, 0.0])
            for n in range(self.spp):
                color += self.ray_color(ray, i, j, self.max_depth, self.p_RR)
            color /= self.spp
            self.canvas[i, j] += color
        self.cnt[None] += 1

    def get_canvas_numpy(self):
        return np.sqrt(self.canvas.to_numpy() / self.cnt[None])

    def reset(self):
        self.canvas.fill(0.0)
        self.cnt[None] = 0
