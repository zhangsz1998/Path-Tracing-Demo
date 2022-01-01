import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import taichi as ti
ti.init(arch=ti.cuda, device_memory_fraction=0.5)
# ti.init(arch=ti.cpu, debug=True, excepthook=True, cpu_max_num_threads=1)
import random
from scene import Scene
from camera import Camera
from renderer import Renderer
from sphere import Sphere
import material
from material import Material
import triangle as triangle
import utils
import pywavefront
import numpy as np
from tqdm import tqdm


def games101_cow(expand=5.0, shift=np.array([0.0, 0.0, 0.0]).reshape([1, 1, 3]),
         rot=0.0, m=material.init()):
    scene = pywavefront.Wavefront('./data/spot/spot_triangulated_good.obj')
    objects_py = []
    rot_mat = utils.get_rot_mat(rot)
    for name, material in scene.materials.items():
        vertex_info = np.array(material.vertices).reshape([-1, 3, 8])
        texture_uv, normals, vertices = vertex_info[:, :, :2], vertex_info[:, :, 2:5], vertex_info[:, :, 5:]
        vertices = vertices.dot(rot_mat) * expand + shift
        for i in range(vertex_info.shape[0]):
            tri = triangle.init(v0=ti.Vector(list(vertices[i, 0])),
                                v1=ti.Vector(list(vertices[i, 1])),
                                v2=ti.Vector(list(vertices[i, 2])),
                                t0=ti.Vector(list(texture_uv[i, 0])),
                                t1=ti.Vector(list(texture_uv[i, 1])),
                                t2=ti.Vector(list(texture_uv[i, 2])),
                                n0=ti.Vector(list(normals[i, 0])),
                                n1=ti.Vector(list(normals[i, 1])),
                                n2=ti.Vector(list(normals[i, 2])),
                                m=m)
            objects_py.append(tri)
    lookfrom = ti.Vector([-1.0, 5.0, 10.0])
    lookat = ti.Vector([0.0, 0.0, 0.0])
    fov = 90
    return objects_py, lookfrom, lookat, fov


def stanford_bunny(expand=1.0,
                   shift=np.array([0.0, 0.0, 0.0]).reshape([1, 1, 3]),
                   m=material.init()):
    scene = pywavefront.Wavefront('./data/bunny.obj')
    objects_py = []
    for name, material in scene.materials.items(): # V3F
        vertex_info = np.array(material.vertices).reshape([-1, 3, 3]) * expand + shift
        for i in range(vertex_info.shape[0]):
            tri = triangle.init(v0=ti.Vector(list(vertex_info[i, 0])),
                                v1=ti.Vector(list(vertex_info[i, 1])),
                                v2=ti.Vector(list(vertex_info[i, 2])),
                                m=m)
            objects_py.append(tri)
    lookfrom = ti.Vector([-1.0, 5.0, 10.0])
    lookat = ti.Vector([0.0, 0.0, 0.0])
    fov = 90
    return objects_py, lookfrom, lookat, fov


def bunny_cow_with_random_spheres():
    spheres = list()
    # Ground
    ground_material = Material(m_type=1, albedo=ti.Vector([0.5, 0.5, 0.5]), fuzz=0.0, ir=0.0, t_type=0)
    spheres.append(Sphere(center=ti.Vector([0, -1000, 0]), radius=1000, m=ground_material))
    # Random balls
    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random.random()
            center = ti.Vector([a + 0.9 * random.random(), 0.2, b + 0.9 * random.random()])
            if (center - ti.Vector([4, 0.2, 0])).norm() > 0.9:
                if choose_mat < 0.8:
                    albedo = utils.rand3(0.0, 1.0)
                    m = Material(m_type=1, albedo=albedo, fuzz=0.0, ir=0.0, t_type=0)
                    spheres.append(Sphere(center=center, radius=0.2, m=m))
                elif choose_mat < 0.95:
                    albedo = utils.rand3(0.5, 1.0)
                    fuzz = random.uniform(0.0, 0.5)
                    m = Material(m_type=2, albedo=albedo, fuzz=fuzz, ir=0.0, t_type=0)
                    spheres.append(Sphere(center=center, radius=0.2, m=m))
                else:
                    albedo = ti.Vector([1.0, 1.0, 1.0])
                    m = Material(m_type=3, albedo=albedo, fuzz=0.0, ir=1.5, t_type=0)
                    spheres.append(Sphere(center=center, radius=0.2, m=m))
    m1 = Material(m_type=3, albedo=ti.Vector([1.0, 1.0, 1.0]), fuzz=0.0, ir=1.5, t_type=0)
    spheres.append(Sphere(center=ti.Vector([2.0, 1.0, 0.0]), radius=1.0, m=m1))

    triangles = list()
    # Stanford bunny
    m3 = Material(m_type=2, albedo=ti.Vector([0.7, 0.6, 0.5]), fuzz=0.0, ir=0.0, t_type=0)
    bunny, _, _, _ = stanford_bunny(expand=16.04,
                                    shift=np.array([4.01680081e+00, 8.89847040e-01 - 1.5, -1.0]), m=m3)
    triangles.extend(bunny)
    # Cow from GAMES101
    m4 = Material(m_type=1, albedo=ti.Vector([0.7, 0.6, 0.5]), fuzz=0.0, ir=0.0, t_type=3)
    cow, _, _, _ = games101_cow(expand=1.5, rot=240, m=m4,
                                shift=np.array([3.0, 1.0, 2.0]).reshape([1, 1, 3]))                          
    triangles.extend(cow)
    lookfrom = ti.Vector([13.0, 2.0, 3.0])
    lookat = ti.Vector([0, 0, 0])
    fov = 45
    return spheres, triangles, lookfrom, lookat, fov


if __name__ == "__main__":
    image_width = 256
    image_height = 256

    spheres, triangles, lookfrom, lookat, fov = bunny_cow_with_random_spheres()
    scene = Scene(spheres, triangles, image_width, image_height)

    camera = Camera(lookfrom=lookfrom, lookat=lookat, fov=fov, aspect_ratio=1.0)
    r = Renderer(image_width, image_height, scene, camera, spp=16)
    gui = ti.GUI("Ray Tracing", res=(image_width, image_height), show_gui=False)

    for i in tqdm(range(100)):
        r.render()
        gui.set_image(r.get_canvas_numpy())
    gui.show('./image.png')
