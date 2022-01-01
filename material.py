import taichi as ti
from utils import random_unit_vector, near_zero, random_in_unit_sphere, clip
from ray import Ray

PI = 3.141592653589793
EPSILON = 1e-5

# m_type: 1-diffuse, 2-metal, 3-dielectric
# t_type: 0-no texture, 1-checker board, 2-earth, 3-cow
Material = ti.types.struct(
    m_type=ti.i32,
    albedo=ti.types.vector(3, ti.f32),
    fuzz=ti.f32,
    ir=ti.f32,
    t_type=ti.i32  # texture type, see get_texture_value function below.
)


def init(m_type=1, albedo=ti.Vector([0.0, 0.0, 0.0]), fuzz=0.0, ir=0.0, t_type=0):
    return Material(
        m_type=m_type,
        albedo=albedo,
        fuzz=fuzz,
        ir=ir,
        t_type=t_type
    )


# ti.init(arch=ti.cuda)
earth_img = ti.Vector.field(3, ti.f32, shape=(1024, 512))
earth_img.from_numpy(ti.imread('./data/earthmap.jpg') / 255.0)


spot_img = ti.Vector.field(3, ti.f32, shape=(1024, 1024))
spot_img.from_numpy(ti.imread('./data/spot/spot_texture.png') / 255.0)


@ti.func
def reflect(v, normal):
    return v - 2.0 * v.dot(normal) * normal


@ti.func
def reflectance(cos, ref_idx):
    r0 = (1.0 - ref_idx) / (1.0 + ref_idx)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * pow(1 - cos, 5)


@ti.func
def refract(uv, normal, etai_over_etat):
    cos_theta = min(-uv.dot(normal), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * normal)
    r_out_parallel = -normal * ti.sqrt(abs(1.0 - r_out_perp.dot(r_out_perp)))
    return r_out_perp + r_out_parallel


@ti.func
def scatter(ray, p, u, v, normal, front_face, m):
    is_scattered = False
    scattered = Ray(origin=ti.Vector([0.0, 0.0, 0.0]), direction=ti.Vector([0.0, 0.0, 0.0]))
    attenuation = get_texture_value(m, p, u, v)
    if m.m_type == 1:
        is_scattered = True
        scatter_direction = (normal + random_unit_vector())
        if near_zero(scatter_direction):
            scatter_direction = normal
        scatter_direction = scatter_direction.normalized()
        scattered = Ray(origin=p, direction=scatter_direction)
    elif m.m_type == 2:
        scatter_direction = (reflect(ray.direction, normal) + m.fuzz * random_in_unit_sphere()).normalized()
        if scatter_direction.dot(normal) > 0.0:
            is_scattered = True
            scattered = Ray(origin=p, direction=scatter_direction)
    elif m.m_type == 3:
        refract_ratio = m.ir
        is_scattered = True
        if front_face:
            refract_ratio = 1.0 / refract_ratio
        cos_theta = min(-ray.direction.dot(normal), 1.0)
        sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)
        cannot_refract = (refract_ratio * sin_theta) > 1.0
        scatter_direction = ti.Vector([0.0, 0.0, 0.0])
        if cannot_refract or reflectance(cos_theta, refract_ratio) > ti.random():
            scatter_direction = reflect(ray.direction, normal).normalized()
        else:
            scatter_direction = refract(ray.direction, normal, refract_ratio).normalized()
        scattered = Ray(origin=p, direction=scatter_direction)
    return is_scattered, scattered, attenuation


@ti.func
def get_texture_value(m, p, u, v):
    albedo = ti.Vector([0.0, 0.0, 0.0])
    if m.t_type == 0:
        albedo = m.albedo
    elif m.t_type == 1:
        sines = ti.sin(10 * p[0]) * ti.sin(10 * p[1]) * ti.sin(10 * p[2])
        if sines < 0.0:
            albedo = ti.Vector([0.5, 0.5, 0.5])
        else:
            albedo = ti.Vector([0.5, 0.0, 0.0])
    elif m.t_type == 2:
        i = int(u * earth_img.shape[0])
        j = int(v * earth_img.shape[1])
        if i >= earth_img.shape[0]:
            i = earth_img.shape[0] - 1
        if j >= earth_img.shape[1] - 1:
            j = earth_img.shape[1] - 1
        albedo = earth_img[i, j]
    elif m.t_type == 3:
        i = int(u * spot_img.shape[0])
        j = int(v * spot_img.shape[1])
        if i >= spot_img.shape[0]:
            i = spot_img.shape[0] - 1
        if j >= spot_img.shape[1] - 1:
            j = spot_img.shape[1] - 1
        albedo = spot_img[i, j]
    return albedo
