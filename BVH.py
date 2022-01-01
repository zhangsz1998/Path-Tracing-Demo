import taichi as ti
import bounds3
from bounds3 import Bounds3
import sphere
from sphere import Sphere
import triangle
from triangle import Triangle
import material
from utils import FMAX
from tqdm import tqdm
import time


BVHNode = ti.types.struct(
    bounds=Bounds3,
    left=ti.i32,
    right=ti.i32,
    obj_type=ti.i32,
    obj_idx=ti.i32,
    area=ti.f32
)


def init():
    return BVHNode(bounds=bounds3.init(), left=-1, right=-1, obj_type=-1, obj_idx=-1, area=-1)


def get_obj_bounds_func(obj_type):
    if obj_type == 0:
        return sphere.get_bounds
    elif obj_type == 1:
        return triangle.get_bounds
    else:
        raise ValueError("Invalid object type {}".format(str(obj_type)))


def get_obj_area_func(obj_type):
    if obj_type == 0:
        return sphere.get_area
    elif obj_type == 1:
        return triangle.get_area
    else:
        raise ValueError("Invalid object type {}".format(str(obj_type)))


@ti.data_oriented
class BVHAccel:

    def __init__(self, spheres_py, triangles_py, max_prims=1, stack_width=512, stack_height=512, stack_depth=20):
        self.stack_depth = stack_depth
        self.stack = ti.field(ti.i32, shape=(stack_width, stack_height, stack_depth))
        self.stack_pt = ti.field(ti.i32, shape=(stack_width, stack_height))

        self.max_prims = max_prims
        self.nodes_py = list()
        print("Building BVH...")
        tic = time.time()
        self.objects_py = spheres_py + triangles_py
        self.obj_types = [0] * len(spheres_py) + [1] * len(triangles_py) # 0-sphere, 1-triangle
        self.type_begin = [0, len(spheres_py)]
        self.recursive_build_py(range(len(self.objects_py)))
        toc = time.time()
        print("Done, time elapsed: {:.3f}".format(toc - tic))

        # move BVHNodePy to taichi scope
        self.n_nodes = len(self.nodes_py)
        print("Number of Spheres: ", len(spheres_py))
        print("Number of Triangles: ", len(triangles_py))
        print("Number of BVH Nodes: ", self.n_nodes)
        print("Moving BVH to Taichi fields...")
        self.nodes = BVHNode.field()
        ti.root.dense(ti.i, self.n_nodes).place(self.nodes)
        self.spheres = Sphere.field()
        ti.root.dense(ti.i, max(len(spheres_py), 1)).place(self.spheres)
        self.triangles = Triangle.field()
        ti.root.dense(ti.i, max(len(triangles_py), 1)).place(self.triangles)
        self.move_to_taichi_field()
        del self.objects_py
        del self.obj_types
        del self.type_begin
        del self.nodes_py

        print("Done.")

    def recursive_build_py(self, indices):
        node = init()
        node.idx = len(self.nodes_py)
        self.nodes_py.append(node)
        bounds = bounds3.init()
        for i in indices:
            bounds = bounds3.bounds_union(bounds, get_obj_bounds_func(self.obj_types[i])(self.objects_py[i]))
        if len(indices) == 1:
            node.bounds = bounds
            node.obj_type = self.obj_types[indices[0]]
            node.obj_idx = indices[0] - self.type_begin[node.obj_type]
            node.area = get_obj_area_func(node.obj_type)(self.objects_py[indices[0]])
            node.is_leaf = True
        elif len(indices) == 2:
            node.bounds = bounds
            node.left = self.recursive_build_py([indices[0]])
            node.right = self.recursive_build_py([indices[1]])
            node.area = self.nodes_py[node.left].area + self.nodes_py[node.right].area
        else:
            centroid_bounds = bounds3.init()
            for i in indices:
                centroid_bounds = bounds3.bounds_vec_union(
                    centroid_bounds,
                    bounds3.get_centroid(get_obj_bounds_func(self.obj_types[i])(self.objects_py[i]))
                )
            dim = bounds3.max_extent(centroid_bounds)
            indices = sorted(
                indices,
                key=lambda x: bounds3.get_centroid(get_obj_bounds_func(self.obj_types[x])(self.objects_py[x]))[dim]
            )
            mid = len(indices) // 2
            node.left = self.recursive_build_py(indices[:mid])
            node.right = self.recursive_build_py(indices[mid:])
            node.bounds = bounds
            node.area = self.nodes_py[node.left].area + self.nodes_py[node.right].area
        return node.idx

    def move_to_taichi_field(self):
        for i in tqdm(range(len(self.objects_py))):
            t = self.obj_types[i]
            idx = i - self.type_begin[t]
            if t == 0:
                self.spheres[idx] = self.objects_py[i]
            elif t == 1:
                self.triangles[idx] = self.objects_py[i]
            else:
                raise ValueError("Invalid object type {}".format(str(t)))

        for i in tqdm(range(self.n_nodes)):
            self.nodes[i] = self.nodes_py[i]

    @ti.func
    def stack_push(self, i, j, node_idx):
        self.stack_pt[i, j] += 1
        self.stack[i, j, self.stack_pt[i, j]] = node_idx

    @ti.func
    def stack_pop(self, i, j):
        res = self.stack[i, j, self.stack_pt[i, j]]
        self.stack_pt[i, j] -= 1
        return res

    @ti.func
    def stack_top(self, i, j):
        return self.stack[i, j, self.stack_pt[i, j]]

    @ti.func
    def stack_empty(self, i, j):
        return self.stack_pt[i, j] < 0

    @ti.func
    def node_intersetc_p(self, node, ray, t_min=1e-3, t_max=1e9):
        is_hit = False
        t = FMAX
        coords = ti.Vector([0.0, 0.0, 0.0])
        normal = ti.Vector([0.0, 0.0, 0.0])
        face_front = True
        u = 0.0
        v = 0.0
        m = material.init()
        if node.obj_type == 0:
            obj = self.spheres[node.obj_idx]
            is_hit, t, coords, normal, face_front, m = sphere.intersect_p(obj, ray, t_min, t_max)
            u, v = sphere.get_uv(obj, coords)
        elif node.obj_type == 1:
            obj = self.triangles[node.obj_idx]
            is_hit, t, coords, normal, face_front, m = triangle.intersect_p(obj, ray, t_min, t_max)
            u, v = triangle.get_uv(obj, coords)
        return is_hit, t, coords, normal, face_front, u, v, m

    @ti.func
    def intersect_p(self, ray, si, sj, t_min=1e-3, t_max=1e9):
        is_hit = False
        t = FMAX
        coords = ti.Vector([0.0, 0.0, 0.0])
        u = 0.0
        v = 0.0
        normal = ti.Vector([0.0, 0.0, 0.0])
        face_front = True
        m = material.init()
        self.stack_pt[si, sj] = -1
        # iterative pre-order traversal
        node_idx = 0
        while node_idx >= 0 or not self.stack_empty(si, sj):
            while node_idx >= 0:
                node = self.nodes[node_idx]
                if bounds3.bounds_intersect(node.bounds, ray):
                    if node.obj_type != -1:
                        is_hit_tmp, t_tmp, coords_tmp, normal_tmp, f_tmp, u_tmp, v_tmp, m_tmp = \
                            self.node_intersetc_p(node, ray, t_min, t)
                        if is_hit_tmp:
                            is_hit = True
                            t = t_tmp
                            coords = coords_tmp
                            normal = normal_tmp
                            face_front = f_tmp
                            u = u_tmp
                            v = v_tmp
                            m = m_tmp
                        break
                    else:
                        self.stack_push(si, sj, node_idx)
                        node_idx = node.left
                else:
                    break
            if not self.stack_empty(si, sj):
                node_idx = self.stack_pop(si, sj)
                node_idx = self.nodes[node_idx].right
            else:
                break
        return is_hit, t, coords, u, v, normal, face_front, m
