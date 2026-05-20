import numpy as np
import open3d as o3d
import math
import copy


THETA_THRESHOLD = 5
DIS_THRESHOLD = 1


def main():
    lists = np.loadtxt('lists_best.txt')
    pcd = o3d.io.read_point_cloud('3.ply')
    vector_list, vertex_list, edge_num = make_index(lists, pcd)
    check_list = np.full((edge_num), False, dtype=bool)
    center_list = np.empty((1, 3))
    hull_list = []
    for i in range(edge_num):
        parallel_list = []
        parallel_list.append(i)
        if check_list[i] == True:
            continue
        for j in range(i+1, edge_num):
            theta = angle(vector_list[i, :], vector_list[j, :])
            dis = dis_p_to_v(vertex_list[i, :],
                             vector_list[i, :], vertex_list[j, :])
            if theta <= THETA_THRESHOLD and dis <= DIS_THRESHOLD and check_list[j] == False:
                parallel_list.append(j)
        inlier = np.shape(parallel_list)[0]
        print('element with ',inlier,' edge')
        if inlier > 1:
            parallel_list, inlier = coplanarity_check(parallel_list, vertex_list, vector_list, edge_num, inlier)
        if inlier == 3:
            c, h = trilateral_reconstruction(
                parallel_list, vertex_list, vector_list, edge_num)
            center_list = np.append(center_list, c, axis=0)
            check_list[parallel_list] = True
            hull_list.append(h)
        elif inlier == 4:
            c, h ,parallel_list = quadrilateral_reconstruction(
                parallel_list, vertex_list, vector_list, edge_num)
            center_list = np.append(center_list, c, axis=0)
            check_list[parallel_list] = True
            hull_list.append(h)
        elif inlier > 4:
            c, h, parallel_list = polygon_reconstruction(parallel_list, vertex_list, vector_list, edge_num)
            center_list = np.append(center_list, c, axis=0)
            check_list[parallel_list] = True
            hull_list.append(h)
    hull_list = check_repeat(center_list, hull_list)
    o3d.visualization.draw_geometries(hull_list)


class edge:
    def __init__(self, points):
        self.index = points
        self.point_num = np.shape(self.index)[0]
        self.middle = np.mean(self.index, axis=0)
        uu, dd, vv = np.linalg.svd(self.index - self.middle)
        self.v = vv[0]

    def limit(self):
        dis_to_mid = np.zeros((self.point_num))
        for i in range(self.point_num):
            dis_to_mid[i] = np.linalg.norm(self.index[i, :]-self.middle)
        d1 = self.index[np.argmax(dis_to_mid), :]
        dis_to_d1 = np.zeros((self.point_num))
        for i in range(self.point_num):
            dis_to_d1[i] = np.linalg.norm(self.index[i, :]-d1)
        d2 = self.index[np.argmax(dis_to_d1), :]
        o = np.array([-10000, -10000, -10000])
        dis = [np.linalg.norm(d1-o), np.linalg.norm(d2-o)]
        if dis[0] > dis[1]:
            self.top = d1
            self.bottom = d2
        else:
            self.top = d2
            self.bottom = d1

        self.newtop = np.dot(self.top-self.middle, self.v)*self.v + self.middle
        self.newbottom = np.dot(self.bottom-self.middle,
                                self.v)*self.v + self.middle
        self.v = unit_vector(self.newtop-self.newbottom)


class vertex_to_mesh:
    def __init__(self, vertex):
        self.vertex = vertex
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertex)
        self.pcd = pcd
        self.hull, _ = pcd.compute_convex_hull()
        self.hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(
            self.hull)
        self.center = np.array([np.mean(self.vertex, axis=0)])


class tri:
    def __init__(self, vertex):
        self.vertex = vertex
        self.dis_list = np.zeros((3))
        self.dis_list[0] = dis_p_to_p(
            vertex[0, :], vertex[1, :]) + dis_p_to_p(vertex[0, :], vertex[2, :])
        self.dis_list[1] = dis_p_to_p(
            vertex[1, :], vertex[0, :]) + dis_p_to_p(vertex[1, :], vertex[2, :])
        self.dis_list[2] = dis_p_to_p(
            vertex[2, :], vertex[0, :]) + dis_p_to_p(vertex[2, :], vertex[1, :])
        self.mid = np.array(
            np.where(self.dis_list == np.min(self.dis_list))[0])[0]

    def rebuild(self):
        if self.mid == 0:
            self.v = self.vertex[1, :] + self.vertex[2, :] - self.vertex[0, :]
        if self.mid == 1:
            self.v = self.vertex[0, :] + self.vertex[2, :] - self.vertex[1, :]
        if self.mid == 2:
            self.v = self.vertex[0, :] + self.vertex[1, :] - self.vertex[2, :]


class surface:
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        AC = self.C - self.A
        AB = self.B - self.A
        v = np.cross(AC, AB)
        d = -(v[0]*self.A[0]+v[1]*self.A[1]+v[2]*self.A[2])
        self.s = np.array([v[0], v[1], v[2], d])


class circle(surface):
    def __init__(self, A, B, C):
        super().__init__(A, B, C)
        [x1, y1, z1] = self.A
        [x2, y2, z2] = self.B
        [x3, y3, z3] = self.C
        a1 = self.s[0]
        b1 = self.s[1]
        c1 = self.s[2]
        d1 = self.s[3]
        a2 = 2 * (x2 - x1)
        b2 = 2 * (y2 - y1)
        c2 = 2 * (z2 - z1)
        d2 = x1*x1 + y1*y1 + z1*z1 - x2*x2 - y2*y2 - z2*z2
        a3 = 2 * (x3 - x1)
        b3 = 2 * (y3 - y1)
        c3 = 2 * (z3 - z1)
        d3 = x1*x1 + y1*y1 + z1*z1 - x3*x3 - y3*y3 - z3*z3
        x = -(b1*c2*d3 - b1*c3*d2 - b2*c1*d3 + b2*c3*d1 + b3*c1*d2 - b3*c2*d1) / \
            (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)
        y = (a1*c2*d3 - a1*c3*d2 - a2*c1*d3 + a2*c3*d1 + a3*c1*d2 - a3*c2*d1) / \
            (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)
        z = -(a1*b2*d3 - a1*b3*d2 - a2*b1*d3 + a2*b3*d1 + a3*b1*d2 - a3*b2*d1) / \
            (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)
        self.center = [x, y, z]
        self.surface = self.s
        self.radius = math.sqrt((x1-x)*(x1-x)+(y1-y)*(y1-y)+(z1-z)*(z1-z))


def outlier_edge_check(xyz,full_point):
    return full_point


def surface_area(p1,p2,p3):
    a = dis_p_to_p(p1, p2)
    b = dis_p_to_p(p1, p3)
    c = dis_p_to_p(p2, p3)
    s = (a + b + c) / 2
    area = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    return area


def section_check(vertex):
    a1 = surface_area(vertex[0,:],vertex[1,:],vertex[2,:])
    a2 = surface_area(vertex[4,:],vertex[5,:],vertex[6,:])
    if 0.5*a1>a2:
        h = vertex[4,:] - vertex[0,:]
        vertex[5,:] = vertex[1,:] + h
        vertex[6,:] = vertex[2,:] + h
        vertex[7,:] = vertex[3,:] + h
        print('section too tiny')
    elif 0.5*a2>a1:
        h = vertex[0,:] - vertex[4,:]
        vertex[1,:] = vertex[5,:] + h
        vertex[2,:] = vertex[6,:] + h
        vertex[3,:] = vertex[7,:] + h
        print('section too tiny')
    return vertex


def fake_tri_check(point,all_point,parallel_list):
    num = np.shape(point)[0]
    dis_list = np.zeros((num))
    for i in range(num):
        check = np.zeros((num))
        for j in range(num):
            check[j] = dis_p_to_p(point[i,:], point[j,:])
        check = np.setdiff1d(check, 0)
        dis_list[i] = np.min(check)
    minist = np.min(dis_list)
    if minist <= 0.05:
        m1 = np.where(dis_list == np.min(dis_list))[0][0]
        m2 = np.where(dis_list == np.min(dis_list))[0][1]
        lists = np.linspace(0, num, num, endpoint=False,dtype = int)
        plists = np.setdiff1d(lists, [m1,m2])
        d11 = dis_p_to_p(all_point[m1], all_point[plists[0]])
        d12 = dis_p_to_p(all_point[m1+num], all_point[plists[0]+num])
        d21 = dis_p_to_p(all_point[m2], all_point[plists[0]])
        d22 = dis_p_to_p(all_point[m2+num], all_point[plists[0]+num])
        d1 = abs(d11-d12)
        d2 = abs(d21-d22)
        if d1 >= d2:
            m = m1
        elif d2 > d1:
            m = m2
        lists = np.setdiff1d(lists, m)
        n = np.shape(lists)[0]
        if n == 3:
            print('is fake triangle only 3 edge ')
            p = all_point[lists,:]
            parallel_list = parallel_list[lists]
            t = tri(p)
            t.rebuild()
            all_point[m,:] = t.v
            p = all_point[lists+num,:]
            t = tri(p)
            t.rebuild()
            all_point[m+num,:] = t.v
        elif n == 4 :
            print('is fake triangle only 4 edge ')
            parallel_list = parallel_list[lists]
            lists = np.append(lists,lists+num)
            all_point = all_point[lists,:]
        elif n > 4 :
            print('is fake triangle move ',n,' edge ')
            parallel_list = parallel_list[lists]
            lists = np.append(lists,lists+num)
            all_point = all_point[lists,:]
    return all_point,parallel_list


def coplanarity_check(parallel_list, vertex_list, vector_list, edge_num, inlier):
    num = inlier
    xyz = np.empty((num*2, 3))
    v_list = np.zeros((num, 3))
    num_list = np.linspace(0, num, num, endpoint=False,dtype = int)
    for i in range(num):
        xyz[i, :] = vertex_list[parallel_list[i], :]
        xyz[i+num, :] = vertex_list[parallel_list[i]+edge_num, :]
        v_list[i, :] = vector_list[parallel_list[i], :]
    for i in range(num):
        for j in range(i+1,num):
            check = copy.deepcopy(num_list)
            check = np.setdiff1d(check, [i,j])
            s = surface(xyz[i,:],xyz[j,:],xyz[i+num,:])
            for k in check:
                dis = dis_p_to_s(s.s, xyz[int(k),:])
                if dis < 0.01:
                    print(' is coplanarity delete 1 edge')
                    mid = coplanarity_remove(xyz,v_list,i,j,k)
                    num_list = np.setdiff1d(num_list, int(mid))
    parallel_list = np.array(parallel_list)
    parallel_list = parallel_list[num_list]
    return parallel_list,np.shape(parallel_list)[0]


def coplanarity_remove(xyz,v_list,A,B,C):
    lists = [int(A), int(B), int(C)]
    xyz = xyz[lists,:]
    v_list = v_list[lists,:]
    dis = np.zeros((3))
    dis[0] = dis_p_to_v(xyz[0], v_list[0], xyz[1])+dis_p_to_v(xyz[0], v_list[0], xyz[2])
    dis[1] = dis_p_to_v(xyz[1], v_list[1], xyz[0])+dis_p_to_v(xyz[1], v_list[1], xyz[2])
    dis[2] = dis_p_to_v(xyz[2], v_list[2], xyz[0])+dis_p_to_v(xyz[2], v_list[2], xyz[1])
    mid = np.where(dis == np.min(dis))[0][0]
    return lists[mid]


def polygon_reconstruction(parallel_list, vertex_list, vector_list, edge_num):
    print('polygon reconstruction')
    num = np.shape(parallel_list)[0]
    xyz = np.empty((num*2, 3))
    v_list = np.zeros((num, 3))
    for i in range(num):
        xyz[i, :] = vertex_list[parallel_list[i], :]
        xyz[i+num, :] = vertex_list[parallel_list[i]+edge_num, :]
        v_list[i, :] = vector_list[parallel_list[i], :]
    xyz = boundary_repair(xyz, v_list,num)
    xyz = boundary_repair(xyz, v_list,num)
    lists = np.linspace(0, num, num, endpoint=False,dtype = int)
    xyz,parallel_list = fake_tri_check(xyz[lists,:],xyz,parallel_list)
    num = int(0.5*(np.shape(xyz)[0]))
    lists = np.linspace(0, num, num, endpoint=False,dtype = int)
    xyz,parallel_list = fake_tri_check(xyz[lists+num,:],xyz,parallel_list)
    num = int(0.5*(np.shape(xyz)[0]))
    lists = np.linspace(0, num, num, endpoint=False,dtype = int)
    if num > 4:
        xyz = outlier_edge_check(xyz[lists,:],xyz)
    xyz = section_check(xyz)
    p = vertex_to_mesh(xyz)
    return p.center, p.hull_ls,parallel_list



def quadrilateral_reconstruction(parallel_list, vertex_list, vector_list, edge_num):
    print('quadrilateral reconstruction')
    xyz = np.empty((8, 3))
    v_list = np.zeros((4, 3))
    for i in range(4):
        xyz[i, :] = vertex_list[parallel_list[i], :]
        xyz[i+4, :] = vertex_list[parallel_list[i]+edge_num, :]
        v_list[i, :] = vector_list[parallel_list[i], :]
    xyz = boundary_repair(xyz, v_list,4)
    xyz = boundary_repair(xyz, v_list,4)
    xyz,parallel_list = fake_tri_check(xyz[[0,1,2,3],:],xyz,parallel_list)
    xyz,parallel_list = fake_tri_check(xyz[[4,5,6,7],:],xyz,parallel_list)
    xyz = section_check(xyz)
    p = vertex_to_mesh(xyz)
    return p.center, p.hull_ls,parallel_list


def trilateral_reconstruction(parallel_list, vertex_list, vector_list, edge_num):
    print('trilateral reconstruction')
    xyz = np.zeros((8, 3))
    v_list = np.zeros((4, 3))
    for i in range(3):
        xyz[i, :] = vertex_list[parallel_list[i], :]
        xyz[i+4, :] = vertex_list[parallel_list[i]+edge_num, :]
        v_list[i, :] = vector_list[parallel_list[i], :]
    v_list[3, :] = v_list[1, :]
    nxyz = xyz[[0,1,2,4,5,6],:]
    nxyz = boundary_repair(nxyz, v_list,3)
    nxyz = boundary_repair(nxyz, v_list,3)
    xyz[[0,1,2,4,5,6],:] = nxyz
    t = tri(xyz[[0,1,2],:])
    t.rebuild()
    xyz[3, :] = t.v
    t = tri(xyz[[4,5,6],:])
    t.rebuild()
    xyz[7, :] = t.v
    xyz = boundary_repair(xyz, v_list,4)
    xyz = boundary_repair(xyz, v_list,4)
    xyz = section_check(xyz)
    p = vertex_to_mesh(xyz)
    return p.center, p.hull_ls


def boundary_repair(vertex_list, vector_list,num):
    print('boundary repair')
    center = vertex_list.mean(axis=0)
    new_vertex_list = np.zeros((num*2, 3))
    dis_list = np.zeros((num))
    for i in range(num):
        dis_list[i] = dis_ve_to_c(vertex_list[i, :], vector_list[i, :], center)
    l = np.max(dis_list)
    for i in range(num):
        c_of_ver = vertex_list[i, :] - dis_list[i]*vector_list[i, :]
        new_vertex_list[i, :] = c_of_ver + l*vector_list[i, :]
    for i in range(num):
        dis_list[i] = dis_ve_to_c(
            vertex_list[i+num, :], vector_list[i, :], center)
    l = np.max(dis_list)
    for i in range(num):
        c_of_ver = vertex_list[i+num, :] + dis_list[i]*vector_list[i, :]
        new_vertex_list[i+num, :] = c_of_ver - l*vector_list[i, :]
    return new_vertex_list


def unit_vector(vector):
    return vector/np.linalg.norm(vector)


def dis_ve_to_c(A, D, P):  # vertex to center
    V = P - A
    return abs(np.dot(V, D))


def dis_p_to_p(p1, p2):
    return np.linalg.norm(p2-p1)


def dis_p_to_s(s, p):
    area = np.linalg.norm(s[0:3])
    d = s[0]*p[0]+s[1]*p[1]+s[2]*p[2]+s[3]
    return abs(d)/area


def check_repeat(center_list, hull_list):
    print('check repeat element')
    center_list = np.delete(center_list, [0], 0)
    num = np.shape(center_list)[0]
    check_list = np.full((num), False, dtype=bool)
    for i in range(num):
        for j in range(i+1, num):
            dis = np.linalg.norm(center_list[i]-center_list[j])
            if dis <= 0.1:
                check_list[j] = True
    for i in range(num-1, -1, -1):
        if check_list[i] == True:
            del hull_list[i]
            print('delete repeat element')
    return hull_list


def make_index(lists, pcd):
    unique = np.unique(lists)
    unique = np.setdiff1d(unique, 0)
    edge_num = np.shape(unique)[0]
    vector_list = np.empty((edge_num, 3))
    vertex_list = np.empty((edge_num*2, 3))
    for i in range(edge_num):
        unique_location = [j for j in range(
            len(lists)) if lists[j] == unique[i]]
        data = np.asarray(pcd.points)[unique_location[0:], :]
        e = edge(data)
        e.limit()
        vector_list[i, :] = e.v
        vertex_list[i, :] = e.newtop
        vertex_list[i+edge_num, :] = e.newbottom
    return vector_list, vertex_list, edge_num


def angle(a, b):
    aba_a = np.linalg.norm(a)
    aba_b = np.linalg.norm(b)
    if aba_a == 0 or aba_b == 0:
        cos_theta = 0
    else:
        cos_theta = abs(np.dot(a, b)/(aba_a*aba_b))
    if cos_theta > 0 and cos_theta <= 1:
        theta = math.degrees(math.acos(cos_theta))
    elif cos_theta < 0 and cos_theta >= -1:
        theta = 180 - math.degrees(math.acos(cos_theta))
    elif cos_theta > 1:
        theta = 0
    elif cos_theta < -1:
        theta = 0
    else:
        theta = 90
    return theta


def dis_p_to_v(A, D, P):
    V = P - A
    l = np.dot(V, D)
    v = np.linalg.norm(V)
    d = math.sqrt(abs(v*v-l*l))
    return d


if __name__ == '__main__':
    main()
