import numpy as np
import open3d as o3d


class make_edge:
    def __init__(self, len, delta, o, v):
        self.len = len
        self.delta = delta
        self.o = o
        self.v = v

    def create(self):
        num = int(self.len/self.delta + 1)
        self.edge = np.empty((num, 3))
        for i in range(num):
            self.edge[i, :] = self.o + i * self.v * self.delta


class rectangle:
    def __init__(self, L, D, H, o):
        self.L = L
        self.D = D
        self.H = H
        self.o = o
        self.o_list = np.zeros((12, 3))
        self.v_list = np.zeros((12, 3))
        self.l_list = np.zeros((12))

    def make(self):
        self.o_list[0, :] = self.o
        self.o_list[1, :] = [0, self.L, 0] + self.o
        self.o_list[2, :] = [self.D, self.L, 0] + self.o
        self.o_list[3, :] = [self.D, 0, 0] + self.o
        self.o_list[4, :] = [0, self.L, 0] + self.o
        self.o_list[5, :] = [self.D, self.L, 0] + self.o
        self.o_list[6, :] = [self.D, 0, 0] + self.o
        self.o_list[7, :] = self.o
        self.o_list[8, :] = [0, 0, self.H] + self.o
        self.o_list[9, :] = [0, self.L, self.H] + self.o
        self.o_list[10, :] = [self.D, self.L, self.H] + self.o
        self.o_list[11, :] = [self.D, 0, self.H] + self.o
        self.v_list[0, :] = [0, 1, 0]
        self.v_list[1, :] = [1, 0, 0]
        self.v_list[2, :] = [0, -1, 0]
        self.v_list[3, :] = [-1, 0, 0]
        self.v_list[4, :] = [0, 0, 1]
        self.v_list[5, :] = [0, 0, 1]
        self.v_list[6, :] = [0, 0, 1]
        self.v_list[7, :] = [0, 0, 1]
        self.v_list[8, :] = [0, 1, 0]
        self.v_list[9, :] = [1, 0, 0]
        self.v_list[10, :] = [0, -1, 0]
        self.v_list[11, :] = [-1, 0, 0]
        self.l_list[0] = self.L
        self.l_list[1] = self.D
        self.l_list[2] = self.L
        self.l_list[3] = self.D
        self.l_list[4] = self.H
        self.l_list[5] = self.H
        self.l_list[6] = self.H
        self.l_list[7] = self.H
        self.l_list[8] = self.L
        self.l_list[9] = self.D
        self.l_list[10] = self.L
        self.l_list[11] = self.D


def main():
    L = 1
    D = 1
    H = 2
    delta = 0.05
    o = np.array([0, 0, 0])
    t = rectangle(L, D, H, o)
    t.make()
    point = np.array([o])
    for i in range(12):
        e = make_edge(t.l_list[i], delta, t.o_list[i, :], t.v_list[i, :])
        e.create()
        point = np.append(point, e.edge, axis=0)
    print(point)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("rectangle2.pcd", pcd)


if __name__ == '__main__':
    main()
