import open3d as o3d
import os
import math
import numpy as np

K1 = 50
r = np.pi / 12

def main():
    pcd = o3d.io.read_point_cloud('21.ply')            #input pcd
    #pcd = pcd.voxel_down_sample(voxel_size=0.05)
    R = pcd.get_rotation_matrix_from_xyz((r, r, r))
    pcd.rotate(R, center=(0, 0, 0))
    b = np.array([0.2,0.2,0.2])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)           #kd_tree
    pcd_size = np.shape(pcd.points)[0]                 #pcd size
    print(pcd_size)
    s_list = np.zeros((pcd_size,3))
    c2_list = np.zeros((pcd_size))
    for i in range(pcd_size):
        [num, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], K1)
        data = np.asarray(pcd.points)[idx[0:], :]
        s,v,c2 = PCA(data)
        s_list[i,:] = s
        c2_list[i] = c2
    inlier_list = np.array(0)
    c2_mean = c2_list.mean(axis=0)
    c2_std = c2_list.std(axis=0)
    boundary_threshold = c2_mean + 1.9*c2_std
    for i in range(pcd_size):
        if c2_list[i] >= boundary_threshold :
            inlier_list = np.append(inlier_list,i)
    inlier_list = np.delete(inlier_list,[0])
    inlier_pcd = pcd.select_by_index(inlier_list)
    R = inlier_pcd.get_rotation_matrix_from_xyz((-1*r, 0, 0))
    inlier_pcd.rotate(R, center=(0, 0, 0))
    R = inlier_pcd.get_rotation_matrix_from_xyz((0, -1*r, 0))
    inlier_pcd.rotate(R, center=(0, 0, 0))
    R = inlier_pcd.get_rotation_matrix_from_xyz((0, 0, -1*r))
    inlier_pcd.rotate(R, center=(0, 0, 0))
    inlier_pcd.paint_uniform_color(b)
    #o3d.io.write_point_cloud("4.ply", inlier_pcd)
    pcd_size = np.shape(inlier_pcd.points)[0]                 #pcd size
    print(pcd_size)
    o3d.visualization.draw_geometries([inlier_pcd])

def PCA(data):
    datamean = data.mean(axis=0)
    C = np.zeros((3,3))
    for i in range(K1):
        p = data[i,:] - datamean
        pT = p.T
        C += p*pT
    C = C/K1
    u, s, v = np.linalg.svd(C)
    si = np.dot(v[1],data[0,:] - datamean)
    ti = np.dot(v[2],data[0,:] - datamean)
    if s[1] == 0:
        c2 =  si*si/s[0] + ti*ti/s[0]
    else:
        c2 = si*si/s[1] + ti*ti/s[0]
    return s,v[0],c2

if __name__ == '__main__':
    main()
