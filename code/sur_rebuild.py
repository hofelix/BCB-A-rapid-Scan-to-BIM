import open3d as o3d
import math
import numpy as np
import time
import random

THETA_THRESHOLD = 5
D2 = 0.15


def main():
    pcd = o3d.io.read_point_cloud('00.pcd')            #input pcd
    #pcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=5))
    v_list = np.asarray(pcd.normals)
    lists = np.load('list.npy')
    #lists = region_growing(pcd, v_list)
    #nlist = np.asarray(lists)
    #np.save('list', nlist)
    lists,groupv,group,unique = major_and_outlier(lists,pcd)
    center_pcd = build_center(group,groupv)
    pair_list = parallel_check(center_pcd,unique)
    mush_list = rebuild(pcd,lists,pair_list,unique)
    #pcd = color_pcd(pcd, lists)
    o3d.visualization.draw_geometries(mush_list,point_show_normal=False)


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


def rebuild(pcd,lists,pair_list,unique):
    mush_list = []
    for i in pair_list:
        local1 = np.where(lists == unique[i[0]])
        local2 = np.where(lists == unique[i[1]])
        point1 = np.asarray(pcd.points)[local1,:]
        point2 = np.asarray(pcd.points)[local2,:]
        xyz = np.append(point1,point2,axis=1)[0,:,:]
        p = vertex_to_mesh(xyz)
        mush_list.append(p.hull_ls)
    return mush_list


def parallel_check(pcd,unique):
    num = np.size(unique)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    check = np.full((num), False, dtype=bool)
    pair_list = []
    for i in range(num):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 5)
        idx.pop(0)
        for j in idx:
            check_parallel = parallel(pcd.normals[i],pcd.normals[j])
            if check_parallel==True:
                check[i] = True
                check[j] = True
                pair = np.array([i,j])
                pair_list.append(pair)
    return pair_list


def build_center(group,groupv):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(group)
    pcd.normals = o3d.utility.Vector3dVector(groupv)
    return pcd


def region_growing(pcd, v_list):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    pcd_size = np.shape(pcd.points)[0]
    seedMark = np.zeros((pcd_size), dtype=int)
    for i in range(pcd_size):
        if seedMark[i] == 0:
            seedMark[i] = i+1
            seedList = []
            seedList.append(i)
            [num, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], D2)
            for j in idx:
                theta = parallel(v_list[i,:],v_list[j,:])
                if seedMark[j] == 0:
                    if theta == True:
                        seedList.append(j)
                        seedMark[j] = seedMark[i]
            seedList.pop(0)
            while (len(seedList)>0):
                j = seedList.pop(0)
                [num, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[j], D2)
                for k in idx:
                    theta = parallel(v_list[j, :], v_list[k, :])
                    if seedMark[k] == 0:
                        if theta == True:
                            seedList.append(k)
                            seedMark[k] = seedMark[i]
    return seedMark


def dis_p_to_s(s, p):
    area = np.linalg.norm(s[0:3])
    d = s[0]*p[0]+s[1]*p[1]+s[2]*p[2]+s[3]
    return abs(d)/area


def parallel(a,b):
    na = a/np.linalg.norm(a)
    nb = b/np.linalg.norm(b)
    if 1.0 - abs(np.dot(na, nb)) < 1e-6:
        return True
    else:
        return False


def color_pcd(pcd, lists):
    unique = np.unique(lists)
    for i in unique:
        unique_location = np.where(lists==i)[0]
        np.asarray(pcd.colors)[unique_location[0:], :] = [
                   random.random(), random.random(), random.random()]
    unique_location = np.where(lists==0)[0]
    np.asarray(pcd.points)[unique_location[0:], :] = [0, 0, 0]
    return pcd


def major_and_outlier(lists,pcd):
    unique,counts = np.unique(lists,return_counts=True)
    outlier_define = 10
    counts_location = [j for j in range(len(counts)) if counts[j]<=outlier_define]
    unique_location = unique[counts_location]
    for i in unique_location:
        lists_location = np.where(lists==i)[0]
        lists[lists_location] = 0
    unique,index,counts = np.unique(lists, return_index=True,return_counts=True)
    zero_local = np.where(unique==0)
    index = np.delete(index, zero_local)
    unique = np.delete(unique, zero_local)
    num = np.size(unique)
    groupv = np.zeros((num,3))
    group = np.zeros((num,3))
    for i in range(num):
        groupv[i] = np.asarray(pcd.normals)[index[i],:]
        g = np.asarray(pcd.points)[np.where(lists==unique[i]),:]
        group[i] = g.mean(axis=1)
    return lists,groupv,group,unique


if __name__=='__main__':
    main()
