import open3d as o3d
import math
import numpy as np
import time
import random
import os

THETA_THRESHOLD = 5
D2 = 0.15
PATH = os.path.abspath('D:\Program\RANSAC_EDGE\output')

def main():
    pcd = o3d.io.read_point_cloud('21.ply')            #input pcd
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    print('input finish')
    #o3d.io.write_point_cloud("1.ply", pcd)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=5))
    v_list = np.asarray(pcd.normals)
    lists = region_growing(pcd, v_list)
    lists,groupv,group,unique = major_and_outlier(lists,pcd)
    center_pcd = build_center(group,groupv)
    pair_list = parallel_check(center_pcd,unique)
    pair_list = element_check(pcd,pair_list,group)
    print(len(pair_list))
    mesh_list = rebuild(pcd,lists,pair_list,unique)
    #save_ply(mesh_list)
    #pcd = color_pcd(pcd, lists)
    o3d.visualization.draw_geometries(mesh_list,point_show_normal=True)


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


def save_ply(mesh_list):
    i = 0
    for mesh in mesh_list:
        i = i+1
        filename = str(i)+'.ply'
        filepath = os.path.join(PATH,filename)
        o3d.io.write_triangle_mesh(filepath, mesh)


def wall_element(pair_list,unique):
    pass



def rebuild(pcd,lists,pair_list,unique):
    mush_list = []
    for i in pair_list:
        point = np.array([[0,0,0]])
        for j in i:
            local = np.where(lists == unique[j])
            new_point = np.asarray(pcd.points)[local]
            point = np.append(point,new_point,axis=0)
        point = np.delete(point, 0, axis=0)
        p = vertex_to_mesh(point)
        mush_list.append(p.hull)
        #mush_list.append(p.hull_ls)
    return mush_list


def element_check(pcd,pair_list,group):
    num = len(pair_list)
    cc_lists = []
    new_pair_list = []
    check = np.full((num), False, dtype=bool)
    for i in pair_list:
        cc_lists.append((group[i[0],:]+group[i[1],:])/2)
    cc_lists = np.asarray(cc_lists)
    for i in range(num):
        new_pair = pair_list[i]
        for j in range(i+1,num):
            if check[j] == False:
                dis = dis_p_to_p(cc_lists[i], cc_lists[j])
                if dis <= 0.1:
                    pair = pair_list[j]
                    new_pair = np.append(new_pair,pair)
                    check[j] == True
        new_pair_list.append(new_pair)
    return new_pair_list


def parallel_check(pcd,unique):
    num = np.size(unique)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    check = np.full((num), False, dtype=bool)
    pair_list = []
    for i in range(num):
        if check[i] == False:
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 5)
            idx.pop(0)
            for j in idx:
                if check[j] == False:
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


def dis_p_to_p(p1, p2):
    return np.linalg.norm(p2-p1)


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
