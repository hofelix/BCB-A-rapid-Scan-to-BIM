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
    s = time.time()
    lists = major_and_outlier(lists)
    e = time.time()
    print(e-s)
    pcd = color_pcd(pcd, lists)
    o3d.visualization.draw_geometries([pcd],point_show_normal=False)


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


def major_and_outlier(lists):
    unique,counts = np.unique(lists,return_counts=True)
    outlier_define = 10
    counts_location = [j for j in range(len(counts)) if counts[j]<=outlier_define]
    unique_location = unique[counts_location]
    for i in unique_location:
        lists_location = np.where(lists==i)[0]
        lists[lists_location] = 0
    return lists
    unique,counts = np.unique(lists,return_counts=True)
    num = np.size(counts)
    group = np.zero((num,3))
    for i in range(num):
        location = np.where(lists==unique[i])[0]
        inlier = pcd.points[location,:]


if __name__=='__main__':
    main()
