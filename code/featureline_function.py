import open3d as o3d
import math
import numpy as np
import random
import time

K1 = 20
D1 = 0.5
D2 = 1
RANSAC_K = 400
RANSAC_N = 3
DIS_THRESHOLD = 0.01
DIS_THRESHOLD2 = 0.02
DIS_THRESHOLD3 = 0.02
THETA_THRESHOLD = 10


def main():
    start = time.time()
    pcd = o3d.io.read_point_cloud('4.ply')
    pcd.paint_uniform_color([1, 0, 0])
    print('finish load point')
    s = time.time()
    A_list, D_list = Calculate_A_D(pcd)
    #A_list = np.load('a.npy')
    #D_list = np.load('d.npy')
    e = time.time()
    print('finish Calculate A D', e-s)
    s = time.time()
    lists = region_growing(pcd, A_list, D_list)
    e = time.time()
    print('finish region growing', e-s)
    s = time.time()
    lists, major_v = major_and_outlier(pcd, lists)
    e = time.time()
    print('finish major and outlier', e-s)
    s = time.time()
    lists = classify_checking(lists, pcd, major_v)
    e = time.time()
    print('finish classify checking', e-s)
    s = time.time()
    lists = outlier_reclassify(lists, pcd, major_v)
    e = time.time()
    print('finish outlier reclassify', e-s)
    s = time.time()
    lists, major_v = major_and_outlier(pcd, lists)
    e = time.time()
    print('finish major and outlier', e-s)
    end = time.time()
    s = time.time()
    lists = classify_checking(lists, pcd, major_v)
    np.savetxt('lists.txt', lists)
    e = time.time()
    print('finish classify checking', e-s)
    pcd = color_pcd(pcd, lists)
    o3d.visualization.draw_geometries([pcd])
    print(end - start)


def Calculate_A_D(pcd):
    pcd_size = np.shape(pcd.points)[0]
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    A_list = np.zeros((pcd_size, 3), dtype=float)
    D_list = np.zeros((pcd_size, 3), dtype=float)
    inlier_list = np.zeros((8), dtype=np.float32)
    for i in range(pcd_size):
        [num, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], K1)
        [num2, idx2, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], D1)
        for j in range(RANSAC_K):
            ransac_list = np.random.choice(idx, RANSAC_N)
            data = np.asarray(pcd.points, dtype=np.float32)[ransac_list[0:], :]
            A, D = least_squares_method(data)
            inlier_count = 0
            o_dis = dis_cacu(A, D, pcd.points[i])
            if o_dis <= DIS_THRESHOLD:
                for l in idx2:
                    dis = dis_cacu(A, D, pcd.points[l])
                    if dis <= DIS_THRESHOLD2:
                        inlier_count += 1
            else:
                pass
            if inlier_count > inlier_list[6]:
                inlier_list[:] = [A[0], A[1], A[2], D[0],
                                  D[1], D[2], inlier_count, o_dis]
            elif inlier_count == inlier_list[6]:
                if o_dis < inlier_list[7]:
                    inlier_list[:] = [A[0], A[1], A[2], D[0],
                                      D[1], D[2], inlier_count, o_dis]
        A_list[i, :] = inlier_list[0:3]
        D_list[i, :] = inlier_list[3:6]
        inlier_list[:] = [0, 0, 0, 0, 0, 0, 0, 0]
    return A_list, D_list


def region_growing(pcd, A_list, D_list):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    pcd_size = np.shape(pcd.points)[0]
    lists = np.zeros((pcd_size), dtype=int)
    for i in range(pcd_size):
        if lists[i] == 0:
            lists[i] = i+1
            wait = np.array([i], dtype=int)
            [num, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], D2)
            for j in idx:
                dis = dis_cacu(A_list[i,:], D_list[i,:], pcd.points[j])
                theta = angle(D_list[i,:],D_list[j,:])
                if lists[j] == 0 and dis <= DIS_THRESHOLD2 and theta <= THETA_THRESHOLD:
                    wait = np.append(wait,j)
                    lists[j] = lists[i]
            wait = np.setdiff1d(wait, i)
            while True:
                if np.shape(wait)[0] >= 1:
                    j = wait[0]
                    [num, idx, _] = pcd_tree.search_radius_vector_3d(
                        pcd.points[j], D2)
                    for k in idx:
                        dis = dis_cacu(
                            A_list[j, :], D_list[j, :], pcd.points[k])
                        theta = angle(D_list[j, :], D_list[k, :])
                        if lists[k] == 0 and dis <= DIS_THRESHOLD2 and theta <= THETA_THRESHOLD:
                            wait = np.append(wait, k)
                            lists[k] = lists[i]
                    wait = np.setdiff1d(wait, j)
                else:
                    break
    return lists


def classify_checking(lists, pcd, major_v):
    group = np.array(major_v[:, 6], dtype=int)
    group_num = np.size(group)
    for i in range(group_num):
        for j in range(i+1, group_num):
            theta = angle(major_v[i, 3:6], major_v[j, 3:6])
            if theta <= THETA_THRESHOLD:
                point_location = [k for k in range(
                    len(lists)) if lists[k] == group[j]]
                point_num = np.size(point_location)
                count = 0
                A = major_v[i, 0:3]
                D = major_v[i, 3:6]
                for k in point_location:
                    if dis_cacu(A, D, pcd.points[k]) <= DIS_THRESHOLD2:
                        count += 1
                if count >= 0.5 * point_num:
                    point_location = [k for k in range(
                        len(lists)) if lists[k] == group[i]]
                    lists[point_location[0:]] = group[j]
                    print(group[i], group[j], 'yes')
                    group[i] = group[j]
    return lists


def outlier_reclassify(lists, pcd, major_v):
    outlier = np.asarray([j for j in range(len(lists)) if lists[j] == 0])
    group = np.array(major_v[:, 6], dtype=int)
    group_num = np.size(group)
    for i in outlier:
        closer = 999
        P = pcd.points[i]
        for j in range(group_num):
            temporary_closer = dis_cacu([major_v[j, 0], major_v[j, 1], major_v[j, 2]], [
                                        major_v[j, 3], major_v[j, 4], major_v[j, 5]], P)
            if temporary_closer < closer and temporary_closer <= DIS_THRESHOLD3:
                closer = temporary_closer
                lists[i] = major_v[j, 6]
    return lists


def color_pcd(pcd, lists):
    unique = np.unique(lists)
    for i in unique:
        unique_location = [j for j in range(len(lists)) if lists[j] == i]
        np.asarray(pcd.colors)[unique_location[0:], :] = [
                   random.random(), random.random(), random.random()]
    unique_location = [j for j in range(len(lists)) if lists[j] == 0]
    np.asarray(pcd.points)[unique_location[0:], :] = [0, 0, 0]
    return pcd


def major_and_outlier(pcd, lists):
    unique, counts = np.unique(lists, return_counts=True)
    group = np.shape(unique)[0]
    major_num = 0
    outlier_define = 5
    major_v = np.zeros((1,7))
    for i in range(group):
        if counts[i] > outlier_define:
            major_num += 1
            unique_location = [j for j in range(
                len(lists)) if lists[j] == unique[i]]
            data = np.asarray(pcd.points)[unique_location[0:], :]
            A, D = least_squares_method(data)
            if major_num == 1:
                major_v = np.array(
                    [[A[0], A[1], A[2], D[0], D[1], D[2], unique[i]]], dtype=np.float32)
            else:
                temporary_v = np.array(
                    [[A[0], A[1], A[2], D[0], D[1], D[2], unique[i]]], dtype=np.float32)
                major_v = np.append(major_v, temporary_v, axis=0)
        else:
            unique_location = [j for j in range(
                len(lists)) if lists[j] == unique[i]]
            np.asarray(lists)[unique_location[0:]] = 0
    return lists, major_v


def least_squares_method(data):
    datamean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - datamean)
    return datamean, vv[0]


def angle(a, b):
    aba_a = math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
    aba_b = math.sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2])
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


def dis_cacu(A, D, P):
    V = P - A
    l = np.dot(V, D)
    v = np.linalg.norm(V)
    d = math.sqrt(abs(v*v-l*l))
    return d


if __name__ == '__main__':
    main()
