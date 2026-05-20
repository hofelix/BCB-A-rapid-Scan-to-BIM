import open3d as o3d
import os
import math
import numpy as np
import random

K1 = 20
D1 = 0.5
D2 = 1
RANSAC_K = 300
RANSAC_N = 3
DISTANCE_THRESHOLD = 0.01
DISTANCE_THRESHOLD2 = 0.02
DISTANCE_THRESHOLD3 = 0.02
THETA_THRESHOLD = 10

def main():
    pcd = o3d.io.read_point_cloud('one_edge.ply')            #input pcd
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    pcd_size = np.shape(pcd.points)[0]
    print('finish load point')
    A_list = np.zeros((pcd_size, 3))
    D_list = np.zeros((pcd_size, 3))
    for i in range(pcd_size):
        [num, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], K1)
        [num2, idx2, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], D1)
        inlier_list = np.zeros((8))
        for j in range(RANSAC_K):
            ransac_list = np.random.choice(idx, RANSAC_N)
            xyz = np.zeros((RANSAC_N,3))
            for k in range(RANSAC_N):
                xyz[k,:] = pcd.points[ransac_list[k]]
            A ,D= least_squares_method(xyz)
            inlier_num = 0
            o_dis = dis_cacu(A, D, pcd.points[i])
            if o_dis <= DISTANCE_THRESHOLD :
                for l in idx2:
                    dis = dis_cacu(A, D, pcd.points[l])
                    if dis <= DISTANCE_THRESHOLD2 :
                        inlier_num += 1
            else:
                inlier_num = 0

            if inlier_num > inlier_list[6]:
                inlier_list[0] = A[0]
                inlier_list[1] = A[1]
                inlier_list[2] = A[2]
                inlier_list[3] = D[0]
                inlier_list[4] = D[1]
                inlier_list[5] = D[2]
                inlier_list[6] = inlier_num
                inlier_list[7] = o_dis
            elif inlier_num == inlier_list[6]:
                if o_dis < inlier_list[7]:
                    inlier_list[0] = A[0]
                    inlier_list[1] = A[1]
                    inlier_list[2] = A[2]
                    inlier_list[3] = D[0]
                    inlier_list[4] = D[1]
                    inlier_list[5] = D[2]
                    inlier_list[6] = inlier_num
                    inlier_list[7] = o_dis
        A_list[i,:] = [inlier_list[0],inlier_list[1],inlier_list[2]]
        D_list[i,:] = [inlier_list[3],inlier_list[4],inlier_list[5]]
    print('finish A D')

    lists = np.zeros((pcd_size))
    for i in range(pcd_size):
        if lists[i] == 0:
            lists[i] = i+1
            wait = np.zeros((1))
            wait[0] = i
            [num, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], D2)
            for j in idx:
                dis = dis_cacu(A_list[i,:], D_list[i,:], pcd.points[j])
                theta = angle(D_list[i,:],D_list[j,:])
                if lists[j] == 0 and dis <= DISTANCE_THRESHOLD2 and theta <= THETA_THRESHOLD:
                    wait = np.append(wait,j)
                    lists[j] = lists[i]
            wait = np.setdiff1d(wait, i)
            while True:
                if np.shape(wait)[0] >= 1:
                     j = int(wait[0])
                     [num2, idx2, _] = pcd_tree.search_radius_vector_3d(pcd.points[j], D2)
                     for k in idx2:
                         dis = dis_cacu(A_list[j,:], D_list[j,:], pcd.points[k])
                         theta = angle(D_list[j,:],D_list[k,:])
                         if lists[k] == 0 and dis <= DISTANCE_THRESHOLD2 and theta <= THETA_THRESHOLD:
                             wait = np.append(wait,k)
                             lists[k] = lists[i]
                     wait = np.setdiff1d(wait, j)
                else:
                    break
    '''outlier reclassify'''
    print('outlier reclassify')
    unique, counts = np.unique(lists, return_counts=True)
    group = np.shape(unique)[0]
    major_v = np.zeros((1,7))
    major_num = 0
    outlier_average = np.average(counts)
    outlier_std = np.std(counts, ddof = 0)
    outlier_define = 3
    print(outlier_define)
    for i in range(group):
        if counts[i] > outlier_define:
            major_num += 1
            unique_location = [ j for j in range(len(lists)) if lists[j] == unique[i]]
            xyz = np.zeros((len(unique_location),3))
            for k in range(len(unique_location)):
                xyz[k,:] = pcd.points[unique_location[k]]
            A ,D= least_squares_method(xyz)
            if major_num == 1:
                major_v[0,0] = A[0]
                major_v[0,1] = A[1]
                major_v[0,2] = A[2]
                major_v[0,3] = D[0]
                major_v[0,4] = D[1]
                major_v[0,5] = D[2]
                major_v[0,6] = unique[i]
            else:
                temporary_v = np.zeros((1,7))
                temporary_v[0,0] = A[0]
                temporary_v[0,1] = A[1]
                temporary_v[0,2] = A[2]
                temporary_v[0,3] = D[0]
                temporary_v[0,4] = D[1]
                temporary_v[0,5] = D[2]
                temporary_v[0,6] = unique[i]
                major_v = np.append(major_v,temporary_v, axis=0)
        else:
            unique_location = [ j for j in range(len(lists)) if lists[j] == unique[i]]
            np.asarray(lists)[unique_location[0:]] = 0
    outlier = np.asarray([ j for j in range(len(lists)) if lists[j] == 0])
    for i in outlier:
        closer = 999
        P = pcd.points[i]
        outlier_check = True
        for j in range(major_num):
            A = [major_v[j,0],major_v[j,1],major_v[j,2]]
            D = [major_v[j,3],major_v[j,4],major_v[j,5]]
            temporary_closer = dis_cacu(A, D, P)
            if temporary_closer < closer and temporary_closer <= DISTANCE_THRESHOLD3:
                closer = temporary_closer
                lists[i] = major_v[j,6]
                outlier_check = False
        if outlier_check == True:
            print(i)
            pcd.points[i] = [0,0,0]


    for i in unique:
        unique_location = [ j for j in range(len(lists)) if lists[j] == i ]
        np.asarray(pcd.colors)[unique_location[1:], :] = [random.random(),random.random(),random.random()]
    o3d.visualization.draw_geometries([pcd])

def least_squares_method(data):
    # Calculate the mean of the points, i.e. the 'center' of the cloud
    datamean = data.mean(axis=0)
    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(data - datamean)
    # Now vv[0] contains the first principal component, i.e. the direction
    # vector of the 'best fit' line in the least squares sense.
    return datamean,vv[0]

def angle(a,b):
    aba_a = math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
    aba_b = math.sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2])
    dot = np.dot(a,b)
    cos_theta = dot/(aba_a*aba_b)
    if cos_theta > 0 and cos_theta <= 1:
        theta = math.degrees(math.acos(cos_theta))
    elif cos_theta < 0 and cos_theta >= -1:
        theta = 180 - math.degrees(math.acos(cos_theta))
    elif cos_theta > 1 :
        theta = 0
    elif cos_theta < -1 :
        theta = 0
    else:
        theta = 90

    return theta

def dis_cacu(A, D, P):
    V = P - A
    l = np.dot(V, D)
    v = math.sqrt(V[0]*V[0]+V[1]*V[1]+V[2]*V[2])
    d = math.sqrt(abs(v*v-l*l))
    return d

if __name__ == '__main__':
    main()
