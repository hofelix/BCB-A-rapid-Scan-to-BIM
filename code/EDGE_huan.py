import open3d as o3d
import os
import math
import numpy as np
import time
import random

K1 = 100
D1 = 0.1
RANSAC_K = 200
RANSAC_N = 5
NORMAL_THRESHOLD = 0.005
G_THETA_THRESHOLD = 90
MAXIC = K1*0.7

def main():
    pcd = o3d.io.read_point_cloud('2.ply')            #input pcd
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)           #kd_tree
    pcd_size = np.shape(pcd.points)[0]
    print(pcd_size)               #points number
    edge_list = np.array([0])
    print('start calculate')
    start = time.time()
    for i in range(pcd_size):
        [num, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], K1)
        [num2, idx2, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], D1)
        data = np.zeros((K1,3))
        for k in range(K1):
            data[k,:] = pcd.points[idx[k]]
        n = run_ransac(data, lambda x, y: is_inlier(x, y, NORMAL_THRESHOLD), RANSAC_N, MAXIC, RANSAC_K)
        distance = check_inlier(n[0],n[1],n[2],n[3],pcd.points[i])
        if distance <= NORMAL_THRESHOLD:
            judge_num = 0
            v_list = np.zeros((2,3))
            for l in idx:
                judge = check_inlier(n[0],n[1],n[2],n[3],pcd.points[l])
                if judge < NORMAL_THRESHOLD:
                    v_list[judge_num,:] = pcd.points[l]
                    judge_num += 1
                    if judge_num == 2:
                        break;
            x = v_list[1,0] - v_list[0,0]
            y = v_list[1,1] - v_list[0,1]
            z = -1*(x*n[0]+y*n[1])/n[2]
            v = unit_vector(np.array([x,y,z]))
            nn = unit_vector(n[:3])
            u = np.cross(nn, v)
            inlier_radius_list = np.zeros((1,3))
            append_check = 0
            for m in idx2:
                judge = check_inlier(n[0],n[1],n[2],n[3],pcd.points[m])
                if judge < NORMAL_THRESHOLD:
                    if append_check == 0:
                        inlier_radius_list = [pcd.points[m]]
                        append_check += 1
                    else:
                        inlier_radius_list = np.append(inlier_radius_list,[pcd.points[m]],axis=0)
                        append_check += 1
            o = pcd.points[i]
            theta_list = np.zeros((append_check-1,1))
            for n in range(append_check-1):
                p = inlier_radius_list[n+1,:]
                op = p - o
                du = np.dot(op,u)
                dv = np.dot(op,v)
                theta = caculus_theta(du,dv)
                if np.isnan(theta) == True:
                    theta = 0
                theta_list[n] = int(theta)
            theta_list = np.sort(theta_list, axis=0)
            if append_check-1 > 2:
                G_theta = caculus_G_theta(theta_list,append_check-1)
                if G_theta > G_THETA_THRESHOLD:
                    edge_list = np.append(edge_list,int(i))
    end = time.time()
    print(end - start)
    print('finish Gtheta\n start output')
    print(edge_list)
    inlier_cloud = pcd.select_by_index(edge_list)
    inlier_cloud.paint_uniform_color([1, 0, 0])
    o3d.io.write_point_cloud("2huan.pcd", inlier_cloud)
    o3d.visualization.draw_geometries([inlier_cloud])


def caculus_G_theta(list,num):
    theta_list = np.zeros((num,1))
    for i in range(num-1):
        theta_list[i] = list[i+1] - list[i]
    theta_list[num-1] = 360 - list[num-1] + list[0]
    G_theta = np.max(theta_list)
    return G_theta

def caculus_theta(du,dv):
    theta = math.degrees(math.atan(du/dv))
    if du > 0 and dv >0:     #1
        pass
    elif du > 0 and dv <0:   #4
        theta += 360
    elif du < 0 and dv >0:   #2
        theta += 180
    elif du < 0 and dv <0:   #3
        theta += 180
    return theta

def unit_vector(vector):
    x = vector[0]
    y = vector[1]
    z = vector[2]
    unit = math.sqrt(x*x+y*y+z*z)
    if unit == 0:
        unit_v = vector
    else:
        unit_v = vector/unit
    return unit_v

def check_inlier(A,B,C,D,point):
    dis = abs(A*point[0]+B*point[1]+C*point[2]+D)/math.sqrt(A*A+B*B+C*C)
    return dis

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def run_ransac(data, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True):
    best_ic = 0
    best_model = None
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1
        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    return best_model

if __name__ == '__main__':
    main()
