import open3d as o3d
import os
import math
import numpy as np
import random

K1 = 100
D1 = 0.1
RANSAC_K = 200
RANSAC_N = 3
DISTANCE_THRESHOLD = 0.005
CROSS_THRESHOLD = 0.2


def main():
    pcd = o3d.io.read_point_cloud('one_edge.ply')
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    while True:
        i = int(input('imput num'))
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        pcd.colors[i] = [1, 0, 0]
        [num, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], K1)
        [num2, idx2, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], D1)
        np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]
        o3d.visualization.VisualizerWithVertexSelection()
        o3d.visualization.draw_geometries([pcd])



if __name__ == '__main__':
    main()
