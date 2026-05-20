import numpy as np
import open3d as o3d




def main():
    pcd = o3d.io.read_point_cloud('21.ply')
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    print(np.asarray(pcd_tree))
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #o3d.visualization.draw_geometries([pcd],point_show_normal = True)





if __name__ == '__main__':
    main()
