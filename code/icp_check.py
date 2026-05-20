import os
import open3d as o3d
import numpy as np

index_path = os.path.abspath('edge_index')

def main():
    pcd = o3d.io.read_point_cloud('1.ply')
    index = os.listdir(index_path)
    index_num = np.size(index)
    rmse_list = np.zeros((index_num),dtype=float)
    fit_list = np.zeros((index_num),dtype=float)
    rmse2_list = np.zeros((index_num),dtype=float)
    fit2_list = np.zeros((index_num),dtype=float)
    dis_list = np.zeros((index_num),dtype=float)
    for i in range(index_num):
        print(i)
        point = os.path.join(index_path,index[i])
        collapse_pcd = o3d.io.read_point_cloud(point)
        rmse_list[i],fit_list[i],rmse2_list[i],fit2_list[i],dis_list[i]= icp(collapse_pcd,pcd)
        print(rmse_list[i],fit_list[i],rmse2_list[i],fit2_list[i],dis_list[i])
    np.savetxt('rmse_list.txt',np.asarray(rmse_list))
    np.savetxt('rmse2_list.txt',np.asarray(rmse2_list))
    np.savetxt('fit2_list.txt',np.asarray(fit2_list))
    np.savetxt('dis_list.txt',np.asarray(dis_list))
    min1 = np.where(rmse_list == np.min(rmse_list))[0]
    min2 = np.where(rmse2_list == np.min(rmse2_list))[0]
    max2 = np.where(fit2_list == np.max(fit2_list))[0]
    dis = np.where(dis_list == np.min(dis_list))[0]
    print(index[min1[0]],index[min2[0]],index[max2[0]],index[dis[0]])


def icp(pcd,ref_pcd):
    voxel_size = 0.05
    radius_normal = voxel_size*2
    radius_feature = voxel_size*5
    ref_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = radius_normal, max_nn=100))
    ref_pcd.orient_normals_to_align_with_direction()
    ref_pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(ref_pcd,o3d.geometry.KDTreeSearchParamHybrid(radius = radius_feature, max_nn=100))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = radius_normal, max_nn=100))
    pcd.orient_normals_to_align_with_direction()
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,o3d.geometry.KDTreeSearchParamHybrid(radius = radius_feature, max_nn=100))
    result_ransac = execute_global_registration(pcd, ref_pcd,
                                            pcd_fpfh, ref_pcd_fpfh,
                                            voxel_size)
    result_icp = refine_registration(pcd, ref_pcd, pcd_fpfh, ref_pcd_fpfh,5,result_ransac)
    evaluation = o3d.pipelines.registration.evaluate_registration(pcd, ref_pcd, 0.05, result_icp.transformation)
    pcd = pcd.transform(result_icp.transformation)
    dists1 = ref_pcd.compute_point_cloud_distance(pcd)
    dists2 = pcd.compute_point_cloud_distance(ref_pcd)
    dists = 0.5*np.mean(dists1)+0.5*np.mean(dists2)
    #final_pcd = pcd.transform(result_icp.transformation)
    return result_icp.inlier_rmse,result_icp.fitness,evaluation.inlier_rmse,evaluation.fitness,dists


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh,voxel_size):
    distance_threshold = 5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
         3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size,result_ransac):
    threshold = 3
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    return result

if __name__ == '__main__':
    main()
