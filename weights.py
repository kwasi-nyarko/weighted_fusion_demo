import numpy as np
import open3d as o3d

from data import np_point_cloud2_pcd, pcd2np_point_cloud


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size):

    trans_init = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    source.transform(trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def compute_rmse(lst_PC, lst_acc, voxel_size):
    PC_list = lst_PC.copy()
    target = np_point_cloud2_pcd(PC_list.pop(lst_acc.index(min(lst_acc))))
    rmse = []
    lst_pcs = []
    PC_list = lst_PC.copy()
    for pc in PC_list:
        source = np_point_cloud2_pcd(pc)

        source, target, source_down, target_down, source_fpfh, target_fpfh = (
            prepare_dataset(source, target, voxel_size)
        )
        result_ransac = execute_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size
        )
        rmse.append(result_ransac.inlier_rmse)
        lst_pcs.append(
            pcd2np_point_cloud(source.transform(result_ransac.transformation))
        )
        # print(result_ransac.inlier_rmse)
    return lst_pcs, np.array(rmse)


def compute_surface_area(pcd, voxel_size):
    radii = [voxel_size, voxel_size * 2, voxel_size * 4, voxel_size * 5]
    radii = o3d.utility.DoubleVector(radii)

    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, radii
    )

    return rec_mesh.get_surface_area()


def compute_weights(reference_pcd, point_clouds, accuracies, voxel_size):
    reference_pcd_ds = reference_pcd.voxel_down_sample(voxel_size=voxel_size * 2)
    reference_surface_area = compute_surface_area(reference_pcd_ds, voxel_size * 2)

    weights = []

    for pc_idx in range(len(point_clouds)):
        pcd = np_point_cloud2_pcd(point_clouds[pc_idx])
        pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size * 2)

        pcd_surface_area = compute_surface_area(pcd_ds, voxel_size * 2)

        completeness = pcd_surface_area / reference_surface_area
        weight = accuracies[pc_idx] / completeness

        weights.append(weight)

    if len(weights) > 1:
        weights = np.sum(weights) - weights

    # weights_ = weights / np.sum(weights)
    lst_PCs, rmse = compute_rmse(point_clouds, accuracies, voxel_size)
    g_weight = weights / (1 - rmse)
    return g_weight
