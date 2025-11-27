import copy
import numpy as np
import open3d as o3d


def prepare_dataset(pcd, voxel_size, is_target=False):
    TRANS_INIT = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    if not is_target:
        # prevent transform mutating pcd passed by reference in place
        pcd = copy.deepcopy(pcd).transform(TRANS_INIT)

    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    params = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    pcd_down.estimate_normals(params)

    radius_feature = voxel_size * 5
    params = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, params)

    return pcd_down, pcd_fpfh


def global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1
    p2p_transformation_estimation = (
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    )
    correspondance_checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
            distance_threshold
        ),
    ]
    convergence_criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
        100000, 0.999
    )

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        p2p_transformation_estimation,
        3,
        correspondance_checkers,
        convergence_criteria,
    )

    return result


def compute_rmse(point_clouds_pcd, accuracies, voxel_size):
    # prepeare target
    target_pcd = point_clouds_pcd[accuracies.index(min(accuracies))]
    (target_down, target_fpfh) = prepare_dataset(target_pcd, voxel_size, is_target=True)

    rmse = []

    for source_pcd in point_clouds_pcd:
        (source_down, source_fpfh) = prepare_dataset(source_pcd, voxel_size)

        result_ransac = global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size
        )
        rmse.append(result_ransac.inlier_rmse)

    return np.array(rmse)


def compute_surface_area(pcd, voxel_size):
    radii = [voxel_size, voxel_size * 2, voxel_size * 4, voxel_size * 5]
    radii = o3d.utility.DoubleVector(radii)

    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, radii
    )

    return rec_mesh.get_surface_area()


def compute_weights(reference_pcd, point_clouds_pcd, accuracies, voxel_size):
    reference_pcd_ds = reference_pcd.voxel_down_sample(voxel_size=voxel_size * 2)
    reference_surface_area = compute_surface_area(reference_pcd_ds, voxel_size * 2)

    weights = []

    for pcd, accuracy in zip(point_clouds_pcd, accuracies):

        pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size * 2)

        pcd_surface_area = compute_surface_area(pcd_ds, voxel_size * 2)

        completeness = pcd_surface_area / reference_surface_area
        weight = accuracy / completeness

        weights.append(weight)

    if len(weights) > 1:
        weights = np.sum(weights) - weights

    rmse = compute_rmse(point_clouds_pcd, accuracies, voxel_size)

    return weights / (1 - rmse)
