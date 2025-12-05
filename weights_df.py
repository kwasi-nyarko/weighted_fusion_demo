import numpy as np
import open3d as o3d

from datafusion import col, functions as f

from data import convert_df2pcd
from expr import COLORS, COORDS, voxel_group_by_expr
from weights import global_registration


def prepare_dataset_df(pc_df, voxel_size, is_target=False):
    aggs = [f.mean(col(c)).alias(c) for c in [*COORDS, *COLORS]]
    pc_df_down = pc_df.aggregate(group_by=voxel_group_by_expr(voxel_size), aggs=aggs)
    pcd_down = convert_df2pcd(pc_df_down)

    TRANS_INIT = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    if not is_target:
        pcd_down = pcd_down.transform(TRANS_INIT)

    radius_normal = voxel_size * 2
    params = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    pcd_down.estimate_normals(params)

    radius_feature = voxel_size * 5
    params = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, params)

    return pcd_down, pcd_fpfh


def compute_rmse_df(point_clouds_df, accuracies, voxel_size):
    # prepeare target
    target_pc_df = point_clouds_df[accuracies.index(min(accuracies))]
    target_down, target_fpfh = prepare_dataset_df(
        target_pc_df, voxel_size, is_target=True
    )

    rmse = []

    for source_pc_df in point_clouds_df:
        (source_down, source_fpfh) = prepare_dataset_df(source_pc_df, voxel_size)

        result_ransac = global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size
        )
        rmse.append(result_ransac.inlier_rmse)

    return np.array(rmse)


def compute_weights_df(reference_df, point_clouds_df, accuracies, voxel_size):
    reference_voxel_count = reference_df.aggregate(
        group_by=voxel_group_by_expr(voxel_size / 2.0), aggs=[]
    ).count()

    weights = []

    for pc_df, accuracy in zip(point_clouds_df, accuracies):
        pc_volxel_count = pc_df.aggregate(
            group_by=voxel_group_by_expr(voxel_size / 2.0), aggs=[]
        ).count()

        completeness = float(pc_volxel_count) / float(reference_voxel_count)
        print("Completeness:", completeness)
        weight = accuracy / completeness

        weights.append(weight)

    if len(weights) > 1:
        weights = np.sum(weights) - weights

    rmse = compute_rmse_df(point_clouds_df, accuracies, voxel_size)

    return weights / (1 - rmse)
