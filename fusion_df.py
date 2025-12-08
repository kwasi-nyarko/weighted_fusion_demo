import numpy as np
import open3d as o3d
import tqdm

from datafusion import col, functions as f

from data import convert_df2pcd
from expr import (
    COLORS,
    COORDS,
    point_buffer_filter_expr,
    voxel_filter_expr,
    voxel_group_by_expr,
)
from fusion import check_points_distribution, estimate_noise, request_points


def adjusted_voxel2(
    reference_ds_sub,
    reference_point,
    super_voxel_pcds,
    global_weights,
    buffer,
    threshold,
):
    fused_sub_pcd = o3d.geometry.PointCloud()

    expr = point_buffer_filter_expr(reference_point, buffer)
    sub_voxel_df = reference_ds_sub.filter(expr)
    sub_voxel_pcd = convert_df2pcd(sub_voxel_df, estimate_normals=False)
    sub_voxel_points = np.asarray(sub_voxel_pcd.points)

    for i in range(sub_voxel_points.shape[0]):
        sub_voxel_point = sub_voxel_points[i]

        sub_voxel_pcds = []
        local_weights = []

        for pcd in super_voxel_pcds:
            filtered = request_points(pcd, sub_voxel_point, buffer * 0.5)

            noise = estimate_noise(filtered, buffer * 0.5 * 0.01)

            local_weight = len(filtered.points) * noise
            local_weights.append(local_weight)

            sub_voxel_pcds.append(filtered)

        weights = global_weights * np.array(local_weights)
        weights = weights / np.sum(weights)

        idx = np.argmax(weights)

        point = sub_voxel_pcds.pop(idx)

        fused_sub_pcd.points.extend(point.points)
        fused_sub_pcd.colors.extend(point.colors)

        if check_points_distribution(point, sub_voxel_point, threshold, buffer):
            for pcd in sub_voxel_pcds:
                fused_sub_pcd.points.extend(pcd.points)
                fused_sub_pcd.colors.extend(pcd.colors)

    return fused_sub_pcd


def weighted_fusion_filter_df(
    point_clouds,
    global_weights,
    reference,
    voxel_size,
    threshold,
    k_global,
    k_local,
):
    fused_pcd = o3d.geometry.PointCloud()

    global_weights = np.array(global_weights)
    buff = float((1.1 * voxel_size) / 2.0)

    # reference points
    reference_df_ds = reference.aggregate(
        group_by=voxel_group_by_expr(voxel_size),
        aggs=[f.mean(col(c)).alias(c) for c in [*COORDS]],
    )

    # mega voxels
    factor = 50
    mega_voxel_size = float(voxel_size * factor)
    mega_voxel_df = reference.aggregate(
        group_by=voxel_group_by_expr(mega_voxel_size), aggs=[]
    )

    pbar = tqdm.tqdm(total=reference_df_ds.count(), smoothing=0, delay=1)

    for rb in mega_voxel_df.execute_stream():
        voxels = rb.to_pyarrow().to_tensor().to_numpy()

        for i in range(voxels.shape[0]):
            voxel_filter = voxel_filter_expr(voxels[i], mega_voxel_size)
            voxel_filter_buf = voxel_filter_expr(voxels[i], mega_voxel_size, buff)

            # cache points
            point_clouds_df_cached = [
                df.filter(voxel_filter_buf).cache() for df in point_clouds
            ]

            # sub voxels
            reference_df_ds_sub = (
                reference.filter(voxel_filter_buf)
                .aggregate(
                    group_by=voxel_group_by_expr(voxel_size / 2),
                    aggs=[f.mean(col(c)).alias(c) for c in [*COORDS, *COLORS]],
                )
                .cache()
            )

            # reference points
            df = reference_df_ds.filter(voxel_filter)
            pcd = convert_df2pcd(df, estimate_normals=False)
            reference_points = np.asarray(pcd.points)

            for i in range(reference_points.shape[0]):
                reference_point = reference_points[i]

                voxel_pcds = []
                local_weights = []

                for pc_df in point_clouds_df_cached:
                    expr = point_buffer_filter_expr(reference_point, buff)
                    voxel_df = pc_df.filter(expr)
                    voxel_pcd = convert_df2pcd(voxel_df, estimate_normals=False)

                    count = len(voxel_pcd.points)
                    noise_point = estimate_noise(voxel_pcd, buff * 0.01)

                    weight = (count / (voxel_size * 10) ** 3) * noise_point

                    voxel_pcds.append(voxel_pcd)
                    local_weights.append(weight)

                local_weights = np.array(local_weights)
                local_weights = local_weights / np.sum(local_weights)
                weights = (k_global * global_weights) + (k_local * local_weights)
                weights = weights / np.sum(weights)

                idx = np.argmax(weights)
                voxel_pcd = voxel_pcds[idx]

                if check_points_distribution(
                    voxel_pcd, reference_point, threshold, voxel_size
                ):
                    voxel_pcd = adjusted_voxel2(
                        reference_df_ds_sub,
                        reference_point,
                        voxel_pcds,
                        global_weights,
                        buff,
                        0.5 * threshold,
                    )

                fused_pcd.points.extend(voxel_pcd.points)
                fused_pcd.colors.extend(voxel_pcd.colors)

                pbar.update()
    pbar.close()

    return fused_pcd
