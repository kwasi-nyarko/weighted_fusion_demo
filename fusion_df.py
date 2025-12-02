import numpy as np
import open3d as o3d
import tqdm

from datafusion import col, lit

from fusion import request_points


def estimate_noise(pcd, distance_threshold):
    if len(pcd.points) <= 10:
        return 1
    else:
        # estimate std
        _, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=5,
            num_iterations=1000,
        )

        pcd_inliers = pcd.select_by_index(inliers)

        dists = pcd.compute_point_cloud_distance(pcd_inliers)

        return 1 - np.std(np.asarray(dists))


def request_points_df(pc_df, reference_point, buff):
    selection = o3d.geometry.PointCloud()

    f = (
        (col("x") < lit(reference_point[0] + buff))
        & (col("x") > lit(reference_point[0] - buff))
        & (col("y") < lit(reference_point[1] + buff))
        & (col("y") > lit(reference_point[1] - buff))
        & (col("z") < lit(reference_point[2] + buff))
        & (col("z") > lit(reference_point[2] - buff))
    )

    for rb in pc_df.select("x", "y", "z", "red", "green", "blue").filter(f).collect():
        pc = rb.to_tensor().to_numpy()
        selection.points.extend(pc[:, :3])
        selection.colors.extend(pc[:, 3:])

    return selection


def check_points_distribution(pcd, reference_point, threshold, voxel_size):
    if len(pcd.points) <= 1:
        return False

    coords_mean = np.mean(np.asarray(pcd.points), axis=0)
    coords_diff_norm = np.linalg.norm(coords_mean - reference_point)

    return coords_diff_norm > (threshold * voxel_size)


def adjusted_voxel2(
    reference_df_ds_sub,
    reference_point,
    super_voxel_pcds,
    global_weights,
    buff,
    threshold,
):
    fused_sub_pcd = o3d.geometry.PointCloud()

    sub_voxel_pcd = request_points_df(reference_df_ds_sub, reference_point, buff)
    sub_voxel_points = np.asarray(sub_voxel_pcd.points)

    for i in range(sub_voxel_points.shape[0]):
        sub_voxel_point = sub_voxel_points[i]

        sub_voxel_pcds = []
        local_weights = []

        for pcd in super_voxel_pcds:
            filtered = request_points(pcd, sub_voxel_point, buff * 0.5)

            noise = estimate_noise(filtered, buff * 0.5 * 0.01)

            local_weight = len(filtered.points) * noise
            local_weights.append(local_weight)

            sub_voxel_pcds.append(filtered)

        weights = global_weights * np.array(local_weights)
        weights = weights / np.sum(weights)

        idx = np.argmax(weights)

        point = sub_voxel_pcds.pop(idx)

        fused_sub_pcd.points.extend(point.points)
        fused_sub_pcd.colors.extend(point.colors)

        if check_points_distribution(point, sub_voxel_point, threshold, buff):
            for pcd in sub_voxel_pcds:
                fused_sub_pcd.points.extend(pcd.points)
                fused_sub_pcd.colors.extend(pcd.colors)

    return fused_sub_pcd


def weighted_fusion_filter_df(
    point_clouds_df,
    global_weights,
    reference_points,
    reference_df_ds_sub,
    voxel_size,
    threshold,
    k_global,
    k_local,
):
    fused_pcd = o3d.geometry.PointCloud()

    global_weights = np.array(global_weights)

    buff = (1.1 * voxel_size) / 2

    for i in tqdm.tqdm(range(reference_points.shape[0])):
        reference_point = reference_points[i]

        voxel_pcds = []
        local_weights = []

        for pc_df in point_clouds_df:
            voxel_pcd = request_points_df(pc_df, reference_point, buff)

            noise_point = estimate_noise(voxel_pcd, buff * 0.01)

            weight = (len(voxel_pcd.points) / (voxel_size * 10) ** 3) * noise_point

            voxel_pcds.append(voxel_pcd)
            local_weights.append(weight)

        local_weights = np.array(local_weights)
        local_weights = local_weights / np.sum(local_weights)
        weights = (k_global * global_weights) + (k_local * local_weights)
        weights = weights / np.sum(weights)

        idx = np.argmax(weights)
        voxel_pcd = voxel_pcds[idx]

        if check_points_distribution(voxel_pcd, reference_point, threshold, voxel_size):
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

    return fused_pcd
