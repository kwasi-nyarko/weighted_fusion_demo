import numpy as np
import open3d as o3d
import tqdm


def estimate_std(pcd, distance_threshold):
    _, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=5,
        num_iterations=1000,
    )

    pcd_inliers = pcd.select_by_index(inliers)

    dists = pcd.compute_point_cloud_distance(pcd_inliers)

    return 1 - np.std(np.asarray(dists))


def request_points(pcd, reference_point, buff):
    points = np.asarray(pcd.points)

    f = (
        (points[:, 0] < reference_point[0] + buff)
        & (points[:, 0] > reference_point[0] - buff)
        & (points[:, 1] < reference_point[1] + buff)
        & (points[:, 1] > reference_point[1] - buff)
        & (points[:, 2] < reference_point[2] + buff)
        & (points[:, 2] > reference_point[2] - buff)
    )
    indices = np.argwhere(f).flatten().tolist()
    selection = pcd.select_by_index(indices=indices)

    if len(selection.points) <= 10:
        noise_point = 1
    else:
        noise_point = estimate_std(selection, buff * 0.01)

    return selection, noise_point


def check_points_distribution(pcd, reference_point, threshold, voxel_size):
    if len(pcd.points) <= 1:
        return False

    coords_mean = np.mean(np.asarray(pcd.points), axis=0)
    coords_diff_norm = np.linalg.norm(coords_mean - reference_point)

    return coords_diff_norm > (threshold * voxel_size)


def adjusted_voxel2(
    reference_pcd_ds_sub,
    reference_point,
    super_voxel_pcds,
    global_weights,
    buff,
    threshold,
):
    fused_sub_pcd = o3d.geometry.PointCloud()

    sub_voxel_pcd, _ = request_points(reference_pcd_ds_sub, reference_point, buff)
    sub_voxel_points = np.asarray(sub_voxel_pcd.points)

    for i in range(sub_voxel_points.shape[0]):
        sub_voxel_point = sub_voxel_points[i]

        sub_voxel_pcds = []
        local_weights = []

        for pcd in super_voxel_pcds:
            filtered, noise = request_points(pcd, sub_voxel_point, buff * 0.5)

            local_weight = len(filtered.points) * noise
            local_weights.append(local_weight)

            sub_voxel_pcds.append(filtered)

        weights = np.array(np.array(global_weights) * np.array(local_weights))
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


def weighted_fusion_filter(
    point_clouds_pcd,
    global_weights,
    reference_points,
    reference_pcd_ds_sub,
    voxel_size,
    k_1,
    k_2,
    threshold,
):
    fused_points = o3d.geometry.PointCloud()

    buff = (voxel_size + voxel_size * 0.1) / 2

    for i in tqdm.tqdm(range(reference_points.shape[0])):
        reference_point = reference_points[i]

        voxel_pcds = []
        local_weights = []

        for pcd in point_clouds_pcd:
            voxel_pcd, noise_point = request_points(pcd, reference_point, buff)
            voxel_pcds.append(voxel_pcd)

            weight = (len(voxel_pcd.points) / (voxel_size * 10) ** 3) * noise_point
            local_weights.append(weight)

        local_weights = local_weights / np.sum(local_weights)

        weights = (k_1 * np.array(global_weights)) + (k_2 * np.array(local_weights))
        weights = weights / np.sum(weights)

        idx = np.argmax(weights)
        voxel_pcd = voxel_pcds[idx]

        if check_points_distribution(voxel_pcd, reference_point, threshold, voxel_size):
            voxel_pcd = adjusted_voxel2(
                reference_pcd_ds_sub,
                reference_point,
                voxel_pcds,
                global_weights,
                buff,
                0.5 * threshold,
            )

        fused_points.points.extend(voxel_pcd.points)
        fused_points.colors.extend(voxel_pcd.colors)

    return fused_points
