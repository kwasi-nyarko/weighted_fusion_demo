import numpy as np
import tqdm

from data import np_point_cloud2_pcd


def estimate_std(point_cloud, distance_threshold):
    pcd = np_point_cloud2_pcd(point_cloud)

    _, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=5, num_iterations=1000
    )

    pcd_inliers = pcd.select_by_index(inliers)

    dists = pcd.compute_point_cloud_distance(pcd_inliers)

    return 1 - np.std(np.asarray(dists))


def request_points(point_cloud, reference_point, buff):
    points = point_cloud[
        np.where(
            (point_cloud[:, 0] < reference_point[0] + buff)
            & (point_cloud[:, 0] > reference_point[0] - buff)
            & (point_cloud[:, 1] < reference_point[1] + buff)
            & (point_cloud[:, 1] > reference_point[1] - buff)
            & (point_cloud[:, 2] < reference_point[2] + buff)
            & (point_cloud[:, 2] > reference_point[2] - buff)
        ),
        :,
    ]

    points = points.reshape(-1, 6)

    noise_point = 1 if points.shape[0] <= 10 else estimate_std(points, buff * 0.01)

    return points, noise_point


def check_points_distribution(selected_point, reference_point, threshold, voxel_size):
    selected_point = selected_point.reshape(-1, 6)

    if selected_point.shape[0] <= 1:
        return False

    coords_diff_norm = np.linalg.norm(
        np.mean(selected_point, axis=0)[:3] - reference_point[:3]
    )

    return coords_diff_norm > (threshold * voxel_size)


def adjusted_voxel2(
    reference_pc_ds_sub,
    reference_point,
    super_voxel_points,
    global_weights,
    buff,
    threshold,
    point_cont,
):
    fused_sub_point = np.zeros((1, 6))
    sub_voxel_points, _ = request_points(reference_pc_ds_sub, reference_point, buff)

    for i in range(sub_voxel_points.shape[0]):
        sub_voxel_point = sub_voxel_points[i]

        sub_vox_pnt = []
        local_weights = []

        for super_voxel_point in super_voxel_points:
            points, noise = request_points(
                super_voxel_point, sub_voxel_point, buff * 0.5
            )
            local_weight = points.shape[0] * noise
            local_weights.append(local_weight)

            sub_vox_pnt.append(points)

        weights = np.array(np.array(global_weights) * np.array(local_weights))
        weights = weights / np.sum(weights)

        idx = np.argmax(weights)
        point = sub_vox_pnt[idx]

        point_cont[idx] = point_cont[idx] + 1

        if check_points_distribution(point, sub_voxel_point, threshold, buff):
            point = np.vstack(sub_vox_pnt)
            point_cont = point_cont + 1

        fused_sub_point = np.concatenate(
            (fused_sub_point, point.reshape(-1, 6)), axis=0
        )

    return fused_sub_point[1:, :], point_cont


def weighted_fusion_filter(
    point_clouds,
    global_weights,
    reference_pc_ds,
    reference_pc_ds_sub,
    voxel_size,
    k_1,
    k_2,
    threshold,
):
    # FIXME: proper point_cont handling. Only point_cont_ from last iteration is returned.
    point_cont = np.zeros(len(point_clouds))
    fused_points = np.zeros((1, 6))
    buff = (voxel_size + voxel_size * 0.1) / 2

    for i in tqdm.tqdm(range(reference_pc_ds.shape[0])):
        reference_point = reference_pc_ds[i]

        points = []
        local_weights = []

        for point_cloud in point_clouds:
            pnts, noise_point = request_points(point_cloud, reference_point, buff)
            points.append(pnts)

            weight = (pnts.shape[0] / (voxel_size * 10) ** 3) * noise_point
            local_weights.append(weight)

        local_weights = local_weights / np.sum(local_weights)

        weights = (k_1 * np.array(global_weights)) + (k_2 * np.array(local_weights))
        weights = weights / np.sum(weights)

        idx = np.argmax(weights)
        point = points[idx]

        if check_points_distribution(point, reference_point, threshold, voxel_size):
            point, point_cont_ = adjusted_voxel2(
                reference_pc_ds_sub,
                reference_point,
                points,
                global_weights,
                buff,
                0.5 * threshold,
                point_cont,
            )
        else:
            point_cont[idx] = point_cont[idx] + 1

        fused_points = np.concatenate((fused_points, point.reshape(-1, 6)), axis=0)

    return fused_points[1:, :], point_cont_ 
