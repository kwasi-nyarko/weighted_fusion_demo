import numpy as np
import tqdm

from data import np_point_cloud2_pcd


def estimate_std(pointCloud, distance_threshold):
    pcd = np_point_cloud2_pcd(pointCloud)
    _, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=5, num_iterations=1000
    )
    pcd_inliers = pcd.select_by_index(inliers)
    dists = pcd.compute_point_cloud_distance(pcd_inliers)
    dists = np.asarray(dists)
    return 1 - np.std(dists)


def request_points(data, reference_data, buff):
    # data = data.reshape(-1,6)
    noise_point = 1
    # print(data.shape)
    points = data[
        np.where(
            (data[:, 0] < reference_data[0] + buff)
            & (data[:, 0] > reference_data[0] - buff)
            & (data[:, 1] < reference_data[1] + buff)
            & (data[:, 1] > reference_data[1] - buff)
            & (data[:, 2] < reference_data[2] + buff)
            & (data[:, 2] > reference_data[2] - buff)
        ),
        :,
    ]
    points = points.reshape(-1, 6)
    if points.shape[0] > 10:
        noise_point = estimate_std(points, buff * 0.01)
    return points, noise_point


def check_points_distribution(selected_point, reference_point, threshold, voxel_size):
    selected_point = selected_point.reshape(-1, 6)
    if selected_point.shape[0] > 1:
        return np.linalg.norm(
            np.mean(selected_point, axis=0)[:3] - reference_point[:3]
        ) > (threshold * voxel_size)
    else:
        return False


def adjusted_voxel2(
    ds_sub_reference,
    ds_reference_pnt,
    super_voxel_points,
    g_weights,
    buff,
    threshold,
    point_cont,
):
    # point_cont = np.zeros(len(super_voxel_points))
    fused_sub_point = np.zeros((1, 6))
    sub_voxel_points, _ = request_points(ds_sub_reference, ds_reference_pnt, buff)
    for i in range(sub_voxel_points.shape[0]):
        sub_vox_pnt = []
        l_weights = []
        for data in range(len(super_voxel_points)):
            points, noise = request_points(
                super_voxel_points[data], sub_voxel_points[i], buff * 0.5
            )
            weight_l = points.shape[0] * noise
            l_weights.append(weight_l)
            sub_vox_pnt.append(points)
        weights = np.array(np.array(g_weights) * np.array(l_weights))
        weights = weights / np.sum(weights)
        point = sub_vox_pnt[np.argmax(weights)]
        # print(np.argmax(weights))
        point_cont[np.argmax(weights)] = point_cont[np.argmax(weights)] + 1

        if check_points_distribution(point, sub_voxel_points[i], threshold, buff):
            point = np.vstack(sub_vox_pnt)
            point_cont = point_cont + 1
        fused_sub_point = np.concatenate(
            (fused_sub_point, point.reshape(-1, 6)), axis=0
        )
    return fused_sub_point[1:, :], point_cont


def weighted_fusion_filter(
    data_list, g_weight, reference_pc, sub_reference_pc, voxel_size, k_1, k_2, threshold
):
    point_cont = np.zeros(len(data_list))
    fused_points = np.zeros((1, 6))
    buff = (voxel_size + voxel_size * 0.1) / 2
    for i in tqdm.tqdm(range(reference_pc.shape[0])):
        # for i in range(6):
        points = []
        l_weights = []
        for data in range(len(data_list)):
            pnts, noise_point = request_points(data_list[data], reference_pc[i], buff)
            points.append(pnts)
            weight = (pnts.shape[0] / (voxel_size * 10) ** 3) * noise_point
            l_weights.append(weight)
        l_weights = l_weights / np.sum(l_weights)
        weights = (k_1 * np.array(g_weight)) + (k_2 * np.array(l_weights))
        weights = weights / np.sum(weights)
        point = points[np.argmax(weights)]

        if check_points_distribution(point, reference_pc[i], threshold, voxel_size):
            point, point_cont_ = adjusted_voxel2(
                sub_reference_pc,
                reference_pc[i],
                points,
                g_weight,
                buff,
                0.5 * threshold,
                point_cont,
            )
        else:
            point_cont[np.argmax(weights)] = point_cont[np.argmax(weights)] + 1

        # point =  opend3d_outlier(point,option="statistical")
        fused_points = np.concatenate((fused_points, point.reshape(-1, 6)), axis=0)
    return fused_points[1:, :], point_cont_
