import open3d as o3d
import numpy as np
import tqdm

from input_output import load_point_cloud
from conversion import np_point_cloud2_pcd, pcd2np_point_cloud


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


def load_PC_acc():
    pc_path = str(input("Enter point cloud directory: "))
    accuracy = float(input("Enter the accuracy of the sensor system in meters: "))
    print("Loading point cloud")
    np_pointCloud = load_point_cloud(pc_path)
    print("Done.")
    return np_pointCloud, accuracy, pc_path


def compute_fusion_accuracy(point_cont, lst_acc):
    lst_contrib = point_cont / np.sum(point_cont)
    acc_contrib = lst_contrib * (np.array(lst_acc) ** 2)
    # acc_contrib = acc_contrib**2
    fusion_acc = np.sqrt(np.sum(acc_contrib))
    return lst_contrib, fusion_acc


def compute_surface_area(pcd, voxel_size):
    radii = [voxel_size, voxel_size * 2, voxel_size * 4, voxel_size * 5]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    return rec_mesh.get_surface_area()


def compute_coverage(reference_pcd, pcd, voxel_size):
    ds_ref_pcd = reference_pcd.voxel_down_sample(voxel_size=voxel_size)
    ds_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    coverage = compute_surface_area(ds_pcd, voxel_size) / compute_surface_area(
        ds_ref_pcd, voxel_size
    )
    return coverage


def weight_compute(reference_pcd, pcd, accuracy, voxel_size):
    completeness = compute_coverage(reference_pcd, pcd, voxel_size)
    weight = accuracy / completeness
    return weight


def compute_weights(reference_pc, pc_list, accuracy, voxel_size):
    weights = []
    reference_pcd = np_point_cloud2_pcd(reference_pc)
    for data in range(len(pc_list)):
        data_pcd = np_point_cloud2_pcd(pc_list[data])
        weights.append(
            weight_compute(reference_pcd, data_pcd, accuracy[data], voxel_size * 2)
        )
        data_pcd = []
    if len(weights) > 1:
        weights = np.sum(weights) - weights

    weights_ = weights / np.sum(weights)
    lst_PCs, rmse = compute_rmse(pc_list, accuracy, voxel_size)
    g_weight = weights / (1 - rmse)
    return g_weight
