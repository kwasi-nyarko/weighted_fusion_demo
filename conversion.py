import numpy as np
import open3d as o3d


def np_point_cloud2_pcd(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])
    pcd.estimate_normals(fast_normal_computation=False)
    return pcd


def pcd2np_point_cloud(pcd):
    pc_down_point = np.asarray(pcd.points)
    pc_down_color = np.asarray(pcd.colors)
    np_point_Cloud = np.concatenate((pc_down_point, pc_down_color), axis=1)
    return np_point_Cloud
