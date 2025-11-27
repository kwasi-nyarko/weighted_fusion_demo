import laspy
import pye57
import open3d as o3d
import numpy as np


U16MAX = np.iinfo(np.uint16).max


def load_point_cloud(pc_path):
    """
    Load a point cloud from a PLY, LAS/LAZ or E57 file.

    Returns a numpy array of shape (n, 6) with x, y, z, and the normalized
    colors red, green, blue.
    """
    pc_ext = pc_path.split(".")[-1].lower()
    if pc_ext == "e57":
        np_pc = load_pye57_file(pc_path)
    elif pc_ext == "las" or pc_ext == "laz":
        np_pc = load_las_file(pc_path)
    elif pc_ext == "ply":
        np_pc = load_ply_file(pc_path)
    else:
        raise IOError("Only PLY, LAS/LAZ and E57 files supported")
    return np_pc


def load_las_file(path, color=True, intensity=False):
    with laspy.open(path) as las:
        data = las.read()

        points = np.stack(
            (
                np.asarray(data.x),
                np.asarray(data.y),
                np.asarray(data.z),
            ),
            axis=1,
        )

        if not color:
            np_pc = points
        else:
            colors = np.stack(
                (
                    np.asarray(data.red),
                    np.asarray(data.green),
                    np.asarray(data.blue),
                ),
                axis=1,
            )
            colors = colors.astype(np.float64) / U16MAX  # normalize colors to [0,1]

            if not intensity:
                np_pc = np.concatenate((points, colors), axis=1)
            else:
                intensity = np.asarray(data.intensity)
                np_pc = np.concatenate((points, colors, intensity), axis=1)

    return np_pc


def load_ply_file(path, color=True, estimate_normals=False):
    pcd = o3d.io.read_point_cloud(path)
    points = pcd.points

    if not color:
        np_pc = np.array(points)
    else:
        colors = pcd.colors
        if estimate_normals:
            pcd.estimate_normals(fast_normal_computation=False)
            normals = pcd.normals
            np_pc = np.concatenate((points, colors, normals), axis=1)
        else:
            np_pc = np.concatenate((points, colors), axis=1)

    return np_pc


def load_pye57_file(path, color=True, intensity=False):
    """
    input:
        path (String): directory to file
        intensity (Boolean): intensity measurement from pointcloud

    output:
        pointcloud (numpy array): point cloud in a numpy array
    """
    e57 = pye57.E57(path)
    data = e57.read_scan(
        0,
        colors=color,
        intensity=intensity,
        ignore_missing_fields=True,
    )

    points = np.array([data["cartesianX"], data["cartesianY"], data["cartesianZ"]]).T

    if not color:
        np_pc = points
    else:

        colors = np.array([data["colorRed"], data["colorGreen"], data["colorBlue"]]).T
        colors = colors.astype(np.float64) / 255  # normalize colors to [0,1]

        if not intensity:
            np_pc = np.concatenate((points, colors), axis=1)
        else:
            intensity = np.array([data["intensity"]]).T
            np_pc = np.concatenate((points, colors, intensity), axis=1)

    return np_pc


def write_las_file(pc_np, las_path):
    header = laspy.LasHeader(point_format=2)
    xmin = np.floor(np.min(pc_np[:, 0]))
    ymin = np.floor(np.min(pc_np[:, 1]))
    zmin = np.floor(np.min(pc_np[:, 2]))
    header.offset = [xmin, ymin, zmin]
    header.scale = [0.001, 0.001, 0.001]

    las = laspy.LasData(header)
    las.x = pc_np[:, 0]
    las.y = pc_np[:, 1]
    las.z = pc_np[:, 2]
    las.red = pc_np[:, 3] * U16MAX
    las.green = pc_np[:, 4] * U16MAX
    las.blue = pc_np[:, 5] * U16MAX

    if pc_np.shape[1] == 7:
        las.intensity = pc_np[:, 6]

    las.write(las_path)


def np_point_cloud2_pcd(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])
    pcd.estimate_normals(fast_normal_computation=False)
    return pcd


def pcd2np_point_cloud(pcd):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return np.concatenate((points, colors), axis=1)
