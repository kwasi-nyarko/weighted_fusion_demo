import laspy
import pye57
import open3d as o3d
import numpy as np


def load_pointCloud(pc_path):
    pc_ext = pc_path.split(".")[-1]
    if pc_ext == "e57":
        np_pointCloud = load_pye57_file(pc_path)
    elif pc_ext == "las" or pc_ext == "laz":
        np_pointCloud = load_las_file(pc_path)
    elif pc_ext == "ply":
        np_pointCloud = load_ply_file(pc_path)
    else:
        print(
            "ERROR: Check the file extension of the file. \n Only accepts *.ply, *.las/*.laz and *.e57En"
        )
    return np_pointCloud


def load_las_file(path, intensity=False):
    """
    convert laspy.lasdata.LasData into a np array
    """
    with laspy.open(path) as las_path:
        las = las_path.read()

    points = np.stack((np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)), axis=1)
    try:
        colors = np.stack(
            (np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)), axis=1
        )
        if intensity:
            intens = np.asarray(las.intensity)
            pc_np = np.concatenate((points, colors, intens), axis=1)
        pc_np = np.concatenate((points, colors), axis=1)
        return pc_np
    except KeyError:
        return points


def load_ply_file(path, color=True, estimate_normals=True):
    pcd = o3d.io.read_point_cloud(path)
    points = pcd.points
    point_cloud = np.array(points)
    if color:
        colors = pcd.colors
        point_cloud = np.concatenate((points, colors), axis=1)

    if estimate_normals:
        pcd.estimate_normals(fast_normal_computation=False)
        normals = pcd.normals
        point_cloud = np.concatenate((points, colors, normals), axis=1)
    return point_cloud


def load_pye57_file(path, intensity=False):
    """
    input:
        path (String): directory to file
        intensity (Boolean): intensity measurement from pointcloud

    output:
        pointcloud (numpy array): point cloud in a numpy array
    """
    e57 = pye57.E57(path)
    data = e57.read_scan(
        0, colors=True, intensity=intensity, ignore_missing_fields=True
    )

    points = np.array([data["cartesianX"], data["cartesianY"], data["cartesianZ"]]).T
    colors = np.array([data["colorRed"], data["colorGreen"], data["colorBlue"]]).T / 255

    # return a numpy array
    if intensity:
        intensity = np.array([data["intensity"]]).T
        point_cloud = np.concatenate((points, colors, intensity), axis=1)

    point_cloud = np.concatenate((points, colors), axis=1)
    return point_cloud


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
    las.red = pc_np[:, 3] * 255
    las.green = pc_np[:, 4] * 255
    las.blue = pc_np[:, 5] * 255

    if pc_np.shape[1] == 7:
        las.intensity = pc_np[:, 6]

    las.write(las_path)
