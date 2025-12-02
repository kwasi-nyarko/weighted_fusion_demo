# %%
# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------

import open3d as o3d

from data import load_point_cloud, np_point_cloud2_pcd

POINT_CLOUDS = ["assets/bun000_Cloud.las", "assets/bun045_Cloud.las"]
VOXEL_SIZE = 0.05

# load
point_clouds_pcd = [np_point_cloud2_pcd(load_point_cloud(pc)) for pc in POINT_CLOUDS]

# merge
reference_pcd = o3d.geometry.PointCloud()
for pcd in point_clouds_pcd:
    reference_pcd.points.extend(pcd.points)
    reference_pcd.colors.extend(pcd.colors)
    reference_pcd.normals.extend(pcd.normals)

# downsampling (voxelization)
reference_pcd_ds = reference_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
reference_pcd_ds_sub = reference_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE / 2)

print("Number of points in search space (voxels): ", len(reference_pcd_ds.points))

# %%
# -----------------------------------------------------------------------------
# Weights
# -----------------------------------------------------------------------------
from weights import compute_weights

POINT_CLOUDS_ACCURACY = [0.01, 0.01]
VOXEL_SIZE = 0.05

# global weights
print("Computing point cloud weights ... ")

global_weights = compute_weights(
    reference_pcd,
    point_clouds_pcd,
    POINT_CLOUDS_ACCURACY,
    voxel_size=VOXEL_SIZE,
)

print("Global Weights:", global_weights)

# %%
# -----------------------------------------------------------------------------
# Fusion
# -----------------------------------------------------------------------------
import numpy as np

from data import pcd2np_point_cloud
from fusion import weighted_fusion_filter

THRESHOLD = 0.01
K_GLOBAL = 1
K_LOCAL = 1
VOXEL_SIZE = 0.05

reference_points = np.asarray(reference_pcd_ds.points)

print("Fuse point clouds ... ")

fused_pcd = weighted_fusion_filter(
    point_clouds_pcd,
    global_weights,
    reference_points,
    reference_pcd_ds_sub,
    VOXEL_SIZE,
    THRESHOLD,
    K_GLOBAL,
    K_LOCAL,
)

# FIXME: this is super inefficient and the fused pcd contains only about 10% unique points
fused_pc = np.unique(pcd2np_point_cloud(fused_pcd), axis=0)


# %%
# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
import os

from data import write_las_file

OUTPUT_DIR = "output"

file_name = "weighted_fused_filtered_{}_cm_vox.las".format(int(VOXEL_SIZE * 100))
write_las_file(fused_pc, os.path.join(OUTPUT_DIR, file_name))

# %%
