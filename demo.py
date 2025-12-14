# %%
# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
import time
import open3d as o3d

from data import load_point_cloud, convert_np2pcd

POINT_CLOUDS = ["assets/bun000_Cloud.las", "assets/bun045_Cloud.las"]
# POINT_CLOUDS = [
#     "data/Vidalaga/drone.las",
#     "data/Vidalaga/p20.las",
#     "data/Vidalaga/vlx.las",
# ]
VOXEL_SIZE = 0.05

print("Load and downsample point clouds ... ")
start = time.time()

# load
point_clouds_pcd = [convert_np2pcd(load_point_cloud(pc)) for pc in POINT_CLOUDS]

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
print(f"Time to prepare point clouds: {time.time() - start:.3}s")

# %%
# -----------------------------------------------------------------------------
# Weights
# -----------------------------------------------------------------------------
from weights import compute_weights

POINT_CLOUDS_ACCURACY = [0.01, 0.01]
# POINT_CLOUDS_ACCURACY = [0.1, 0.005, 0.01]

# global weights
print("Computing point cloud weights ... ")
start = time.time()

(global_weights, completeness, rmse) = compute_weights(
    reference_pcd,
    point_clouds_pcd,
    POINT_CLOUDS_ACCURACY,
    voxel_size=VOXEL_SIZE,
)

print("Global Weights:", global_weights)
print(f"Time to calculate weights: {time.time() - start:.3}s")

# %%
# -----------------------------------------------------------------------------
# Fusion
# -----------------------------------------------------------------------------
import numpy as np

from data import convert_pcd2np
from fusion import weighted_fusion_filter

THRESHOLD = 0.01
K_GLOBAL = 1
K_LOCAL = 1

print("Fuse point clouds ... ")
start = time.time()

reference_points = np.asarray(reference_pcd_ds.points)

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
fused_pc = np.unique(convert_pcd2np(fused_pcd), axis=0)
print(f"Time to fuse point clouds: {time.time() - start:.3}s")

# %%
# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
import os

from data import write_las_file

OUTPUT_DIR = "output"

file_name = "bunny_fused.las"
# file_name = "vidalaga_fused.las"
write_las_file(fused_pc, os.path.join(OUTPUT_DIR, file_name))

# %%
