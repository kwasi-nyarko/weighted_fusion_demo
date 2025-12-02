# %%
# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
from datafusion import SessionContext, col, lit, functions as f, str_lit

from data import U16MAX
from expr import voxel_group_by_expr


POINT_CLOUDS = ["assets/bun000_Cloud.parquet", "assets/bun045_Cloud.parquet"]
VOXEL_SIZE = 0.05

# register
ctx = SessionContext()

coords = ["x", "y", "z"]
colors = ["red", "green", "blue"]

colors_select = [
    (f.arrow_cast(col(c), str_lit("Float64")) / lit(float(U16MAX))).alias(c)
    for c in colors
]


point_clouds_df = []
for i, pc in enumerate(POINT_CLOUDS):
    name = f"pc{i}"
    ctx.register_parquet(name, pc)
    df = ctx.table(name).select(*coords, *colors_select)
    point_clouds_df.append(df)

# merged
ctx.register_parquet("reference", "assets")
reference_df = ctx.table("reference").select(*coords, *colors_select)

# downsampling (voxelization)
reference_df_ds = reference_df.aggregate(
    group_by=voxel_group_by_expr(VOXEL_SIZE),
    aggs=[f.mean(col(column)).alias(column) for column in [*coords, *colors]],
)

reference_df_ds_sub = reference_df.aggregate(
    group_by=voxel_group_by_expr(VOXEL_SIZE / 2),
    aggs=[f.mean(col(column)).alias(column) for column in [*coords, *colors]],
)

print("Number of points in search space (voxels): ", reference_df_ds.count())


# %%
# -----------------------------------------------------------------------------
# Weights
# -----------------------------------------------------------------------------

from weights_df import compute_weights_df

POINT_CLOUDS_ACCURACY = [0.01, 0.01]
VOXEL_SIZE = 0.05

# global weights
print("Computing point cloud weights ... ")


global_weights = compute_weights_df(
    reference_df,
    point_clouds_df,
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
from fusion_df import weighted_fusion_filter_df

THRESHOLD = 0.01
K_GLOBAL = 1
K_LOCAL = 1


reference_points = []
for rb in reference_df_ds.select(*coords).collect():
    points = rb.to_tensor().to_numpy()
    reference_points.append(points)
reference_points = np.vstack(reference_points)


fused_pcd = weighted_fusion_filter_df(
    point_clouds_df,
    global_weights,
    reference_points,
    reference_df_ds_sub,
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

file_name = "weighted_fused_filtered_{}_cm_vox_scale.las".format(int(VOXEL_SIZE * 100))
write_las_file(fused_pc, os.path.join(OUTPUT_DIR, file_name))

# %%
