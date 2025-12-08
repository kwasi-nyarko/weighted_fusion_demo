# %%
# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
import time
from datafusion import SessionContext, col, lit, functions as f, str_lit

from data import U16MAX
from expr import voxel_group_by_expr, COLORS, COORDS


VOXEL_SIZE = 0.05
OUTPUT_DIR = "output"


POINT_CLOUDS = ["assets/bun000_Cloud.parquet", "assets/bun045_Cloud.parquet"]
POINT_CLOUDS_ACCURACY = [0.01, 0.01]
OUTPUT_FILE_NAME = f"{OUTPUT_DIR}/bunny_fused_scale.las"

# POINT_CLOUDS = [
#     "data/Vidalaga/drone.parquet",
#     "data/Vidalaga/p20.parquet",
#     "data/Vidalaga/vlx.parquet",
# ]
# POINT_CLOUDS_ACCURACY = [0.1, 0.005, 0.01]
# OUTPUT_FILE_NAME = f"{OUTPUT_DIR}/vidalaga_fused_scale.las"

# POINT_CLOUDS = [
#     "data/BridgeAchen/Brucke_Aachen_Drohne.parquet",
#     "data/BridgeAchen/Brucke_Aachen_P20.parquet",
#     "data/BridgeAchen/Brucke_Aachen_VLX2.parquet",
# ]
# POINT_CLOUDS_ACCURACY = [0.1, 0.005, 0.01]


print("Load and downsample point clouds ... ")
start = time.time()

ctx = SessionContext()

# parts
point_clouds_df = []
colors_select = [
    (f.arrow_cast(col(c), str_lit("Float64")) / lit(float(U16MAX))).alias(c)
    for c in COLORS
]
for i, pc in enumerate(POINT_CLOUDS):
    name = f"pc{i}"
    ctx.register_parquet(name, pc)
    df = ctx.table(name).select(*COORDS, *colors_select)
    point_clouds_df.append(df)

# merged
reference_df = None
for pc_df in point_clouds_df:
    if reference_df is None:
        reference_df = pc_df
    else:
        reference_df = reference_df.union(pc_df)

# voxelization
reference_df_ds = reference_df.aggregate(
    group_by=voxel_group_by_expr(VOXEL_SIZE),
    aggs=[f.count_star().alias("count")],
)

print("Number of points in search space (voxels): ", reference_df_ds.count())
print(f"Time to prepare point clouds: {time.time() - start:.3}s")

# %%
# -----------------------------------------------------------------------------
# Weights
# -----------------------------------------------------------------------------
from weights_df import compute_weights_df


# global weights
print("Computing point cloud weights ... ")
start = time.time()

global_weights = compute_weights_df(
    reference_df,
    point_clouds_df,
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
from fusion_df import weighted_fusion_filter_df

THRESHOLD = 0.01
K_GLOBAL = 1
K_LOCAL = 1

print("Fuse point clouds ... ")
start = time.time()

fused_pcd = weighted_fusion_filter_df(
    point_clouds_df,
    global_weights,
    reference_df,
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
from data import write_las_file


write_las_file(fused_pc, OUTPUT_FILE_NAME)

# %%
