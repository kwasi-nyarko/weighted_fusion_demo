from datafusion import Expr, col, lit, functions as f, str_lit

COORDS = ["x", "y", "z"]
COLORS = ["red", "green", "blue"]
VOXELS = ["vx", "vy", "vz"]


def voxel_group_by_expr(voxel_size: float) -> list[Expr]:
    return [
        f.arrow_cast(f.floor(col(c) / lit(voxel_size)), str_lit("Int64")).alias(v)
        for c, v in zip(COORDS, VOXELS)
    ]


def voxel_filter_expr(voxel: list[int], size: float, buffer: float = 0.0) -> Expr:
    assert len(voxel) == 3

    xmin = voxel[0] * size - buffer
    xmax = (voxel[0] + 1) * size + buffer
    ymin = voxel[1] * size - buffer
    ymax = (voxel[1] + 1) * size + buffer
    zmin = voxel[2] * size - buffer
    zmax = (voxel[2] + 1) * size + buffer

    return (
        (col(COORDS[0]) >= lit(xmin))
        & (col(COORDS[0]) < lit(xmax))
        & (col(COORDS[1]) >= lit(ymin))
        & (col(COORDS[1]) < lit(ymax))
        & (col(COORDS[2]) >= lit(zmin))
        & (col(COORDS[2]) < lit(zmax))
    )


def point_buffer_filter_expr(point: list[float], buffer: float) -> Expr:
    return (
        (col("x") < lit(point[0] + buffer))
        & (col("x") > lit(point[0] - buffer))
        & (col("y") < lit(point[1] + buffer))
        & (col("y") > lit(point[1] - buffer))
        & (col("z") < lit(point[2] + buffer))
        & (col("z") > lit(point[2] - buffer))
    )
