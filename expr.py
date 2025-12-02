from datafusion import col, lit, functions as f, str_lit


def voxel_group_by_expr(voxel_size):
    return [
        f.arrow_cast(f.floor(col(c) / lit(voxel_size / 2)), str_lit("Int64")).alias(
            f"v{c}"
        )
        for c in ["x", "y", "z"]
    ]
