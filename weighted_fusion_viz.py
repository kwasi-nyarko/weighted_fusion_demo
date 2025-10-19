import pyvista as pv
import numpy as np


def viz_pyvista(list_of_pointcloud: list, save_img: bool = False, **kwargs):
    filename = kwargs.get("filename", "filename.png")
    # point_size = kwargs.get("point_size","filename.png")

    point_size = 1

    pl = pv.Plotter()
    pl.background_color = "w"

    sargs = dict(
        title_font_size=20,
        label_font_size=16,
        shadow=True,
        n_labels=3,
        italic=True,
        fmt="%.1f",
        font_family="arial",
        height=0.3,
        vertical=True,
        position_x=0.75,
        position_y=0.2,
    )

    for pc in list_of_pointcloud:
        points = pc[:, :3]
        if pc.shape[1] > 4:
            # rgba = pc[:,3:]/np.max(pc[:,3:],axis=0)
            rgba = pc[:, 3:]
            pl.add_points(
                points,
                scalars=rgba,
                rgba=True,
                point_size=point_size,
                render_points_as_spheres=True,
                smooth_shading=True,
            )

        elif pc.shape[1] == 4:
            scalars = pc[:, -1]
            pl.add_points(
                points,
                cmap="Spectral",
                scalars=scalars,
                point_size=5,
                scalar_bar_args=sargs,
            )
            # pl.add_scalar_bar("Coverage Score",fmt='%10.5f',label_font_size=30,)
        else:
            rgba = np.ones(pc[:, :3].shape)
            point_size = 5
            pl.add_points(
                points,
                scalars=rgba,
                rgba=True,
                point_size=point_size,
                render_points_as_spheres=True,
                smooth_shading=True,
            )

    if save_img:
        # pl.camera.zoom(1.5)
        pl.screenshot(filename)
