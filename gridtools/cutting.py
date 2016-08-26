from .resampling import resample_2d, DS_MEAN, US_LINEAR


def cut_2d(src, out_rect, src_geom=None, ds_method=DS_MEAN, us_method=US_LINEAR, fill_value=None, mode_rank=1):

    if not out_rect:
        return src

    if not src_geom:
        src_geom = {0., 0., 1., -1., 0., 0.}

    _, _, src_cell_size_x, src_cell_size_y, src_ref_cell_x, src_ref_cell_y = src_geom

    out_map_x1, out_map_y1, out_map_x2, out_map_y2 = out_rect
    out_w = int(abs(out_map_x2 - out_map_x1) / abs(src_cell_size_x) + 0.5)
    out_h = int(abs(out_map_y2 - out_map_y1) / abs(src_cell_size_y) + 0.5)
    out_geom = {out_map_x1, out_map_y1, src_cell_size_x, src_cell_size_y, 0, 0}

    # TODO: check if we really need to "resample".
    # Resampling is only required if the the src grid and out grid geometries do not exactly
    # snap onto each other. With other words, if cells are overlapping.

    return resample_2d(src, out_w, out_h,
                       src_geom=src_geom, out_geom=out_geom,
                       ds_method=ds_method, us_method=us_method,
                       fill_value=fill_value, mode_rank=mode_rank, out=None), out_geom
