import xarray as xr

import magnify.registry as registry


@registry.component("drop")
def drop(
    xp: xr.Dataset,
    roi_only: bool = False,
    drop_tiles: bool = True,
):
    if roi_only:
        return xp.roi.assign_attrs(xp.attrs)
    elif drop_tiles:
        return xp.drop_vars(["tile", "tile_row", "tile_col"], errors="ignore")
    else:
        return xp


@registry.component("restore_format")
def restore_format(xp: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    xp = xp.unstack()

    if "__time__" in xp.dims:
        # Restore the time coordinate to its original name if it got changed to not clash.
        xp = xp.rename({"__time__": "time"})

    # Remove any dimensions that got added by standardize_format.
    standard_dims = ["channel", "time", "tile_row", "tile_col", "tile_y", "tile_x"]
    for dim in standard_dims:
        if dim not in xp.__original_tile_dims__ and dim in xp.dims:
            xp = xp.squeeze(dim)

    # Restore the original dimension order in the context of dimensions that didn't exist before.
    for name, var in xp.variables.items():
        original_dims = [dim for dim in xp.__original_tile_dims__ if dim in var.dims]
        # If there are no original dimensions in this variable then don't do anything.
        if len(original_dims) > 0:
            var_dims = list(var.dims)
            # Find the indices of original dimensions in the current variable dimensions.
            idxs = [i for i, x in enumerate(var_dims) if x in original_dims]
            # Permute the original dimensions to match their order in __original_tile_dims__.
            dim_order = list(var_dims)
            for idx, dim in zip(idxs, original_dims, strict=True):
                dim_order[idx] = dim
            xp[name] = var.transpose(*dim_order)

    del xp.attrs["__original_tile_dims__"]

    return xp
