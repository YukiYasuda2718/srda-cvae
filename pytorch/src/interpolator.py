import numpy as np
import xarray as xr
from scipy.interpolate import interp2d


def interpolate_dataarray_2d_gauss_jet(
    da: xr.DataArray,
    nx: int,
    ny: int,
    *,
    min_x: float = 0.0,
    max_x: float = 2 * np.pi,
    min_y: float = -np.pi / 2,
    max_y: float = np.pi / 2,
) -> np.ndarray:

    assert da.ndim == 2, "Input is not 2D dataarray."

    y_name, x_name = da.dims
    itp = interp2d(da[x_name].values, da[y_name].values, da.values)

    x = np.linspace(min_x, max_x, nx, endpoint=False)
    y = np.linspace(min_y, max_y, ny, endpoint=True)

    return itp(x, y)


def interpolate_dataset_2d_gauss_jet(
    ds: xr.Dataset, nx: int, ny: int, var_names: list = ["u", "v", "ω"]
) -> np.ndarray:

    time_series = []
    for it in range(len(ds["time"])):
        ary = []
        for var_name in var_names:
            da = ds[var_name].isel(time=it, zC=0)  # 2D data, so zC is always 0
            itp = interpolate_dataarray_2d_gauss_jet(da, nx, ny)
            ary.append(itp)
        time_series.append(np.stack(ary))

    return np.stack(time_series)