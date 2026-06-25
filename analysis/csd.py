# Current source density analyses
import numpy as np
import xarray as xr

def csd_xr(lfp, coord='ch_y', conductivity=1.0):
    """
    Calculates 1D CSD using the Vaknin et al. (1988) method with boundary extrapolation.
    
    Parameters:
    - lfp: xarray DataArray containing potential values (phi).
    - conductivity: float, sigma_z (default 1.0).
    - coord: str, the name of the spatial coordinate dimension.
    """
    
    # Identify the axis index
    grad_dim = np.where(np.array(lfp.dims) == coord)[0][0]
    
    # We pad the data array with the edge values ('edge' mode repeats the first/last value).
    pad_width = [(0, 0)] * lfp.values.ndim
    pad_width[grad_dim] = (1, 1)  # Pad one element on both sides of the spatial axis
    
    phi_padded = np.pad(lfp.values, pad_width, mode='edge')

   
    # calculate second spatial derivative using finite differences
    spacing = lfp.coords[coord].diff(coord).mean().item()  # Average spacing
    first_deriv = np.gradient(phi_padded, spacing, axis=grad_dim)
    second_deriv = np.gradient(first_deriv, spacing, axis=grad_dim)
    
    # cut down to original size
    slices = [slice(None)] * lfp.values.ndim
    slices[grad_dim] = slice(1, -1)
    csd_values = second_deriv[tuple(slices)]

    # scale for sign and conductivity
    csd_values = -1 * conductivity * csd_values

    # Reconstruct xarray
    csd_xr = xr.DataArray(
        csd_values,
        coords=lfp.coords,
        dims=lfp.dims,
        attrs=lfp.attrs
    )
    
    return csd_xr