# Analysis functions for evoked responses

from mimo_pack.analysis.lfp import evoked_lfp
from mimo_pack.analysis.ap import evoked_ap
import inspect

def evoked_field_summary(times, ap_path, lf_path, **kwargs):
    """Calculate and summarize evoked AP and LFP responses around specified event times.

    This function computes the evoked responses for both the action potential (AP)
    and local field potential (LFP) bands by aligning data to a given set of
    event times. It then returns a summary of these responses.

    Parameters
    ----------
    times : array_like
        A sequence of event times in seconds to align the data to.
    ap_path : str or pathlib.Path
        Path to the binary file containing the action potential (AP) band data.
    lf_path : str or pathlib.Path
        Path to the binary file containing the local field potential (LFP) band data.

    Optional
    --------
    **kwargs
        Additional keyword arguments to pass to the evoked_lfp and evoked_ap functions.
        
    Returns
    -------
    dict
        A dictionary containing the summary of evoked responses with the following keys:
        - 'ap' : The evoked action potential envelope.
        - 'lfp' : The mean evoked local field potential.
        - 'csd' : The mean evoked current source density.

    See Also
    --------
    evoked_lfp : Function to compute evoked LFP and CSD.
    evoked_ap : Function to compute evoked AP envelope.

    """
    # Get the expected keyword arguments for each function
    lfp_params = inspect.signature(evoked_lfp).parameters
    ap_params = inspect.signature(evoked_ap).parameters

    # Filter kwargs for each function
    lfp_kwargs = {k: v for k, v in kwargs.items() if k in lfp_params}
    ap_kwargs = {k: v for k, v in kwargs.items() if k in ap_params}

    # Call the functions with the filtered kwargs
    lfp_mean, csd_mean = evoked_lfp(lf_path, times, **lfp_kwargs)
    ap_env = evoked_ap(ap_path, times, **ap_kwargs)
    return {'ap': ap_env, 'lfp': lfp_mean, 'csd': csd_mean}