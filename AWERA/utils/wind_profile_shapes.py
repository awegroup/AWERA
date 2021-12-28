import pandas as pd
import numpy as np


def export_wind_profile_shapes(heights,
                               u_wind, v_wind,
                               output_file=None,
                               do_scale=True,
                               ref_height=100.):
    """
    From given wind profile return formatted pandas data frame for evaluation.

    Parameters
    ----------
    heights : list
        Height steps of vertical wind profile.
    u_wind : list
        u-component of vertical wind profile wind speed.
    v_wind : list
        v-component of vertical wind profile wind speed.
    output_file : string, optional
        If given, write csv output to this file. The default is None.
    do_scale : Bool, optional
        If True, scale wind profile to 1 at reference height.
        The default is True.
    ref_height : Float, optional
        Reference height where the wind speed is scaled to 1 - if do_scale.
        The default is 100..

    Returns
    -------
    df : pandas DataFrame
        Absolute wind profile shapes and scale factors.

    """
    df = pd.DataFrame({
        'height [m]': heights,
    })
    scale_factors = []
    if len(u_wind.shape) == 1:
        single_profile = True
    else:
        single_profile = False

    for i, (u, v) in enumerate(zip(u_wind, v_wind)):
        if single_profile:
            u, v = u_wind, v_wind
        w = (u**2 + v**2)**.5
        if do_scale:
            # Get normalised wind speed at reference height via linear
            # interpolation
            w_ref = np.interp(ref_height, heights, w)
            # Scaling factor such that the normalised absolute wind speed
            # at the reference height is 1
            sf = 1/w_ref
        else:
            sf = 1.
        dfi = pd.DataFrame({
            'u{} [-]'.format(i+1): u*sf,
            'v{} [-]'.format(i+1): v*sf,
            'scale factor{} [-]'.format(i+1): sf,
        })
        df = pd.concat((df, dfi), axis=1)

        scale_factors.append(sf)

        if single_profile:
            break

    if output_file is not None:
        assert output_file[-4:] == ".csv"
        df.to_csv(output_file, index=False, sep=";")

    return df
