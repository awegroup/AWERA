import numpy as np

# Generator 100kW GS (?)

load_steps = [10, 25, 50, 75, 100, 125]  # %
freq_steps = [10, 20, 30, 40, 50, 80, 100]  # Hz
efficiency_by_frequency_load = [
    # Load / Frequency
    # 10% 25% 50% 75% 100% 125%
    [73.1, 85.9, 89.3, 88.6, 86.5, 83.4],  # 10Hz
    [81.3, 90.7, 93.4, 93.4, 92.6, 91.3],  # 20Hz
    [83.9, 92.3, 94.7, 94.9, 94.5, 93.7],  # 30Hz
    [85.0, 92.9, 95.3, 95.6, 95.3, 94.7],  # 40Hz
    [85.4, 93.1, 95.6, 96.0, 95.6, 95.3],  # 50Hz
    [90.8, 95.4, 96.2, 95.8, 94.8, 93.2],  # 80Hz
    [91.5, 95.5, 96.0, 95.2, 93.4, 0],  # 100Hz
    ]


def get_frequency_from_reeling_speed(vr,
                                     gear_ratio=10,
                                     r_drum=0.105):
    """
    Get Generator frequency from tether reeling speed.

    Parameters
    ----------
    vr : float
        Tether reeling speed in m/s.
    gear_ratio : float, optional
        Gear conversion from input to output angular reeling velocity.
        The default is gear_ratio.
    r_drum : float, optional
        Drum radius of the generator in meter, converting reeling speed
        to angular velocity.
        Soure: 53.5 kW upscaled system AWE book ch14 Fechner:
        Fechner, Uwe & Schmehl, Roland. (2013). Model-Based Efficiency Analysis of Wind Power Conversion by a Pumping Kite Power System. 10.1007/978-3-642-39965-7_14.
        The default is r_drum.

    Returns
    -------
    freq : float
        Generator frequency resulting from given reeling speed, in Hz.

    """
    vr = np.abs(vr)  # no gear switch for reel-in/out
    freq = gear_ratio/(2*np.pi * r_drum)

    return freq


def get_gen_eff(power, vr,
                rated_power=100,
                load_steps=load_steps, freq_steps=freq_steps,
                efficiency_by_frequency_load=efficiency_by_frequency_load):
    """
    Interpolate the efficiency from the efficiency table by load and frequency.

    Parameters
    ----------
    power : Float
        Power at generator.
        Impacts efficiency relative to generator rated power.
    vr : Float
        Tehter reeling speed in m/s.
    load_steps : list(Float), optional
        Load values described in the efficiency table.
        The default is load_steps.
    freq_steps : list(Float), optional
        Frequency values described in the efficiency table.
        The default is freq_steps.
    efficiency_by_frequency_load : list, optional
        List of efficiencies by generator load for varying
        generator frequencies. The default is efficiency_by_frequency_load.

    Returns
    -------
    eff : Float
        Interpolated efficiency for power and reeling speed setting.

    """
    from scipy import interpolate
    eff_table = interpolate.interp2d(load_steps, freq_steps,
                                     efficiency_by_frequency_load,
                                     bounds_error=False,
                                     fill_value=0.)
    load = power/rated_power
    freq = get_frequency_from_reeling_speed(vr)
    eff = eff_table(load, freq)

    return eff


def get_winch_eff(eff_pump, eff_o, eff_i, eff_batt=0.95):
    """
    Calculate the total winch efficiency for a given pumping cycle.

    Parameters
    ----------
    eff_pump : Float
        Pumping efficiency: Total Energy divided by reel-out energy.
    eff_o : Float
        Generator efficiency for reel-out.
    eff_i : Float
        Generator efficiency for reel-in.
    eff_batt : Float, optional
        Battery efficiency, impacting the used power during reel-in.
        The default is 0.95.

    Returns
    -------
    eff : Float
        Total winch efficiency as defined in
        Fechner, Uwe & Schmehl, Roland. (2013). Model-Based Efficiency Analysis of Wind Power Conversion by a Pumping Kite Power System. 10.1007/978-3-642-39965-7_14.

    """
    eff = (eff_o * eff_i * eff_batt - 1 + eff_pump)/(
        eff_i * eff_batt * eff_pump)
    return eff
