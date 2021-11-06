import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path

path = Path(__file__).parent
data_dir = "/home/mark/WindData/FGW/"

all_headers = ["Timestamp", "Position", "Temperature", "Wiper Count", "40m CNR (dB)", "40m Radial Wind Speed (m/s)", "40m Radial Wind Speed Dispersion (m/s)", "40m Wind Speed (m/s)", "40m Wind Direction (°)", "40m X-wind (m/s)", "40m Y-wind (m/s)", "40m Z-wind (m/s)", "", "60m CNR (dB)", "60m Radial Wind Speed (m/s)", "60m Radial Wind Speed Dispersion (m/s)", "60m Wind Speed (m/s)", "60m Wind Direction (°)", "60m X-wind (m/s)", "60m Y-wind (m/s)", "60m Z-wind (m/s)", "", "80m CNR (dB)", "80m Radial Wind Speed (m/s)", "80m Radial Wind Speed Dispersion (m/s)", "80m Wind Speed (m/s)", "80m Wind Direction (°)", "80m X-wind (m/s)", "80m Y-wind (m/s)", "80m Z-wind (m/s)", "", "100m CNR (dB)", "100m Radial Wind Speed (m/s)", "100m Radial Wind Speed Dispersion (m/s)", "100m Wind Speed (m/s)", "100m Wind Direction (°)", "100m X-wind (m/s)", "100m Y-wind (m/s)", "100m Z-wind (m/s)", "", "120m CNR (dB)", "120m Radial Wind Speed (m/s)", "120m Radial Wind Speed Dispersion (m/s)", "120m Wind Speed (m/s)", "120m Wind Direction (°)", "120m X-wind (m/s)", "120m Y-wind (m/s)", "120m Z-wind (m/s)", "", "140m CNR (dB)", "140m Radial Wind Speed (m/s)", "140m Radial Wind Speed Dispersion (m/s)", "140m Wind Speed (m/s)", "140m Wind Direction (°)", "140m X-wind (m/s)", "140m Y-wind (m/s)", "140m Z-wind (m/s)", "", "160m CNR (dB)", "160m Radial Wind Speed (m/s)", "160m Radial Wind Speed Dispersion (m/s)", "160m Wind Speed (m/s)", "160m Wind Direction (°)", "160m X-wind (m/s)", "160m Y-wind (m/s)", "160m Z-wind (m/s)", "", "180m CNR (dB)", "180m Radial Wind Speed (m/s)", "180m Radial Wind Speed Dispersion (m/s)", "180m Wind Speed (m/s)", "180m Wind Direction (°)", "180m X-wind (m/s)", "180m Y-wind (m/s)", "180m Z-wind (m/s)", "", "200m CNR (dB)", "200m Radial Wind Speed (m/s)", "200m Radial Wind Speed Dispersion (m/s)", "200m Wind Speed (m/s)", "200m Wind Direction (°)", "200m X-wind (m/s)", "200m Y-wind (m/s)", "200m Z-wind (m/s)"]
alts = np.array([40, 60, 80, 100, 120, 140, 160, 180, 200])
east_cols = ["{}m X-wind (m/s)".format(a) for a in alts]
north_cols = ["{}m Y-wind (m/s)".format(a) for a in alts]
headers = ["Timestamp", "Position"] + [item for sublist in zip(east_cols, north_cols) for item in sublist]
usecols = [i for i, x in enumerate(all_headers) if x in headers]


def read_data(read_raw_data=False):
    if read_raw_data:
        regex = data_dir+"*.rtd"
        daily_files = sorted(glob(regex))

        for i, f in enumerate(daily_files):
            print(f)
            df = pd.read_csv(f, skiprows=41, encoding="latin_1", sep='\t', error_bad_lines=False, usecols=usecols, names=headers, header=0, parse_dates=[0], index_col=False, na_values={'Position': 'V'})  #, dtype=dtypes)

            df = df[df.Position.notnull()]  # Ignoring the observations of the vertical laser beam position

            cols_with_errors = df.columns[df.dtypes.eq('object')]  # All columns should be numeric - if not; parsing has failed.

            # Ignore rows for which parsing failed
            mask_nan_before = df[cols_with_errors].apply(pd.notnull)
            df[cols_with_errors] = df[cols_with_errors].apply(pd.to_numeric, errors='coerce')
            mask_nan_after = df[cols_with_errors].apply(pd.notnull)
            mask_error = (mask_nan_before != mask_nan_after).any(axis=1)
            print(np.sum(mask_error), "line(s) skipped")
            df = df[~mask_error]

            df_downsampled = df.resample('H', on='Timestamp').mean()  # Ignores NaN's
            # if f.endswith('WLS7-573_2020_01_08__00_00_00.rtd'):
            #     import matplotlib.pyplot as plt
            #     plt.plot(df['Timestamp'], df['200m X-wind (m/s)'].values)
            #     plt.plot(df_downsampled.index, df_downsampled['200m X-wind (m/s)'].values)
            #     plt.show()
            df_downsampled = df_downsampled.dropna()  # Remove rows that don't have values for all fields

            if i == 0:
                df_complete = df_downsampled
            else:
                df_complete = df_complete.append(df_downsampled)

        df_complete.to_csv(path / "downsampled_fgw_lidar_data.csv")
    else:
        df_complete = pd.read_csv(path / "downsampled_fgw_lidar_data.csv", parse_dates=[0], index_col='Timestamp')

    dts = df_complete.index.values
    res = {
        'wind_speed_east': df_complete[east_cols].values,
        'wind_speed_north': df_complete[north_cols].values,
        'n_samples': len(df_complete),
        'datetime': dts,
        'altitude': alts,
        'years': (dts[0].astype('datetime64[Y]').astype(int)+1970, dts[-1].astype('datetime64[Y]').astype(int)+1970),
    }

    return res


if __name__ == '__main__':
    data = read_data(read_raw_data=True)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)
    lines = ax[0].plot(data['datetime'], data['wind_speed_east'])
    for l, alt in zip(lines, data['altitude']): l.set_label('{} m'.format(alt))
    ax[0].legend()
    ax[0].set_ylabel("Wind speed east [m/s]")
    ax[1].plot(data['datetime'], data['wind_speed_north'])
    ax[1].set_ylabel("Wind speed north [m/s]")
    for a in ax: a.grid()
    plt.show()
