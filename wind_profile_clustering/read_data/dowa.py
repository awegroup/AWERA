import numpy as np
import xarray as xr
import pickle
from pathlib import Path

path = Path(__file__).parent


with open(path / 'dowa_grid.pickle', 'rb') as f:
    try:
        dowa_grid = pickle.load(f, encoding="latin1")
    except:
        dowa_grid = pickle.load(f)

lats_dowa_grid = dowa_grid['latitudes']
lons_dowa_grid = dowa_grid['longitudes']


def determine_grid_point(lat, lon, point='ur'):
    for i_loc, (lon_grid_point, lat_grid_point) in enumerate(zip(lons_dowa_grid.reshape(-1), lats_dowa_grid.reshape(-1))):
        if lat_grid_point > lat and lon_grid_point > lon:
            break
    else:
        raise ValueError

    n_lons = lons_dowa_grid.shape[1]
    i_lat = i_loc // n_lons
    i_lon = i_loc % n_lons

    if point[0] == 'l' and i_lat > 0:  # Lower grid point w.r.t. true coordinate.
        i_lat -= 1

    if point[1] == 'l' and i_lon > 0:  # Left grid point w.r.t. true coordinate.
        i_lon -= 1

    return i_lat, i_lon


def determine_grid_point_coords(lat, lon, point='ll'):
    i_lat, i_lon = determine_grid_point(lat, lon, point=point)
    return lats_dowa_grid[i_lat, i_lon], lons_dowa_grid[i_lat, i_lon]


def find_closest_dowa_grid_point(lat, lon):
    grid_cell_corners = ['ul', 'ur', 'lr', 'll']
    coords_grid_cell_corners = [determine_grid_point_coords(lat, lon, rp) for rp in grid_cell_corners]

    distances = []
    for a, b in coords_grid_cell_corners:
        distances.append(((a - lat)**2 + (b - lon)**2)**.5)
    closest_corner = grid_cell_corners[np.argmin(distances)]
    i_lat, i_lon = determine_grid_point(lat, lon, point=closest_corner)
    return i_lat, i_lon


def read_netcdf(i_lat, i_lon, data_dir):
    iy, ix = i_lat+1, i_lon+1
    file = '{}DOWA_40h12tg2_fERA5_NETHERLANDS.NL_' \
           'ix{:03d}_iy{:03d}_2008010100-2018010100_v1.0.nc'.format(data_dir, ix, iy)
    ds = xr.open_dataset(file)
    # Variables: Lambert_Conformal, wdir, wspeed, ta (air temperature), p (air pressure), hur (relative humidity)

    altitude = ds['height'].values  # [m above Lowest Astronomical Tide]
    datetime = ds['time'].values

    # years = datetime.astype('datetime64[Y]').astype(int) + 1970
    # mask_years = (start_year <= years) & (years <= final_year)
    # print("{} points found between {} - {}".format(np.sum(mask_years), start_year, final_year))

    wind_dir_deg = ds['wdir'].values[:, :, 0, 0]  # Wind from/upwind direction [deg] CW+ w.r.t. North
    wind_speed = ds['wspeed'].values[:, :, 0, 0]  # [m/s]

    wind_dir = -(wind_dir_deg + 90.) * np.pi/180.  # Downwind direction [rad] CCW+ w.r.t. East

    vw_east = np.cos(wind_dir)*wind_speed
    vw_north = np.sin(wind_dir)*wind_speed

    ds.close()  # Close the input file.

    return vw_east, vw_north, datetime, altitude


def read_data(grid_points={'coords': (52.85, 3.44)}, data_dir='DOWA/'):
    if 'coords' in grid_points:
        k, l = find_closest_dowa_grid_point(*grid_points['coords'])
        vw_east, vw_north, dts, alts = read_netcdf(k, l, data_dir)
    elif 'i_lat' in grid_points and 'i_lon' in grid_points:
        k, l = grid_points['i_lat'], grid_points['i_lon']
        vw_east, vw_north, dts, alts = read_netcdf(k, l, data_dir)
    elif 'iy' in grid_points and 'ix' in grid_points:
        vw_east, vw_north, dts, alts = read_netcdf(grid_points['iy']-1, grid_points['ix']-1, data_dir)
    elif 'ids' in grid_points or 'mult_coords' in grid_points:  # Mulitple locations.
        if 'ids' in grid_points:
            i_lats, i_lons = grid_points['ids'][0], grid_points['ids'][1]
        elif 'mult_coords' in grid_points:  # Mulitple locations given as list of [(lat,lon), (lat1,lon1)].
            i_lats, i_lons = [], []
            for loc in grid_points['mult_coords']:
                i_lat, i_lon = find_closest_dowa_grid_point(*loc)
                i_lats.append(i_lat)
                i_lons.append(i_lon)
        n_locs = len(i_lats)

        first_iter = True
        for i_loc, (i_lat, i_lon) in enumerate(zip(i_lats, i_lons)):
            vwe, vwn, dts, alts = read_netcdf(i_lat, i_lon, data_dir)
            if first_iter:
                n_alts = len(alts)
                vw_east = np.zeros((len(dts), n_locs, n_alts))
                vw_north = np.zeros((len(dts), n_locs, n_alts))
                first_iter = False
                dts0 = dts
            vw_east[:, i_loc, :] = vwe
            vw_north[:, i_loc, :] = vwn
            assert np.all(dts == dts0)
        vw_east = vw_east.reshape((-1, n_alts))
        vw_north = vw_north.reshape((-1, n_alts))
        dts = np.repeat(dts, n_locs, axis=0)

    res = {
        'wind_speed_east': vw_east,
        'wind_speed_north': vw_north,
        'n_samples': len(dts),
        'datetime': dts,
        'altitude': alts,
        'years': (dts[0].astype('datetime64[Y]').astype(int)+1970, dts[-1].astype('datetime64[Y]').astype(int)+1970-1),
    }
    return res


if __name__ == '__main__':
    read_data({'iy': 111, 'ix': 56})
