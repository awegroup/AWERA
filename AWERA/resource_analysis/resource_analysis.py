import numpy as np
import matplotlib.pyplot as plt

from ..wind_profile_clustering.read_requested_data import get_wind_data
from .single_loc_plots import plot_figure_5a

from .plot_maps import plot_all
# TODO import all functions needed
# TODO add processing functionality
# TODO add plot maps single functions

# TODO careful old code: height range from top down

# TODO add stuff to config: and remove standalone config
# # Plots of figure 5 use data from 2016.
# start_year = 2016
# end_year = 2016


class ResourceAnalysis:
    #TODO inherit from config... or as is set config object as config item?

    def __init__(self, config):
        # Set configuration from Config class object
        # TODO yes/no? super().__init__(config)
        setattr(self, 'config', config)

    def single_loc_plot(self,
                        loc=(51.0, 1.0),
                        time_ids=None,
                        ceiling=500,
                        floor=50):
        hours, v_req_alt, v_ceilings, optimal_heights = \
            self.eval_single_location(loc=loc,
                                      time_ids=time_ids,
                                      ceilings=[ceiling],
                                      floor=floor)

        ref_height = self.config.General.ref_height
        v_at_ref_height = v_req_alt[
            :, self.config.Data.height_range.index(ref_height)]
        # TODO SINGLE ceiling selected hard coded -> config
        dates = plot_figure_5a(hours, v_ceilings[:, 0], optimal_heights[:, 0],
                               height_range=None,
                               ref_velocity=v_at_ref_height,
                               height_bounds=[floor, ceiling],
                               v_bounds=[3, 25],  # TODO v_bounds from average over all clusters - not really showing max v_bounds - maybe just use max in the end?
                               show_n_hours=24*7)
        plt.savefig(
            self.config.IO.result_dir
            + 'optimal_harvesting_height_over_time'
            + '_{:.2f}_lat_{:.2f}_lon_{}_time.pdf'
            .format(loc[0], loc[1], dates[0]))
        # plot_figure_5b(hours, v_req_alt, v_ceilings[:, 0], optimal_heights[:, 1], heights_of_interest,
        #                analyzed_heights_ids['ceilings'][1], analyzed_heights_ids['floor'])

        # Plots of figure 6 use data from 2011 until 2017.
        # start_year = 2011
        # end_year = 2017
        # hours, v_req_alt, v_ceilings, optimal_heights = eval_single_location(eval_lat, eval_lon, start_year, end_year)
        # plot_figure_6a(optimal_heights[:, 1])
        # plot_figure_6b(optimal_heights[:, 1])
        # plot_weibull_fixed_and_ceiling(v_req_alt, heights_of_interest, [100., 500., 1500.], v_ceilings[:, 1])  # figure 6c
        # plot_figure_6d(v_ceilings, analyzed_heights['ceilings'])
        # plot_weibull_fixed_and_ceiling(v_req_alt, heights_of_interest, [1500.], v_ceilings[:, 0], 300.)  # figure 6e



    def eval_single_location(self,
                             loc=None,
                             time_ids=None,
                             ceilings=[500],
                             floor=50):
        """"Execute analyses on the data of single grid point.

        Args:
            loc (float, tuple): Latitude, longitude of evaluated grid point.
            start_year (int): Process wind data starting from this year.
            final_year (int): Process wind data up to this year.

        Returns:
            tuple of ndarray: Tuple containing hour timestamps, wind speeds
                at `heights_of_interest`, optimal wind speeds in
                analyzed height ranges, and time series of
                corresponding optimal heights

        """
        # TODO update docstring, include params to config
        if time_ids is not None:
            # TODO change to input real time?
            # TODO use config input sample ids?
            sel_sample_ids = time_ids
        else:
            sel_sample_ids = []
        if loc is None:
            loc = self.config.Data.locations[0]
        # TODO more height range below ceiling!?
        data = get_wind_data(self.config,
                             locs=[loc],
                             sel_sample_ids=sel_sample_ids)

        hours = data['datetime']
        v_req_alt = np.sqrt(data['wind_speed_east']**2
                            + data['wind_speed_north']**2)
        v_ceilings = np.zeros((len(hours), len(ceilings)))
        optimal_heights = np.zeros((len(hours), len(ceilings)))
        ceilings_ids = [self.config.Data.height_range.index(c)
                        for c in ceilings]
        floor_id = self.config.Data.height_range.index(floor)
        for i, ceiling_id in enumerate(ceilings_ids):
            # Find the height maximizing the wind speed for each hour.
            v_ceilings[:, i] = np.amax(v_req_alt[:, floor_id:ceiling_id + 1],
                                       axis=1)
            print(v_ceilings)
            v_ceiling_ids = np.argmax(v_req_alt[:, floor_id:ceiling_id + 1],
                                      axis=1) + floor_id
            print(v_ceiling_ids)
            optimal_heights[:, i] = [self.config.Data.height_range[max_id]
                                     for max_id in v_ceiling_ids]

        return hours, v_req_alt, v_ceilings, optimal_heights

# TODO include processing
    def plot_all_maps(self):
        plot_all()
# TODO include plot_maps

    def height_range_sanity_check(self):

        get_wind_data(self.config, sel_sample_ids=[0])
        self.config.update({
            'Data': {
                'height_range': self.config.Data.height_range_DOWA}
            })
        print('DOWA:')
        get_wind_data(self.config, sel_sample_ids=[0])