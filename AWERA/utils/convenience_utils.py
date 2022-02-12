def write_timing_info(info, time_elapsed):
    print('{} - Time lapsed: \t{:.0f}m {:.0f}s'
          .format(info, time_elapsed // 60, time_elapsed % 60))
