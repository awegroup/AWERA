"""Utility functions."""
import datetime as dt
from pytz import utc
import numpy as np

datetime0 = dt.datetime(1900, 1, 1, 0, 0, 0, tzinfo=utc)  # Date used as starting point for counting hours.


def write_timing_info(info, time_elapsed):
    print('{} - Time lapsed: \t{:.0f}m {:.0f}s'
          .format(info, time_elapsed // 60, time_elapsed % 60))


def hour_to_date_str(hour, str_format=None):
    """Convert hour since 1900-01-01 00:00 to string of date.

    Args:
        hour (int): Hour since 1900-01-01 00:00.
        str_format (str, optional): Explicit format string from datetime packages. Defaults to isoformat.

    Returns:
        str: String representing the timestamp.

    """
    date = hour_to_date(hour)
    if str_format is None:
        return date.isoformat()
    else:
        return date.strftime(str_format)


def hour_to_date(hour):
    """Convert hour since 1900-01-01 00:00 to datetime object.

    Args:
        hour (int): Hour since 1900-01-01 00:00.

    Returns:
        datetime: Datetime object of timestamp.

    """
    date = (datetime0 + dt.timedelta(hours=int(hour)))
    return date