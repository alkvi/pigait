"""
    Useful filtering methods
"""

import copy
import scipy

from ..plotting import plot_signal


def filter_signal_butter(data, low, order, plot_filter=False):
    """
    Low-passes data for every sensor type in supplied data object

    Parameters
    ----------
    data : IMUData
        Data containing IMU raw data
    low : float
        Cut-off frequency
    order : string
        Order for butterworth filter
    plot_filter : bool
        Plot the signal with comparison before/after
    """

    # Very short signals cannot be padded
    if data.gyro_data.shape[0] < 15:
        print("WARN: not filtering, signal too short")
        print(data.gyro_data.shape)
        return data

    fn = data.fs / 2
    b, a = scipy.signal.butter(order, low / fn, 'low')
    gyro_data_filt = scipy.signal.filtfilt(b, a, data.gyro_data, axis=0)
    acc_data_filt = scipy.signal.filtfilt(b, a, data.acc_data, axis=0)

    if data.mag_data is not None:
        mag_data_filt = scipy.signal.filtfilt(b, a, data.mag_data, axis=0)
    else:
        mag_data_filt = None

    if plot_filter:
        plot_signal.plot_filter(data.gyro_data, data.acc_data, data.mag_data,
                                gyro_data_filt, acc_data_filt, mag_data_filt)

    # return a copy with filtered data
    data_copy = copy.deepcopy(data)
    data_copy.acc_data = acc_data_filt
    data_copy.gyro_data = gyro_data_filt
    data_copy.mag_data = mag_data_filt

    return data_copy
