"""
    Gait event or contact detection functions
"""

import pywt
import scipy
import numpy as np

from ..util import util
from ..data import imu_data
from ..data import event_data
from ..plotting import plot_signal


# Get scale for a wavelet and signal frequency
def _get_scales(wavelet, max_freq, fs):
    fc = pywt.central_frequency(wavelet)
    fa = max_freq
    sampling_period = 1 / fs
    scales = fc / (fa * sampling_period)
    return scales


# Get a wavelet object
def _get_wavelet(wavelet_name):
    discrete_wavelets = ['db2']
    continuous_wavelets = ['gaus1', 'gaus2']
    if wavelet_name in discrete_wavelets:
        wavelet = pywt.Wavelet(wavelet_name)
    elif wavelet_name in continuous_wavelets:
        wavelet = pywt.ContinuousWavelet(wavelet_name)
    else:
        print((f"ERROR: wavelet type {wavelet_name}"
               "not a valid type for this method"))
        raise ValueError
    return wavelet


def _find_dominant_freq(acc_ap_pp, fs):
    signal_len = acc_ap_pp.shape[0]
    half_idx = int(signal_len / 2)
    f_x = scipy.fft.fft(acc_ap_pp) / signal_len
    freqs = np.fft.fftfreq(signal_len, d=1 / fs)
    freqs_halfside = freqs[0:half_idx]
    power = 10 * np.log10(np.abs(f_x[0:half_idx]))
    max_power = power.max()
    max_idx = power.argmax()
    max_freq = freqs_halfside[max_idx]
    print(f"Dominant frequency: {max_freq}, with power: {max_power}")
    return max_freq


def _find_invalid_hs(hs_idx, to_idx):
    # Some validation.
    # HS has to be followed by TO. otherwise mark as invalid.
    invalid_hs_idx = []
    for i in range(0, len(hs_idx) - 1):
        hs = hs_idx[i]
        next_hs = hs_idx[i + 1]
        next_to = to_idx[to_idx > hs]
        if len(next_to) < 1:
            # No found TO. Mark HS as invalid.
            invalid_hs_idx = np.append(invalid_hs_idx, hs)
        else:
            # Found next TO. Make sure it's in the right order
            # compared to upcoming HS.
            next_to = next_to[0]
            if next_hs < next_to:
                invalid_hs_idx = np.append(invalid_hs_idx, hs)

    # If the very last HS is not followed by TO, mark as invalid.
    if hs_idx[-1] > to_idx[-1]:
        invalid_hs_idx = np.append(invalid_hs_idx, hs_idx[-1])

    # Convert to int array
    invalid_hs_idx = np.array(invalid_hs_idx)
    invalid_hs_idx = invalid_hs_idx.astype(int)
    return invalid_hs_idx


# Calculates HS and TO with CWT, based on Pham et al., 2017.
def add_hs_to_wavelet(data, ap_axis=0, wavelet_type="gaus1",
                      plot_detected_results=False):
    """
    Detects heel strikes and toe off events in supplied data,
    and adds events to supplied data object.
    Event detection is done via continous wavelet transform (CWT),
    based on Pham et al., 2017 (https://doi.org/10.3389/fneur.2017.00457)

    Parameters
    ----------
    data : IMUData
        Data containing IMU raw data
    ap_axis : int
        Which axis is the anterior-posterior axis (default 0)
    wavelet_type : string
        Which type of wavelet for 2nd order differentiation (default gaus1)
    plot_detected_results : bool
        Plot the signal with detected events
    """

    util.check_type(data, "data", imu_data.IMUData)

    # Get AP acc data
    acc_ap = data.acc_data[:, ap_axis]

    # Preprocess acceleration data (see paragraph "Extraction of HS and TO from
    # IMU" lines 1-4 in Pham et al., 2017)
    acc_detrend = scipy.signal.detrend(acc_ap)
    fn = data.fs / 2
    b, a = scipy.signal.butter(2, 10 / fn, 'low')
    acc_ap_pp = scipy.signal.filtfilt(b, a, acc_detrend, axis=0)

    # Integrate detrended and filtered acceleration data (see paragraph
    # "Extraction of HS and TO from IMU" lines 4-7 in Pham et al., 2017)
    signal_len = acc_ap_pp.shape[0]
    time_vector = np.arange(0, signal_len) / data.fs
    acc_int = scipy.integrate.cumulative_trapezoid(acc_ap_pp, x=time_vector,
                                                   axis=0, initial=0)

    # Find the dominant frequency of the acceleration (see paragraph
    # "Extraction of HS and TO from IMU" lines 8-15 in Pham et al., 2017)
    max_freq = _find_dominant_freq(acc_ap_pp, data.fs)

    # Select wavelet for smoothing (see paragraph
    # "Extraction of HS and TO from IMU" lines 4-7 in Pham et al., 2017)
    wavelet_dcwt1 = pywt.ContinuousWavelet('gaus1')
    scales = _get_scales(wavelet_dcwt1, max_freq, data.fs)

    # Differentiate integrated signal to smooth acceleration signal
    acc_wave, _ = pywt.cwt(acc_int, scales, wavelet_dcwt1)
    # Invert to match original signal, see Pham and McCamley
    acc_wave = -acc_wave[0, :]
    acc_wave = np.real(acc_wave)

    # Find heel strike events (see Fig. 2 in Pham et al., 2017)
    # Here, we want to find local minima. Apply findpeaks to negative signal.
    acc_wave_detrended = scipy.signal.detrend(acc_wave)
    hs_idx, _ = scipy.signal.find_peaks(-acc_wave_detrended)

    # Find toe off events (see paragraph "Adaptation of the cwt to the Steps
    # in the Home-Like Assessment" in Pham et al., 2017).
    # Paper recommends db2 for straight walking and gaus2 for turning,
    # although here gaus1 was found to work best.
    wavelet_dcwt2 = _get_wavelet(wavelet_type)
    scales = _get_scales(wavelet_dcwt2, max_freq, data.fs)
    acc_wave_2, _ = pywt.cwt(acc_wave, scales, wavelet_dcwt2)
    acc_wave_2 = -acc_wave_2[0, :]
    acc_wave_2 = np.real(acc_wave_2)

    # Find maxima of differentiated signal to get TO
    to_idx, _ = scipy.signal.find_peaks(acc_wave_2)

    # Peak selection by magnitude. From paper:
    # "HS/TO was as follows: magnitude >40% of the mean of
    # all peaks detected by the findpeaks function."
    # The limit has here been modified to 0.2 for HS
    # which was found to work better.
    pks_hs = acc_wave[hs_idx]
    pks_to = acc_wave_2[to_idx]
    hs_mean = np.mean(pks_hs)
    to_mean = np.mean(pks_to)
    hs_selected_idx = hs_idx[np.where(pks_hs < hs_mean * 0.2)]
    to_selected_idx = to_idx[np.where(pks_to > to_mean * 0.4)]

    # Find out-of-order hs
    invalid_hs_idx = _find_invalid_hs(hs_selected_idx, to_selected_idx)

    # Add to data structure
    for hs_idx in hs_selected_idx:
        hs_valid = event_data.GaitEventValidity.VALID
        if hs_idx in invalid_hs_idx:
            hs_valid = event_data.GaitEventValidity.OUT_OF_ORDER
        event = event_data.GaitEvent(event_data.GaitEventType.HEEL_STRIKE,
                                     event_data.GaitEventSide.NA, hs_idx,
                                     validity=hs_valid)
        data.add_event(event)
    for to_idx in to_selected_idx:
        event = event_data.GaitEvent(event_data.GaitEventType.TOE_OFF,
                                     event_data.GaitEventSide.NA, to_idx)
        data.add_event(event)

    if plot_detected_results:
        plot_signal.plot_wavelet_events(data, acc_ap_pp=acc_ap_pp,
                                        acc_wave_detrended=acc_wave_detrended,
                                        acc_wave_2=acc_wave_2)
