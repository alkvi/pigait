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


def add_hs_to_wavelet(data, ap_axis=0, wavelet_type="gaus1",
                      plot_detected_results=False):
    """
    Detects heel strikes and toe off events in supplied data,
    and adds events to supplied data object.
    Event detection is done via continous wavelet transform (CWT),
    based on Pham et al., 2017 (https://doi.org/10.3389/fneur.2017.00457)

    Parameters
    ----------
    data : :py:class:`pigait.data.imu_data.IMUData`
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


# Calculates HS, TO and FF with gyroscope data, based on Salarian et al., 2004.
# TODO: make parameters configurable
def add_hs_to_gyro(data, event_side=None, plot_detected_results=False):
    """
    Detects heel strikes, toe off, mid-swing and foot-flat events
    in supplied data, and adds events to supplied data object.
    Event detection is done via peak detection on gyroscope data,
    specifically the y axis representing rotation around the mediolateral
    axis (pitch),
    based on Salarian et al., 2004 (https://doi.org/10.1109/tbme.2004.827933)

    Parameters
    ----------
    data : :py:class:`pigait.data.imu_data.IMUData`
        Data containing IMU raw data
    event_side : :py:class:`pigait.data.event_data.GaitEventSide`
        Which side detected events should be assigned
    plot_detected_results : bool
        Plot the signal with detected events
    """

    # we called on this 2 times, one for each foot
    # hs_lf, to_lf, ff_lf, stance_lf = -gyro_data_lf_cut[:,1]
    # hs_rf, to_rf, ff_rf, stance_rf = -gyro_data_rf_cut[:,1]
    # just do both here? no, we take one sensor...

    # We'll use the gyroscope data in the -y direction,
    # assumed to be rotation around the mediolateral axis,
    # i.e. pitch
    gyro_data = -data.gyro_data[:, 1]

    # Normalize data
    gyro_data = scipy.stats.zscore(gyro_data)

    # Prepare a filter
    # Order 48 FIR, low-pass, cutoff 30 Hz
    fn = data.fs / 2
    b = scipy.signal.firwin(48, 30 / fn, pass_zero="lowpass")

    # First identify maxima of signal to identifty mid-swing
    # Salarian et al 2004: Those peaks that were larger than
    # 50 deg/s were candidates
    # If multiple adjacent peaks within a maximum distance of
    # 500 ms were detected,
    # the peak with the highest amplitude was selected and the
    # others were discarded
    min_midswing_height = 1  # normalized signal, not an absolute value
    min_peak_distance_time = 0.5
    min_midswing_distance_samples = int(data.fs * min_peak_distance_time)
    ms_idx, _ = scipy.signal.find_peaks(gyro_data, height=min_midswing_height,
                                        distance=min_midswing_distance_samples)

    # Salarian et al 2004: local minimum peaks of shank signal
    # inside interval -1.5s +1.5s were searched.
    # The nearest local minimum after MS was selected as IC.
    search_interval = 1.5
    search_interval_samples = int(data.fs * search_interval)

    # Scan around each mid-swing
    all_hs = np.array([])
    all_to = np.array([])
    for i in range(0, len(ms_idx)):
        t_ms_idx = ms_idx[i]
        start_idx = int(t_ms_idx - search_interval_samples)
        end_idx = int(t_ms_idx + search_interval_samples)
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(gyro_data)-1:
            end_idx = len(gyro_data)-1

        signal_interval = gyro_data[start_idx:end_idx]

        # Salarian et al 2004: to smooth the signal and
        # to get rid of spurious peaks, the signal was filtered
        # using a low-pass FIR filter with cutoff frequency
        # of 30 Hz and pass-band attenuation of less than 0.5 dB.
        signal_interval = scipy.signal.filtfilt(b, [1.0], signal_interval)

        # The nearest local minimum after the t_ms
        # was selected as IC (i.e. HS).
        min_peak_height = 0.1
        if search_interval_samples > t_ms_idx:
            signal_interval_hs = signal_interval[t_ms_idx:-1]
        else:
            signal_interval_hs = signal_interval[
                search_interval_samples:-1]

        # In case we have a very tiny segment, use argmax for peak.
        # Otherwise find_peaks.
        if (len(signal_interval_hs)) < 1:
            # Interval might be at the end of signal, leave empty
            hs_idx = []
        elif (len(signal_interval_hs)) < 3:
            hs_idx = [np.argmax(signal_interval_hs)]
        else:
            hs_idx, _ = scipy.signal.find_peaks(-signal_interval_hs,
                                                height=min_peak_height)

        # Add HS if found
        if len(hs_idx) > 0:
            hs_idx = hs_idx[0]
            all_hs = np.append(all_hs, t_ms_idx+hs_idx)

        # Salarian et al 2004: the minimum prior to t_ms
        # with amplitude less than -20 deg/s was
        # selected as the terminal contact (i.e. TO).
        if search_interval_samples < t_ms_idx:
            signal_interval_to = signal_interval[0: search_interval_samples]
        else:
            signal_interval_to = signal_interval[0: t_ms_idx]

        # Find TO and add if found
        min_peak_height = 1
        min_to_distance = 0.15
        min_to_distance_samples = int(data.fs * min_to_distance)
        to_idx, _ = scipy.signal.find_peaks(-signal_interval_to,
                                            height=min_peak_height,
                                            distance=min_to_distance_samples)
        if len(to_idx) >= 1:
            to_idx = to_idx[-1]
            all_to = np.append(all_to, start_idx+to_idx)

    # Make sure we have integers
    all_hs = all_hs.astype(int)
    all_to = all_to.astype(int)
    all_ms = ms_idx.astype(int)

    # Also return foot flat and stance time points.
    # Stance is time between HS and TO on same leg.
    # Identify foot flat times as when angular velocity absolute value
    # is below a certain threshold, during each stance phase.
    all_tff = np.array([])
    for i in range(0, len(all_hs)):

        hs = all_hs[i]
        next_to = all_to[all_to > hs]
        if (len(next_to) < 1):
            continue
        next_to = next_to[0]

        gyro_interval_times = np.array(range(hs, next_to))
        gyro_interval = gyro_data[hs:next_to]
        gyro_interval_flat_time = gyro_interval_times[
            np.where(abs(gyro_interval) < 0.2)[0]]

        # Only take instants
        if len(gyro_interval_flat_time) < 1:
            continue
        median_idx = int(np.floor(len(gyro_interval_flat_time)/2))
        ff_median = gyro_interval_flat_time[median_idx]
        all_tff = np.append(all_tff, ff_median)
    all_tff = all_tff.astype(int)

    # Add all events to specified side, if any
    side = event_data.GaitEventSide.NA
    if event_side:
        side = event_side

    # For now, consider all events valid
    validity = event_data.GaitEventValidity.VALID

    # Add events to data
    for hs_idx in all_hs:
        event = event_data.GaitEvent(event_data.GaitEventType.HEEL_STRIKE,
                                     side, hs_idx,
                                     validity=validity)
        data.add_event(event)

    for to_idx in all_to:
        event = event_data.GaitEvent(event_data.GaitEventType.TOE_OFF,
                                     side, to_idx,
                                     validity=validity)
        data.add_event(event)

    for ms_idx in all_ms:
        event = event_data.GaitEvent(event_data.GaitEventType.MID_SWING,
                                     side, ms_idx,
                                     validity=validity)
        data.add_event(event)

    for tff_idx in all_tff:
        event = event_data.GaitEvent(event_data.GaitEventType.FOOT_FLAT,
                                     side, tff_idx,
                                     validity=validity)
        data.add_event(event)

    if plot_detected_results:
        plot_signal.plot_gyro_detection(data, side, gyro_data)
