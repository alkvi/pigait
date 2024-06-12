"""
    Inertial navigation system (INS) methods
"""

import scipy
import numpy as np

from ..plotting import plot_signal
from ..filtering import filtering
from ..data import event_data


def _remove_gravity_pitch(data, stationary_samples=500, plot_result=False):

    # Rotate all acc vectors around Y axis (pitch)
    # until mean Z acc is the absolute largest
    acc_z_mean = np.mean(data.acc_data[0: stationary_samples, 2])
    largest_z = acc_z_mean
    pitch_deg = 0
    pitch_for_largest_z = 0
    for pitch_deg in np.arange(-180, 180, 0.5):
        global_rot = scipy.spatial.transform.Rotation.from_rotvec(
            pitch_deg * np.array([0, 1, 0]), degrees=True)
        acc_xyz_rot = global_rot.inv().apply(data.acc_data)
        acc_z_mean = np.mean(acc_xyz_rot[0: stationary_samples, 2])

        if acc_z_mean > largest_z:
            largest_z = acc_z_mean
            pitch_for_largest_z = pitch_deg

    print("Largest Z: {largest_z}, pitch for largest Z: {pitch_for_largest_z}")

    # Rotate body accelerations around Y so largest Z coincides with gravity
    global_rot = scipy.spatial.transform.Rotation.from_rotvec(
        pitch_for_largest_z * np.array([0, 1, 0]), degrees=True)
    acc_xyz_global = global_rot.inv().apply(data.acc_data)

    # Remove gravity from measurements (in rotated frame)
    grav_offset = largest_z
    gravity_vector = np.transpose(np.array([np.zeros(acc_xyz_global.shape[0]),
                                            np.zeros(acc_xyz_global.shape[0]),
                                            np.ones(acc_xyz_global.shape[0])
                                            * grav_offset]))
    acc_xyz_global = acc_xyz_global - gravity_vector

    if plot_result:
        plot_signal.plot_axes_with_hs(data, data.acc_data,
                                      acc_xyz_global, 'Acc')

    return acc_xyz_global


# Calculate velocity between heel strikes
def _calculate_velocity(hs_lf, hs_rf, fs, acc_xyz_global):
    all_hs = np.sort(np.concatenate((hs_lf, hs_rf)))
    vel = np.zeros(acc_xyz_global.shape)
    for hs_idx in range(0, len(all_hs) - 1):

        hs = all_hs[hs_idx]
        next_hs = all_hs[hs_idx + 1]
        if hs == next_hs:
            continue
        time_vector = np.arange(hs, next_hs) / fs

        vel[hs:next_hs, 0] = scipy.integrate.cumulative_trapezoid(
            acc_xyz_global[hs:next_hs, 0], x=time_vector, axis=0, initial=0)
        vel[hs:next_hs, 1] = scipy.integrate.cumulative_trapezoid(
            acc_xyz_global[hs:next_hs, 1], x=time_vector, axis=0, initial=0)
        vel[hs:next_hs, 2] = scipy.integrate.cumulative_trapezoid(
            acc_xyz_global[hs:next_hs, 2], x=time_vector, axis=0, initial=0)

    return vel


# Integrate velocity between heel strikes to get positions
def _calculate_position(hs_lf, hs_rf, vel, fs):
    all_hs = np.sort(np.concatenate((hs_lf, hs_rf)))
    pos = np.zeros(vel.shape)
    for hs_idx in range(0, len(all_hs) - 1):

        hs = all_hs[hs_idx]
        next_hs = all_hs[hs_idx + 1]
        if hs == next_hs:
            continue

        initial_x = pos[hs - 1, 0]
        initial_y = pos[hs - 1, 1]
        initial_z = pos[hs - 1, 2]
        time_vector = np.arange(hs, next_hs) / fs
        pos[hs:next_hs, 0] = scipy.integrate.cumulative_trapezoid(
            vel[hs:next_hs, 0], x=time_vector, axis=0, initial=0) + initial_x
        pos[hs:next_hs, 1] = scipy.integrate.cumulative_trapezoid(
            vel[hs:next_hs, 1], x=time_vector, axis=0, initial=0) + initial_y
        pos[hs:next_hs, 2] = scipy.integrate.cumulative_trapezoid(
            vel[hs:next_hs, 2], x=time_vector, axis=0, initial=0) + initial_z

    # Add last index
    pos[all_hs[-1] - 1:, 0] = pos[all_hs[-1] - 1, 0]
    pos[all_hs[-1] - 1:, 1] = pos[all_hs[-1] - 1, 1]
    pos[all_hs[-1] - 1:, 2] = pos[all_hs[-1] - 1, 2]

    return pos


def calculate_positions_lumbar(data, stationary_samples=500,
                               plot_result=False, plot_debug=False):
    """
    Calculates position from supplied data with events.
    Adds velocity and positions to the data structure.
    TODO: document assumptions of stationary, axis

    Parameters
    ----------
    data : IMUData
        Data containing IMU raw data
    stationary_samples : int (default 500)
        Samples where the sensor stays mostly still
    plot_result : bool (default False)
        Whether to plot resulting positions
    plot_debug : bool (default False)
        Plots calculated velocity and positions along the way

    """

    # Low-pass signals for smoothing, 4 Hz, 4th order butterworth
    data_filt = filtering.filter_signal_butter(data, 4, 4,
                                               plot_filter=plot_debug)

    # Get gravity-compensated acceleration
    acc_xyz_global = _remove_gravity_pitch(
        data_filt,
        stationary_samples=stationary_samples,
        plot_result=plot_debug)

    # Get events to calculate positions between
    hs_lf = data_filt.get_events(event_data.GaitEventType.HEEL_STRIKE,
                                 event_data.GaitEventSide.LEFT)
    hs_lf = [event.sample_idx for event in hs_lf]
    hs_rf = data_filt.get_events(event_data.GaitEventType.HEEL_STRIKE,
                                 event_data.GaitEventSide.RIGHT)
    hs_rf = [event.sample_idx for event in hs_rf]

    # Get velocity between heel strikes
    vel = _calculate_velocity(hs_lf, hs_rf, data_filt.fs, acc_xyz_global)

    # Add velocity to original data
    data.velocity = vel
    if plot_debug:
        plot_signal.plot_velocity(data)

    pos = _calculate_position(hs_lf, hs_rf, vel, data_filt.fs)

    # Zijlstra and Hof 2003: To detrend positions, high-pass with 0.1 Hz.
    fn = data_filt.fs / 2
    b, a = scipy.signal.butter(4, 0.1 / fn, 'high')
    pos_filt = scipy.signal.filtfilt(b, a, pos, axis=0)

    # Add positions to original data
    data.position = pos_filt
    if plot_debug:
        plot_signal.plot_axes_with_hs(data, pos, pos_filt, 'Pos')

    # Plot result?
    if plot_result or plot_debug:
        plot_signal.plot_position(data)
