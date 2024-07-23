"""
    Inertial navigation system (INS) methods
"""

import scipy
import numpy as np
import ahrs

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


# Calculate velocity between events
def _integrate_acceleration(event_indices, fs, acc_xyz_global,
                            dedrift_linear=False):

    vel = np.zeros(acc_xyz_global.shape)
    for event_idx in range(0, len(event_indices) - 1):

        event = event_indices[event_idx]
        next_event = event_indices[event_idx + 1]
        if event == next_event:
            continue
        time_vector = np.arange(event, next_event) / fs

        vel[event:next_event, 0] = scipy.integrate.cumulative_trapezoid(
            acc_xyz_global[event:next_event, 0],
            x=time_vector, axis=0, initial=0)
        vel[event:next_event, 1] = scipy.integrate.cumulative_trapezoid(
            acc_xyz_global[event:next_event, 1],
            x=time_vector, axis=0, initial=0)
        vel[event:next_event, 2] = scipy.integrate.cumulative_trapezoid(
            acc_xyz_global[event:next_event, 2],
            x=time_vector, axis=0, initial=0)

        if dedrift_linear:
            # Get drift between events
            vel_interval = vel[event: next_event, :]
            vel_interval_t = np.arange(vel_interval.shape[0])
            vel_linear_drift_x = scipy.interpolate.interp1d(
                [vel_interval_t[0], vel_interval_t[-1]],
                [vel_interval[0, 0], vel_interval[-1, 0]])
            vel_linear_drift_y = scipy.interpolate.interp1d(
                [vel_interval_t[0], vel_interval_t[-1]],
                [vel_interval[0, 1], vel_interval[-1, 1]])
            vel_linear_drift_z = scipy.interpolate.interp1d(
                [vel_interval_t[0], vel_interval_t[-1]],
                [vel_interval[0, 2], vel_interval[-1, 2]])
            drift_x = vel_linear_drift_x(vel_interval_t)
            drift_y = vel_linear_drift_y(vel_interval_t)
            drift_z = vel_linear_drift_z(vel_interval_t)

            # De-drift
            vel[event:next_event, 0] = vel[event:next_event, 0] - drift_x
            vel[event:next_event, 1] = vel[event:next_event, 1] - drift_y
            vel[event:next_event, 2] = vel[event:next_event, 2] - drift_z

    return vel


# Integrate velocity between events to get positions
def _integrate_velocity(event_idx_sorted, fs, vel):

    pos = np.zeros(vel.shape)
    for event_idx in range(0, len(event_idx_sorted) - 1):

        event = event_idx_sorted[event_idx]
        next_event = event_idx_sorted[event_idx + 1]
        if event == next_event:
            continue

        initial_x = pos[event - 1, 0]
        initial_y = pos[event - 1, 1]
        initial_z = pos[event - 1, 2]
        time_vector = np.arange(event, next_event) / fs
        pos[event:next_event, 0] = scipy.integrate.cumulative_trapezoid(
            vel[event:next_event, 0],
            x=time_vector, axis=0, initial=0) + initial_x
        pos[event:next_event, 1] = scipy.integrate.cumulative_trapezoid(
            vel[event:next_event, 1],
            x=time_vector, axis=0, initial=0) + initial_y
        pos[event:next_event, 2] = scipy.integrate.cumulative_trapezoid(
            vel[event:next_event, 2],
            x=time_vector, axis=0, initial=0) + initial_z

    # Add last index
    pos[event_idx_sorted[-1] - 1:, 0] = pos[event_idx_sorted[-1] - 1, 0]
    pos[event_idx_sorted[-1] - 1:, 1] = pos[event_idx_sorted[-1] - 1, 1]
    pos[event_idx_sorted[-1] - 1:, 2] = pos[event_idx_sorted[-1] - 1, 2]

    return pos


def calculate_positions_lumbar(data, stationary_samples=500,
                               plot_result=False, plot_debug=False):
    """
    Calculates position from supplied lumbar data
    with heel strike events.
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
    all_hs_sorted = np.sort(np.concatenate((hs_lf, hs_rf)))
    vel = _integrate_acceleration(all_hs_sorted, data_filt.fs, acc_xyz_global)

    # Add velocity to original data
    data.velocity = vel

    # Get positions between heel strikes
    pos = _integrate_velocity(all_hs_sorted, data_filt.fs, vel)

    # Zijlstra and Hof 2003: To detrend positions, high-pass with 0.1 Hz.
    fn = data_filt.fs / 2
    b, a = scipy.signal.butter(4, 0.1 / fn, 'high')
    pos_filt = scipy.signal.filtfilt(b, a, pos, axis=0)

    # Add positions to original data
    data.position = pos_filt

    # Debug plotting
    if plot_debug:
        plot_signal.plot_velocity(data)
        plot_signal.plot_axes_with_hs(data, pos, pos_filt, 'Pos')

    # Plot result
    hs_all = data_filt.get_events(event_data.GaitEventType.HEEL_STRIKE)
    if plot_result or plot_debug:
        plot_signal.plot_position(data, events=hs_all)


def calculate_positions_zupt(data, start_orientation=None,
                             force_zero_acc=False,
                             plot_result=False, plot_debug=False):
    """
    Calculates position from supplied foot sensor data with
    foot-flat events, using zero-velocity updates (ZUPT) and
    a Madgwick orientation filter to determine sensor rotation.
    The zero-velocity points are assumed to be the foot-flat
    time points.
    Adds velocity and positions to the data structure.

    Parameters
    ----------
    data : IMUData
        Data containing IMU raw data
    start_orientation : array of float (default None)
        Initial orientation of the Madgwick
        orientation filter (q0)
    force_zero_acc : bool (default False)
        If set to True, forces zero acceleration
        on foot-flat time points with a linear function
    plot_result : bool (default False)
        Whether to plot resulting positions
    plot_debug : bool (default False)
        Plots calculated velocity and positions along the way

    """

    # Low-pass signals for smoothing, 4 Hz, 4th order butterworth
    data_filt = filtering.filter_signal_butter(data, 4, 4,
                                               plot_filter=plot_debug)
    acc_data = data_filt.acc_data
    gyro_data = data_filt.gyro_data
    mag_data = data_filt.mag_data

    # Get foot flat events
    ff = data_filt.get_events(event_data.GaitEventType.FOOT_FLAT)
    ff = [event.sample_idx for event in ff]

    # Calculate orientation quaternions in a global frame (ENU)
    # via Madgwick's orientation filter
    # Gain as recommended for MARG systems from Madgwick et al 2011
    madgwick_gain = 0.041
    madgwick = ahrs.filters.Madgwick(gyr=gyro_data, acc=acc_data,
                                     mag=mag_data, frequency=data.fs,
                                     Dt=1 / data.fs, gain=madgwick_gain,
                                     q0=start_orientation)

    # Scipy library expects quaternion in (x,y,z,w)
    orientation_quaternion = madgwick.Q
    quat_s = np.copy(orientation_quaternion)
    quat_s[:, 0] = orientation_quaternion[:, 1]
    quat_s[:, 1] = orientation_quaternion[:, 2]
    quat_s[:, 2] = orientation_quaternion[:, 3]
    quat_s[:, 3] = orientation_quaternion[:, 0]
    quat_r = scipy.spatial.transform.Rotation.from_quat(quat_s)

    # Rotate body accelerations to Earth frame (ENU)
    acc_xyz_global = quat_r.inv().apply(acc_data)

    # Remove gravity from measurements (in earth frame)
    gravity_vector = np.transpose(
        np.array([np.zeros(acc_data.shape[0]),
                  np.zeros(acc_data.shape[0]),
                  np.ones(acc_data.shape[0]) * 9.81]))
    acc_xyz_global = acc_xyz_global - gravity_vector

    # If set, get an additional adjustment by forcing
    # zero acceleration on FF with function
    # f(x(t)) = t/T*x(t), T = t(ff)
    if force_zero_acc:
        acc_xyz_global_corr = np.copy(acc_xyz_global)
        for ff_idx in range(0, len(ff) - 1):
            tff = ff[ff_idx]
            next_tff = ff[ff_idx + 1]
            for t in range(tff, next_tff):
                acc_xyz_global_corr[t, 0] = (acc_xyz_global[t, 0] *
                                             ((next_tff - t) / (next_tff-tff)))
                acc_xyz_global_corr[t, 1] = (acc_xyz_global[t, 1] *
                                             ((next_tff - t) / (next_tff-tff)))
                acc_xyz_global_corr[t, 2] = (acc_xyz_global[t, 2] *
                                             ((next_tff - t) / (next_tff-tff)))
        # And replace global Z acc with corrected version
        acc_xyz_global[:, 2] = acc_xyz_global_corr[:, 2]

    # Calculate linearly de-drifted velocity between foot flats
    vel = _integrate_acceleration(ff, data_filt.fs, acc_xyz_global,
                                  dedrift_linear=True)

    # Add velocity to original data
    data.velocity = vel

    # Integrate velocity to yield position
    pos = _integrate_velocity(ff, data_filt.fs, vel)

    # Add positions to original data
    data.position = pos

    # Debug plotting
    if plot_debug:
        plot_signal.plot_axes_with_hs(data, data.acc_data,
                                      acc_xyz_global, 'Acc')
        plot_signal.plot_quaternions_euler(madgwick.Q, ff)
        plot_signal.plot_velocity(data)
        plot_signal.plot_profile(ff, vel, "velocity")
        plot_signal.plot_axes_with_hs(data, pos, pos, 'Pos')
        plot_signal.plot_profile(ff, pos, "position")

    # Plot result
    ff_events = data_filt.get_events(event_data.GaitEventType.FOOT_FLAT)
    if plot_result or plot_debug:
        plot_signal.plot_position(data, events=ff_events)
