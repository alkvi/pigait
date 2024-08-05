"""
    Functions for calculating spatiotemporal gait
    parameters
"""

import numpy as np
from ..data import event_data
from ..data import imu_data


def get_step_time(sensor_set, max_lim=None):
    """
    Calculates step times from supplied sensor set.
    The sensor set must contain gait cycles.
    A step is heel strike on one foot, to heel strike
    on the opposite foot.

    Parameters
    ----------
    sensor_set : :py:class:`SensorSet`
        Data containing sensors with events
    max_lim : float
        Optional parameter for filtering values

    Returns
    ----------
    step_times_left : array of float
        Array of calculated step times in seconds
    step_times_right : array of float
        Array of calculated step times in seconds

    """

    fs = sensor_set.sensor_data[0].fs
    invalid_events = [event.sample_idx for event in sensor_set.events if
                      event.validity != event_data.GaitEventValidity.VALID]
    step_times_left = []
    step_times_right = []
    for cycle in sensor_set.gait_cycles:

        if (cycle.hs_start.sample_idx in invalid_events
            or cycle.hs_opposite.sample_idx in invalid_events
                or cycle.hs_end.sample_idx in invalid_events):
            print("Skipping invalid step cycle for step time calculation")
            continue

        hs_diff = cycle.hs_opposite.sample_idx - cycle.hs_start.sample_idx
        step_time_start = hs_diff * 1 / fs
        hs_diff = cycle.hs_end.sample_idx - cycle.hs_opposite.sample_idx
        step_time_opposite = hs_diff * 1 / fs

        if max_lim and (step_time_start > max_lim
                        or step_time_opposite > max_lim):
            print((
                f"Skipping gait cycle with "
                f"value {step_time_start}/{step_time_opposite}"
                f" - outside limit of {max_lim}"))
            continue

        if cycle.hs_start.side == event_data.GaitEventSide.RIGHT:
            step_times_right = np.append(step_times_right, step_time_start)
            step_times_left = np.append(step_times_left, step_time_opposite)
        else:
            step_times_left = np.append(step_times_left, step_time_start)
            step_times_right = np.append(step_times_right, step_time_opposite)

    return step_times_left, step_times_right


def get_cadence(step_times):
    """
    Calculates cadence from supplied step times.
    Cadence is amount of steps per minute.
    Step times should be in seconds.

    Parameters
    ----------
    step_times : array of float
        Step times to transform into cadence values

    Returns
    ----------
    cadence : array of float
        Array of calculated cadence values in steps/min

    """

    cadence = np.divide(np.ones(step_times.shape), step_times) * 60
    return cadence


def get_single_support(sensor_set):
    """
    Calculates single support times from supplied sensor set.
    Start with a TO. Get time until next HS.

    Parameters
    ----------
    sensor_set : :py:class:`SensorSet`
        Data containing sensors with events

    Returns
    ----------
    tss_times_left : array of float
        Array of calculated single support times in seconds
    tss_times_right : array of float
        Array of calculated single support times in seconds

    """

    fs = sensor_set.sensor_data[0].fs
    invalid_events = [event.sample_idx for event in sensor_set.events if
                      event.validity != event_data.GaitEventValidity.VALID]
    tss_times_left = []
    tss_times_right = []
    for cycle in sensor_set.gait_cycles:

        to_frame = cycle.to_opposite.sample_idx
        next_hs_frame = cycle.hs_opposite.sample_idx
        next_to_frame = cycle.to.sample_idx
        last_hs_frame = cycle.hs_end.sample_idx

        if next_hs_frame in invalid_events or last_hs_frame in invalid_events:
            print("Skipping invalid step cycle for single support calculation")
            continue

        tss_start = (next_hs_frame - to_frame) * (1 / fs)
        tss_opposite = (last_hs_frame - next_to_frame) * (1 / fs)

        if cycle.hs_start.side == event_data.GaitEventSide.RIGHT:
            tss_times_right = np.append(tss_times_right, tss_start)
            tss_times_left = np.append(tss_times_left, tss_opposite)
        else:
            tss_times_left = np.append(tss_times_left, tss_start)
            tss_times_right = np.append(tss_times_right, tss_opposite)

    return tss_times_left, tss_times_right


def get_double_support(sensor_set):
    """
    Calculates double support times from supplied sensor set.
    Double support consists of both inital and terminal double support.
    Initial is between RHS and LTO. Terminal is between LHS and RTO.
    Add these together to get double support.
    Start with assuming first event is right (would be the same if we
    started with left).

    Parameters
    ----------
    sensor_set : :py:class:`SensorSet`
        Data containing sensors with events

    Returns
    ----------
    tds_times : array of float
        Array of calculated double support times in seconds

    """

    fs = sensor_set.sensor_data[0].fs
    tds_times = []
    for cycle in sensor_set.gait_cycles:
        initial_double_support = (cycle.to_opposite.sample_idx
                                  - cycle.hs_start.sample_idx) * 1 / fs
        term_double_support = (cycle.to.sample_idx
                               - cycle.hs_opposite.sample_idx) * 1 / fs
        tds_times = np.append(tds_times,
                              initial_double_support + term_double_support)
    return tds_times


# The amplitude of changes in vertical position (h)
# was determined as the difference between highest and
# lowest position during a step cycle
# Assuming the lumbar sensor is placed around L5,
# use factor l = height x 0.53 (Del Din 2016)
def _get_step_length(z_interval, subject_height):
    if len(z_interval) < 1:
        print("No z position data between HS")
        return None
    delta_z = z_interval.max() - z_interval.min()
    delta_z = abs(delta_z)
    step_length = 2 * np.sqrt(2 * (subject_height / 100)
                                * 0.53 * delta_z - np.power(delta_z, 2))
    return step_length


def get_step_length_speed_lumbar(sensor_set, subject_height):
    """
    Calculates step lengths and walking speed from supplied sensor set.
    Calculates for left and right feet, based on positions of lumbar sensor.
    Method based on Ziljstra and Hof 2003
    (https://doi.org/10.1016/s0966-6362(02)00190-x)
    and Del Din et al., 2016 (https://doi.org/10.1109/jbhi.2015.2419317),
    using an inverted pendulum model or IPM

    Parameters
    ----------
    sensor_set : :py:class:`SensorSet`
        Data containing sensors with events
    subject_height : float
        Subject height in cm

    Returns
    ----------
    step_lengths_left : array of float
        Array of calculated step lengths in m
    step_lengths_right : array of float
        Array of calculated step lengths in m
    walking_speeds_left : array of float
        Array of calculated walking speeds in m/s
    walking_speeds_right : array of float
        Array of calculated walking speeds in m/s

    """

    lumbar_data = [sensor_data for sensor_data in sensor_set.sensor_data
                   if sensor_data.sensor_position == imu_data.SensorPosition.LUMBAR]
    if len(lumbar_data) != 1:
        print("Requires one lumbar data sensor")
    lumbar_data = lumbar_data[0]

    # Use Z positions and calculate parameters
    z_positions = lumbar_data.position[:, 2]
    fs = sensor_set.sensor_data[0].fs
    invalid_events = [event.sample_idx for event in sensor_set.events if
                      event.validity != event_data.GaitEventValidity.VALID]

    step_lengths_left = []
    step_lengths_right = []
    walking_speeds_left = []
    walking_speeds_right = []
    for cycle in sensor_set.gait_cycles:

        # Skip cycles involving invalid steps
        if (cycle.hs_opposite.sample_idx in invalid_events
                or cycle.hs_start.sample_idx in invalid_events):
            print("Skipping invalid step cycle for step length calculation")
            continue

        # Start side
        z_interval = z_positions[cycle.hs_start.sample_idx:
                                 cycle.hs_opposite.sample_idx]
        step_length_start = _get_step_length(z_interval, subject_height)
        if not step_length_start:
            continue
        hs_diff = cycle.hs_opposite.sample_idx - cycle.hs_start.sample_idx
        walking_speed_start = step_length_start / (hs_diff / fs)

        # Opposite side
        z_interval = z_positions[cycle.hs_opposite.sample_idx:
                                 cycle.hs_end.sample_idx]
        step_length_opposite = _get_step_length(z_interval, subject_height)
        if not step_length_opposite:
            continue
        hs_diff = cycle.hs_end.sample_idx - cycle.hs_opposite.sample_idx
        walking_speed_opposite = step_length_opposite / (hs_diff / fs)

        if cycle.hs_start.side == event_data.GaitEventSide.RIGHT:
            step_lengths_right = np.append(step_lengths_right,
                                           step_length_start)
            walking_speeds_right = np.append(walking_speeds_right,
                                             walking_speed_start)
            step_lengths_left = np.append(step_lengths_left,
                                          step_length_opposite)
            walking_speeds_left = np.append(walking_speeds_left,
                                            walking_speed_opposite)
        else:
            step_lengths_right = np.append(step_lengths_right,
                                           step_length_opposite)
            walking_speeds_right = np.append(walking_speeds_right,
                                             walking_speed_opposite)
            step_lengths_left = np.append(step_lengths_left, step_length_start)
            walking_speeds_left = np.append(walking_speeds_left,
                                            walking_speed_start)

    return (step_lengths_left, step_lengths_right,
            walking_speeds_left, walking_speeds_right)


# Stride lengths and walking speeds from 3D positions of foot sensor.
# Stride: how much one foot travels from FF to FF.
def get_stride_length_walking_speed_foot(data, heading_steps=2,
                                         min_lim=0,
                                         max_lim=2):
    """
    Calculates stride lengths and walking speed from supplied data with events.
    Calculates for a single foot, based on the 3D positions of the sensor.
    TODO: document assumptions, fix trajectory plot

    Parameters
    ----------
    data : IMUData
        Data containing IMU raw data

    Returns
    ----------
    step_lengths : array of float
        Array of calculated step lengths in m
    walking_speeds : array of float
        Array of calculated walking speeds in m/s

    """

    # Get foot-flat events
    ff_events = data.get_events(event_data.GaitEventType.FOOT_FLAT)
    ff_indices = [event.sample_idx for event in ff_events]

    # Get start and end positions
    end_ff = ff_indices[-1]
    # start_pos = data.position[0, :]
    if end_ff >= data.position.shape[0]:
        end_ff = data.position.shape[0]-1
        ff_indices[-1] = end_ff
    end_pos = data.position[end_ff, :]

    # Calculate strides between foot flats
    stride_lengths = []
    walking_speeds = []
    for ff_idx in range(0, len(ff_indices) - 1):
        ff = ff_indices[ff_idx]
        next_ff = ff_indices[ff_idx + 1]
        pos_first = data.position[ff, :]
        pos_second = data.position[next_ff, :]
        pos_diff = pos_second - pos_first

        # Project stride vector into a local heading direction.
        # If heading_steps additional FFs exist,
        # use the position of the last FF.
        # Otherwise, use the final position.
        if ff_idx+heading_steps < len(ff_indices):
            heading_ff = ff_indices[ff_idx + heading_steps]
            heading_vector = (data.position[heading_ff, :] -
                              data.position[ff, :])
        else:
            heading_vector = end_pos - data.position[ff, :]

        # Perform scalar projection
        stride_length = (np.dot(pos_diff, heading_vector) /
                         np.linalg.norm(heading_vector))

        # Make sure stride is within certain limits
        if stride_length < min_lim or stride_length > max_lim:
            print((
                f"Skipping stride length with "
                f"value {stride_length} (outside limit)"))
        else:
            stride_lengths = np.append(stride_lengths, stride_length)
            walking_speed = stride_length / ((next_ff-ff) / data.fs)
            walking_speeds = np.append(walking_speeds, walking_speed)

    return stride_lengths, walking_speeds
