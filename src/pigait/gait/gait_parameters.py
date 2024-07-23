"""
    Functions for calculating spatiotemporal gait
    parameters
"""

import numpy as np
from ..data import event_data


def get_step_time(data, start_side):
    """
    Calculates step times from supplied data with events.
    A step is heel strike on one foot, to heel strike
    on the opposite foot.

    Parameters
    ----------
    data : IMUData
        Data containing IMU raw data
    start_side : GaitEventSide
        Which side to start calculating steps from

    Returns
    ----------
    step_times : array of float
        Array of calculated step times in seconds

    """

    hs_lf = data.get_events(event_data.GaitEventType.HEEL_STRIKE,
                            event_data.GaitEventSide.LEFT)
    hs_lf = np.array([event.sample_idx for event in hs_lf])
    hs_rf = data.get_events(event_data.GaitEventType.HEEL_STRIKE,
                            event_data.GaitEventSide.RIGHT)
    hs_rf = np.array([event.sample_idx for event in hs_rf])
    invalid_hs = [event.sample_idx for event in data.events if
                  (event.event_type == event_data.GaitEventType.HEEL_STRIKE
                   and event.validity != event_data.GaitEventValidity.VALID)]

    if start_side == event_data.GaitEventSide.RIGHT:
        hs_start_side = hs_rf
        hs_other_side = hs_lf
    elif start_side == event_data.GaitEventSide.LEFT:
        hs_start_side = hs_lf
        hs_other_side = hs_rf
    else:
        raise TypeError("Unknown starting side")

    step_times = []
    for hs_idx in range(0, len(hs_start_side)):
        hs = hs_start_side[hs_idx]
        other_side_hs = hs_other_side[hs_other_side > hs]
        if len(other_side_hs) < 1:
            break
        other_side_hs = other_side_hs[0]
        if hs in invalid_hs or other_side_hs in invalid_hs:
            print("Skipping invalid step cycle for step time calculation")
            continue
        hs_diff = other_side_hs - hs
        step_time = hs_diff * 1 / data.fs
        step_times = np.append(step_times, step_time)

    return step_times


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


def get_single_support(data):
    """
    Calculates single support times from supplied data with events.
    Single support L and R are calculated identically.
    Start with a TO. Get time until next HS.

    Parameters
    ----------
    data : IMUData
        Data containing IMU raw data

    Returns
    ----------
    tss_times : array of float
        Array of calculated single support times in seconds

    """

    hs = data.get_events(event_data.GaitEventType.HEEL_STRIKE)
    hs = np.array([event.sample_idx for event in hs])
    to = data.get_events(event_data.GaitEventType.TOE_OFF)
    to = np.array([event.sample_idx for event in to])
    invalid_hs = [event.sample_idx for event in data.events if
                  (event.event_type == event_data.GaitEventType.HEEL_STRIKE
                   and event.validity != event_data.GaitEventValidity.VALID)]
    tss_times = []
    for to_idx in range(0, len(to)):
        to_frame = to[to_idx]
        next_hs_frame = hs[hs > to_frame]
        if len(next_hs_frame) == 0:
            break
        next_hs_frame = next_hs_frame[0]
        if next_hs_frame in invalid_hs:
            print("Skipping invalid step cycle for single support calculation")
            continue
        tss = (next_hs_frame - to_frame) * (1 / data.fs)
        tss_times = np.append(tss_times, tss)
    return tss_times


def get_double_support(data):
    """
    Calculates double support times from supplied data with events.
    Double support consists of both inital and terminal double support.
    Initial is between RHS and LTO. Terminal is between LHS and RTO.
    Add these together to get double support.
    Start with assuming first event is right (would be the same if we
    started with left).

    Parameters
    ----------
    data : IMUData
        Data containing IMU raw data

    Returns
    ----------
    tds_times : array of float
        Array of calculated double support times in seconds

    """

    hs = data.get_events(event_data.GaitEventType.HEEL_STRIKE)
    hs = np.array([event.sample_idx for event in hs])
    to = data.get_events(event_data.GaitEventType.TOE_OFF)
    to = np.array([event.sample_idx for event in to])
    tds_times = []
    for hs_idx in range(0, len(hs)):
        right_hs_frame = hs[hs_idx]
        left_to_frame = to[to > right_hs_frame]
        if len(left_to_frame) == 0:
            break
        left_to_frame = left_to_frame[0]
        initial_double_support = (left_to_frame - right_hs_frame) * 1 / data.fs
        left_hs_frame = hs[hs > right_hs_frame]
        if len(left_hs_frame) == 0:
            break
        left_hs_frame = left_hs_frame[0]
        right_to_frame = to[to > left_hs_frame]
        if len(right_to_frame) == 0:
            break
        right_to_frame = right_to_frame[0]
        term_double_support = (right_to_frame - left_hs_frame) * 1 / data.fs
        tds_times = np.append(tds_times,
                              initial_double_support + term_double_support)
    return tds_times


def get_step_length_speed_lumbar(data, subject_height):
    """
    Calculates step lengths and walking speed from supplied data with events.
    Calculates for left and right feet, based on positions of lumbar sensor.
    Method based on Ziljstra and Hof 2003
    (https://doi.org/10.1016/s0966-6362(02)00190-x)
    and Del Din et al., 2016 (https://doi.org/10.1109/jbhi.2015.2419317),
    using an inverted pendulum model or IPM

    Parameters
    ----------
    data : IMUData
        Data containing IMU raw data
    subject_height : float
        Subject height in cm

    Returns
    ----------
    step_lengths : array of float
        Array of calculated step lengths in m
    walking_speeds : array of float
        Array of calculated walking speeds in m/s

    """

    hs_lf = data.get_events(event_data.GaitEventType.HEEL_STRIKE,
                            event_data.GaitEventSide.LEFT)
    hs_lf = [event.sample_idx for event in hs_lf]
    hs_rf = data.get_events(event_data.GaitEventType.HEEL_STRIKE,
                            event_data.GaitEventSide.RIGHT)
    hs_rf = [event.sample_idx for event in hs_rf]
    invalid_hs = [event.sample_idx for event in data.events if
                  (event.event_type == event_data.GaitEventType.HEEL_STRIKE
                   and event.validity != event_data.GaitEventValidity.VALID)]

    hs_start_side = hs_rf
    hs_other_side = hs_lf
    if hs_lf[0] < hs_rf[0]:
        hs_start_side = hs_lf
        hs_other_side = hs_rf

    # Use Z positions and calculate parameters
    z_positions = data.position[:, 2]
    step_lengths = []
    walking_speeds = []
    for hs_idx in range(0, len(hs_start_side) - 1):

        # Make sure we have enough HS to calculate this cycle
        if hs_idx >= len(hs_other_side):
            continue

        # Make sure events are in the correct order
        if hs_other_side[hs_idx] < hs_start_side[hs_idx]:
            print(("Skipping wrong order step cycle for"
                   " step length calculation"))
            continue

        # Skip cycles involving invalid steps
        if (hs_other_side[hs_idx] in invalid_hs
                or hs_start_side[hs_idx] in invalid_hs):
            print("Skipping invalid step cycle for step length calculation")
            continue

        # The amplitude of changes in vertical position (h)
        # was determined as the difference between highest and
        # lowest position during a step cycle
        # Assuming the lumbar sensor is placed around L5,
        # use factor l = height x 0.53 (Del Din 2016)
        z_interval = z_positions[hs_start_side[hs_idx]:hs_other_side[hs_idx]]
        if len(z_interval) < 1:
            print("No z position data between HS")
            continue
        delta_z = z_interval.max() - z_interval.min()
        delta_z = abs(delta_z)
        step_length = 2 * np.sqrt(2 * (subject_height / 100)
                                  * 0.53 * delta_z - np.power(delta_z, 2))
        step_lengths = np.append(step_lengths, step_length)
        hs_diff = hs_other_side[hs_idx] - hs_start_side[hs_idx]
        walking_speed = step_length / (hs_diff / data.fs)
        walking_speeds = np.append(walking_speeds, walking_speed)

    return step_lengths, walking_speeds


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
