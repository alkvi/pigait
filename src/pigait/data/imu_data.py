"""
    Data classes for holding IMU data
"""

from enum import Enum
import numpy as np
from ..data import event_data
from ..plotting import plot_signal


def _check_numpy_array(item, item_name="data"):
    if not isinstance(item, (np.ndarray)):
        raise TypeError((f"{item_name} must be"
                        "given as ndarray, got {type(item)}"))


def _check_numpy_equal_shape(array_1, array_2):
    ax_1_eq = array_1.shape[0] == array_2.shape[0]
    ax_2_eq = array_1.shape[1] == array_2.shape[1]
    if not ax_1_eq or not ax_2_eq:
        raise TypeError("Given arrays are not equal shape")


class SensorPosition(Enum):
    """
    Enum for sensor position on body.
    """

    LUMBAR = 1
    """
    For sensors positioned around lumbar or back
    """
    LEFT_FOOT = 2
    """
    For sensors positioned on the left foot
    """
    RIGHT_FOOT = 3
    """
    For sensors positioned on the right foot
    """
    LEFT_ARM = 4
    """
    For sensors positioned on the left arm
    """
    RIGHT_ARM = 5
    """
    For sensors positioned on the right arm
    """
    CHEST = 6
    """
    For sensors positioned on the chest
    """
    NA = 7
    """
    Sensor position not applicable
    """

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        return NotImplemented


class IMUData:
    """
    Raw data

    Attributes
    ----------
    acc_data : ndarray
        Accelerometer data as TxN np array, T=samples, N=axes
    gyro_data : ndarray
        Gyroscope data as TxN np array, T=samples, N=axes
    mag_data : ndarray | None
        Magnetometer data as TxN np array, T=samples, N=axes
    fs : float
        sampling frequency
    sensor_position : Enum | None
        Position of sensor
    events : array of :py:class:`pigait.data.event_data.GaitEvent`
        Gait events occurring in the data
    length : int
        Length of total signal
    time_vector : array of float
        Signal sample time points represented in seconds
    """

    # Constructor. Data is TxN np arrays, T=samples, N=axes
    def __init__(self, acc_data, gyro_data, fs,
                 mag_data=None, sensor_position=None) -> None:
        """
        Parameters
        ----------
        acc_data : ndarray
            Accelerometer data as TxN np array, T=samples, N=axes
        gyro_data : ndarray
            Gyroscope data as TxN np array, T=samples, N=axes
        mag_data : ndarray
            Magnetometer data as TxN np array, T=samples, N=axes
        """

        # Make sure we get numpy arrays
        _check_numpy_array(acc_data)
        _check_numpy_array(gyro_data)
        if mag_data is not None:
            _check_numpy_array(mag_data)

        # Make sure they're of equal size
        _check_numpy_array(acc_data, gyro_data)
        if mag_data is not None:
            _check_numpy_equal_shape(acc_data, mag_data)

        self.acc_data = acc_data
        self.gyro_data = gyro_data
        self.mag_data = mag_data
        self.velocity = None
        self.position = None
        self.fs = fs
        self.events = []

        if sensor_position:
            self.sensor_position = sensor_position
        else:
            self.sensor_position = SensorPosition.NA

        # Some computed properties
        self._length = self.acc_data.shape[0]
        self._time_vector = np.arange(0, self.length) / self.fs

    @property
    def length(self):
        """
        Get the length of the signal

        Returns
        ----------
        length : int
            Signal length in n samples
        """

        self._length = self.acc_data.shape[0]
        return self._length

    @property
    def time_vector(self):
        """
        Get the sample time points of the signal

        Returns
        ----------
        time_vector : array of float
            Array of time points of samples
        """

        self._time_vector = np.arange(0, self.length) / self.fs
        return self._time_vector

    def _crop_array(self, data, start, end):
        if data is not None:
            data = data[start:end, :]
        return data

    def crop(self, start_time, end_time):
        """
        Crop the data to specified start and end time points

        Parameters
        ----------
        start_time : float
            Starting time point in seconds
        end_time : float
            Ending time point in seconds

        Returns
        ----------
        data : :py:class:`IMUData`
            Cropped instance of data
        """

        start_frame = int(np.floor(start_time * self.fs))
        end_frame = int(np.floor(end_time * self.fs))

        end_frame = min(end_frame, self.length)
        start_frame = max(start_frame, 0)

        self.acc_data = self._crop_array(self.acc_data,
                                         start_frame, end_frame)
        self.gyro_data = self._crop_array(self.gyro_data,
                                          start_frame, end_frame)
        self.mag_data = self._crop_array(self.mag_data,
                                         start_frame, end_frame)

        if self.velocity is not None and self.velocity.any():
            self.velocity = self._crop_array(self.velocity,
                                         start_frame, end_frame)
        if self.position is not None and self.position.any():
            self.position = self._crop_array(self.position,
                                         start_frame, end_frame)

        keep_events = [event for event in self.events if (event.sample_idx >= start_frame and event.sample_idx <= end_frame)]
        for event in keep_events:
            event.sample_idx = event.sample_idx - start_frame
        self.events = keep_events

        return start_frame, end_frame

    def add_event(self, event, sort=False):
        """
        Add a gait event

        Parameters
        ----------
        event : :py:class:`pigait.data.event_data.GaitEvent`
            Event to add
        sort : bool
            Sort events by sample index after adding
        """

        self.events.append(event)
        if sort:
            self.events.sort(key=lambda x: x.sample_idx, reverse=False)

    def add_events(self, events, sort=False):
        """
        Add gait events

        Parameters
        ----------
        events : array of :py:class:`pigait.data.event_data.GaitEvent`
            Events to add
        sort : bool
            Sort events by sample index after adding
        """

        self.events.extend(events)
        if sort:
            self.events.sort(key=lambda x: x.sample_idx, reverse=False)

    def _get_opposite_side(self, side):
        if side == event_data.GaitEventSide.RIGHT:
            return event_data.GaitEventSide.LEFT
        return event_data.GaitEventSide.RIGHT

    # Assign every other event as left/right
    # For example with start right heel strike,
    # assumed order is RHS, LTO, LHS, RTO ...
    def assign_alternating_events(self,
                                  start_side=event_data.GaitEventSide.RIGHT,
                                  start_type=None):
        """
        Assign left and right side to gait events
        in an alternating order

        Parameters
        ----------
        start_side : :py:class:`pigait.data.event_data.GaitEventSide`
            Which side to start assigning from. Events
            occurring before the first event on this side
            will be discarded
        start_type : :py:class:`pigait.data.event_data.GaitEventType`
            Which type to start assigning from. Events
            occurring before the first event of this type
            will be discarded
        """

        # Sort the events
        self.events.sort(key=lambda x: x.sample_idx, reverse=False)

        # Remove any event that is before the starting event type, if any
        if start_type:
            start_event = [event for event in self.events if
                           event.event_type == start_type]
            if len(start_event) < 1:
                raise ValueError(f"No event with {start_type}")
            start_event = start_event[0]
            start_event_sample = start_event.sample_idx
            delete_indices = []
            for event_idx in range(0, len(self.events)):
                if self.events[event_idx].sample_idx < start_event_sample:
                    delete_indices.append(event_idx)
            for del_idx in sorted(delete_indices, reverse=True):
                del self.events[del_idx]

        # Assign alternating sides
        start_type = self.events[0].event_type
        side_start_type = start_side
        side_other_type = self._get_opposite_side(start_side)

        # If we assigned a heel strike one side (e.g. right),
        # then the next toe off should be assigned the other
        # side (e.g. left)
        for event in self.events:
            if event.validity != event_data.GaitEventValidity.VALID:
                event.side = side_start_type
                continue
            if event.event_type == start_type:
                event.side = side_start_type
                side_start_type = self._get_opposite_side(side_start_type)
            else:
                event.side = side_other_type
                side_other_type = self._get_opposite_side(side_other_type)

    def get_events(self, event_type, side=None, sorted=False):
        """
        Get gait events of specified type

        Parameters
        ----------
        event_type : :py:class:`pigait.data.event_data.GaitEventType`
            Which type of gait events to get
        side : :py:class:`pigait.data.event_data.GaitEventSide` | None
            Which side to get events from
        sorted : bool
            Sort returned events

        Returns
        ----------
        events : array of :py:class:`pigait.data.event_data.GaitEvent`
            Events of specified type
        """

        if side:
            events = [event for event in self.events if
                      event.event_type == event_type and event.side == side]
        else:
            events = [event for event in self.events if
                      event.event_type == event_type]
        if sorted:
            events.sort(key=lambda x: x.sample_idx, reverse=False)
        return events

    def get_first_event(self):
        """
        Get first event

        Returns
        ----------
        event : :py:class:`pigait.data.event_data.GaitEvent`
            The first event that occurred (with the lowest sample index)
        """

        events_copy = self.events.copy()
        events_copy.sort(key=lambda x: x.sample_idx, reverse=False)
        return events_copy[0]


class SensorSet:
    """
    A set of multiple sensor data.
    Combines gait events from included :py:class:`IMUData`
    and constructs gait cycles.

    Attributes
    ----------
    sensor_data : array of :py:class:`IMUData`
            Data from sensors
    events : array of :py:class:`pigait.data.event_data.GaitEvent`
            Combined events of all included sensors
    gait_cycles : array of :py:class:`pigait.data.event_data.GaitCycle`
            All gait cycles found in the data
    """

    def __init__(self, sensor_data) -> None:
        """
        Parameters
        ----------
        sensor_data : array of :py:class:`IMUData`
            Data from sensors
        """

        # TODO: check equal lengths
        # TODO: assert one sensor per location max
        self.gait_cycles = []
        self.sensor_data = sensor_data

        self.events = []
        for sensor in sensor_data:
            self.events.extend(sensor.events)
        if len(self.events) > 0:
            self.events.sort(key=lambda x: x.sample_idx, reverse=False)

        self._construct_cycles()

    def _construct_cycle(self, start_event, list_hs, list_to,
                         list_hs_opposite, list_to_opposite):

        # Make sure we have np arrays for easier comparison
        list_hs = np.array(list_hs)
        list_to = np.array(list_to)
        list_hs_opposite = np.array(list_hs_opposite)
        list_to_opposite = np.array(list_to_opposite)

        # Start from HS, seek e.g. for right starting side
        # RHS LTO LHS RTO RHS ...
        hs_start = start_event

        sample_indices = [event.sample_idx for event in list_to_opposite]
        to_opposite = list_to_opposite[sample_indices > hs_start.sample_idx]
        if len(to_opposite) < 1:
            return None
        to_opposite = to_opposite[0]

        sample_indices = [event.sample_idx for event in list_hs_opposite]
        hs_opposite = list_hs_opposite[sample_indices > to_opposite.sample_idx]
        if len(hs_opposite) < 1:
            return None
        hs_opposite = hs_opposite[0]

        sample_indices = [event.sample_idx for event in list_to]
        to = list_to[sample_indices > hs_opposite.sample_idx]
        if len(to) < 1:
            return None
        to = to[0]

        sample_indices = [event.sample_idx for event in list_hs]
        hs_end = list_hs[sample_indices > to.sample_idx]
        if len(hs_end) < 1:
            return None
        hs_end = hs_end[0]

        # If either HS event is invalid, skip this cycle
        if (hs_start.validity != event_data.GaitEventValidity.VALID
                or hs_end.validity != event_data.GaitEventValidity.VALID):
            return None

        cycle = event_data.GaitCycle(hs_start, to_opposite, hs_opposite,
                                     to, hs_end)
        return cycle

    # TODO: assumptions, asserts
    def _construct_cycles(self):
        all_lhs = [event for event in self.events if (
            event.event_type == event_data.GaitEventType.HEEL_STRIKE
            and event.side == event_data.GaitEventSide.LEFT)]
        all_rhs = [event for event in self.events if (
            event.event_type == event_data.GaitEventType.HEEL_STRIKE
            and event.side == event_data.GaitEventSide.RIGHT)]
        all_lto = [event for event in self.events if (
            event.event_type == event_data.GaitEventType.TOE_OFF
            and event.side == event_data.GaitEventSide.LEFT)]
        all_rto = [event for event in self.events if (
            event.event_type == event_data.GaitEventType.TOE_OFF
            and event.side == event_data.GaitEventSide.RIGHT)]

        # Construct gait cycles from earliest starting HS event.
        gait_cycles = []
        if len(all_rhs) < 1 or len(all_lhs) < 1:
            self.gait_cycles = gait_cycles
            return
        if all_rhs[0].sample_idx < all_lhs[0].sample_idx:
            for hs_idx in range(0, len(all_rhs)):
                hs = all_rhs[hs_idx]
                step = self._construct_cycle(hs, all_rhs, all_rto, all_lhs,
                                             all_lto)
                if step:
                    gait_cycles.append(step)
        else:
            for hs_idx in range(0, len(all_lhs)):
                hs = all_lhs[hs_idx]
                step = self._construct_cycle(hs, all_lhs, all_lto, all_rhs,
                                             all_rto)
                if step:
                    gait_cycles.append(step)
        self.gait_cycles = gait_cycles

    def plot_raw(self, axis=0, data_type="gyro"):
        """
        Plots data from all sensors in the set,
        including any events

        Parameters
        ----------
        axis : int
            Which axis in sensors to plot
        data_type : string
            Which type of data to plot.
            Choices: gyro, acc, mag
        """
        plot_signal.plot_sensors(self, axis=axis, data_type=data_type)
