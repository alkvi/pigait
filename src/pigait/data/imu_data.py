"""
    Data classes for holding IMU data
"""

import numpy as np
from ..data import event_data


def _check_numpy_array(item, item_name="data"):
    if not isinstance(item, (np.ndarray)):
        raise TypeError((f"{item_name} must be"
                        "given as ndarray, got {type(item)}"))


def _check_numpy_equal_shape(array_1, array_2):
    ax_1_eq = array_1.shape[0] == array_2.shape[0]
    ax_2_eq = array_1.shape[1] == array_2.shape[1]
    if not ax_1_eq or not ax_2_eq:
        raise TypeError("Given arrays are not equal shape")


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
    sensor_position : string | None
        Position of sensor
    events : array of GaitEvent
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
            self.sensor_position = "NA"

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
        data : IMUData
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

        # TODO: take care of events outside crop
        return start_frame, end_frame

    def add_event(self, event):
        """
        Add a gait event

        Parameters
        ----------
        event : GaitEvent
            Event to add
        """

        self.events.append(event)

    def add_events(self, events):
        """
        Add gait events

        Parameters
        ----------
        events : array of GaitEvent
            Events to add
        """

        self.events.extend(events)

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
        start_side : GaitEventSide
            Which side to start assigning from. Events
            occurring before the first event on this side
            will be discarded
        start_type : GaitEventType
            Which type to start assigning from. Events
            occurring before the first event on this side
            will be discarded
        """

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
        for event in self.events:
            if event.event_type == start_type:
                event.side = side_start_type
            else:
                event.side = side_other_type
            # If we assigned a heel strike one side (e.g. right),
            # then the next toe off should be assigned the other
            # side (e.g. left)
            side_start_type = self._get_opposite_side(side_start_type)
            side_other_type = self._get_opposite_side(side_other_type)

        # Sort the events
        self.events.sort(key=lambda x: x.sample_idx, reverse=False)

    def get_events(self, event_type, side=None):
        """
        Get gait events of specified type

        Parameters
        ----------
        event_type : GaitEventType
            Which type of gait events to get
        side : GaitEventSide | None
            Which side to get events from

        Returns
        ----------
        events : array of GaitEvent
            Events of specified type
        """

        if side:
            events = [event for event in self.events if
                      event.event_type == event_type and event.side == side]
        else:
            events = [event for event in self.events if
                      event.event_type == event_type]
        return events

    def get_first_event(self):
        """
        Get first event

        Returns
        ----------
        event : GaitEvent
            The first event that occurred (with the lowest sample index)
        """

        events_copy = self.events.copy()
        events_copy.sort(key=lambda x: x.sample_idx, reverse=False)
        return events_copy[0]
