"""
    Data classes for holding data about events
"""

from enum import Enum


class GaitEventType(Enum):
    """
    Enum for type of gait event
    """

    HEEL_STRIKE = 1
    """
    A heel strike, or initial contact
    """
    TOE_OFF = 2
    """
    A toe off, or final contact
    """
    MID_SWING = 3
    """
    The moment the limb is in mid-swing
    """
    FOOT_FLAT = 4
    """
    The moment the foot is still
    """

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        return NotImplemented


class GaitEventSide(Enum):
    """
    Enum for which side an event occurred on
    """

    NA = 0
    """
    Not defined
    """
    LEFT = 1
    """
    Left side or left foot
    """
    RIGHT = 2
    """
    Right side or right foot
    """

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        return NotImplemented


class GaitEventValidity(Enum):
    """
    Enum for whether the event is valid to some criterion
    """

    VALID = 0
    """
    Event is valid
    """
    OUT_OF_ORDER = 1
    """
    Event was not in the expected order
    """


class GaitEvent:
    """
    A gait event during a walking bout

    Attributes
    ----------
    type : :py:class:`GaitEventType`
        Type of event
    side : :py:class:`GaitEventSide`
        Which side the event occurred on
    sample_idx : int
        Which sample the event occurred on
    validity : :py:class:`GaitEventValidity`
        Validity of the event
    """

    # Constructor
    def __init__(self, event_type, side, sample_idx,
                 validity=GaitEventValidity.VALID) -> None:
        self.event_type = event_type
        self.side = side
        self.sample_idx = sample_idx
        self.validity = validity

    def __str__(self):
        return (f"Event at idx {self.sample_idx}: side {self.side}, "
                f"type {self.event_type}, valid {self.validity}")

    def get_short_str(self):
        """
        Get a short string representation of the event
        """
        return f"{self.event_type}, {self.side}"


class GaitCycle:
    """
    A complete gait cycle involving
    events on both feet in order

    Attributes
    ----------
    hs_start : :py:class:`GaitEvent`
        A heel-strike, 1st event
    to_opposite : :py:class:`GaitEvent`
        A toe-off on opposite foot, 2nd event
    hs_opposite : :py:class:`GaitEvent`
        A heel-strike on opposite foot, 3rd event
    to : :py:class:`GaitEvent`
        A toe-off on start foot, 4th event
    hs_end : :py:class:`GaitEvent`
        A heel-strike on start foot, 5th event
    """

    def __init__(self, hs_start, to_opposite, hs_opposite, to, hs_end):
        self.hs_start = hs_start
        self.to_opposite = to_opposite
        self.hs_opposite = hs_opposite
        self.to = to
        self.hs_end = hs_end

    def __str__(self):
        return (f"Gait cycle:\n{self.hs_start}\n{self.to_opposite}\n"
                f"{self.hs_opposite}\n{self.to}\n{self.hs_end}")
