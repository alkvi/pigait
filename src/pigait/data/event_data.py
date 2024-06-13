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
                "type {self.event_type}, valid {self.validity}")
