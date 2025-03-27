""" `properties.basic`

Basic (or fundamental) physical properties (time, length, area, velocity, angle).
"""
import attr
import numpy as np

from ..base import PhysicalProperty
from ..units import UnitConverter
unit_converter = UnitConverter(unit_set='standard')

@attr.s(auto_attribs=True)
class Time(PhysicalProperty):
    """Time property (base unit: s)."""
    def convert(self, to_unit: str) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        return unit_converter.time(self.value, self.unit, to_unit)

@attr.s(auto_attribs=True)
class Length(PhysicalProperty):
    """Length property (base unit: meters)."""
    def convert(self, to_unit: str) -> np.ndarray:
        if self.unit == to_unit or (self.unit is None and to_unit is None):
            return self.value
        return unit_converter.length(self.value, self.unit, to_unit)

@attr.s(auto_attribs=True)
class Rate(PhysicalProperty):
    """Rate property (base unit: 1/s)."""
    def convert(self, to_unit: str) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        # Extract the time units from the rate unit format (e.g., "1/s" or "1/min")
        _, unit_in = self.unit.split("/")
        _, unit_out = to_unit.split("/")
        # Convert the rate by converting the time units and taking the reciprocal
        converted_time = unit_converter.time(1.0, unit_out, unit_in)
        return self.value * converted_time

@attr.s(auto_attribs=True)
class Area(PhysicalProperty):
    """Area property (base unit: m^2)."""
    def convert(self, to_unit: str) -> np.ndarray:
        if self.unit == to_unit or (self.unit is None and to_unit is None):
            return self.value
        return unit_converter.convert_x("area", self.value, self.unit, to_unit)

@attr.s(auto_attribs=True)
class Angle(PhysicalProperty):
    """Angle property (base unit: radians).
    
    Supports units: ["rad", "deg"].
    """
    is_reference_vertical: bool = True

    def convert(self, to_unit: str) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        if self.unit == "rad" and to_unit == "deg":
            return np.degrees(self.value)
        elif self.unit == "deg" and to_unit == "rad":
            return np.radians(self.value)
        raise ValueError(f"Conversion from {self.unit} to {to_unit} not supported for Angle.")

    def to_horizontal(self) -> 'Angle':
        """
        Return a new Angle instance with the reference changed to horizontal.
        If the current angle is measured from vertical, convert it such that:
            horizontal_angle = 90° - vertical_angle (or π/2 - vertical_angle in radians).
        """
        if not self.is_reference_vertical:
            return self.copy()
        if self.unit == "deg":
            new_value = 90.0 - self.value
        elif self.unit == "rad":
            new_value = (np.pi / 2) - self.value
        else:
            raise ValueError("Angle conversion to horizontal is only supported for 'deg' and 'rad'.")
        return Angle(
            name=self.name,
            value=new_value,
            unit=self.unit,
            doc=self.doc,
            bounds=self.bounds,
            is_reference_vertical=False
        )
    
    def to_vertical(self) -> 'Angle':
        """
        Return a new Angle instance with the reference changed to vertical.
        If the current angle is measured from horizontal, convert it such that:
            vertical_angle = 90° - horizontal_angle (or π/2 - horizontal_angle in radians).
        """
        if self.is_reference_vertical:
            return self.copy()
        if self.unit == "deg":
            new_value = 90.0 - self.value
        elif self.unit == "rad":
            new_value = (np.pi / 2) - self.value
        else:
            raise ValueError("Angle conversion to vertical is only supported for 'deg' and 'rad'.")
        return Angle(
            name=self.name,
            value=new_value,
            unit=self.unit,
            doc=self.doc,
            bounds=self.bounds,
            is_reference_vertical=True
        )
        
@attr.s(auto_attribs=True)
class Velocity(PhysicalProperty):
    """Velocity property (base unit: m/s)."""
    def convert(self, to_unit: str) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        return unit_converter.velocity(self.value, self.unit, to_unit)
