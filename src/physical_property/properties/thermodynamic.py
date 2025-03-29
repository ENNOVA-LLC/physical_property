""" `properties.thermodynamic`

Contains classes for thermodynamic properties (e.g., temperature, pressure, molar mass).
"""
import attr
from typing import Tuple, ClassVar
import numpy as np

from ..base import PhysicalProperty

@attr.s(auto_attribs=True)
class Mass(PhysicalProperty):
    """Mass property (base unit: kg)."""
    def convert(self, to_unit: str) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        return self.converter.mass(self.value, self.unit, to_unit)
    
    def to_moles(self, MW: float) -> 'Moles':
        """
        Convert mass to moles using the molar mass.
        """
        return Moles(name=self.name, value=self.value / MW, unit="mol", doc=self.doc)

@attr.s(auto_attribs=True)
class Moles(PhysicalProperty):
    """Moles property (base unit: mol)."""
    def convert(self, to_unit: str) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        return self.converter.mol(self.value, self.unit, to_unit)
    
    def to_mass(self, MW: float) -> 'Mass':
        """
        Convert moles to mass using the molar mass.
        """
        MW /= 1000  # Convert from g/mol to kg/mol
        return Mass(name=self.name, value=self.value * MW, unit="kg", doc=self.doc)

@attr.s(auto_attribs=True)
class CompositionArray(PhysicalProperty):
    """Composition property (e.g., mole percent, potentially requiring MW for mass-mole conversion)."""
    def convert(self, to_unit: str, MW=None) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        return self.converter.convert("composition", self.value, self.unit, to_unit, MW=MW)
    
@attr.s(auto_attribs=True)
class MolarMass(PhysicalProperty):
    """Molar mass property (base unit: g/mol)."""
    def convert(self, to_unit: str) -> np.ndarray:
        if self.unit == to_unit:
            return self.value

        try:
            mass_from, mole_from = self.unit.split('/')
            mass_to, mole_to = to_unit.split('/')
        except Exception as e:
            raise ValueError("Molar mass units must be in 'mass/mole' format (e.g., 'g/mol').") from e

        # Convert a unit value of 1 for mass and mole parts
        mass_factor = self.converter.mass(1, mass_from, mass_to)
        mole_factor = self.converter.mol(1, mole_from, mole_to)

        # Adjust the molar mass: new_value = original_value * (mass conversion factor / mole conversion factor)
        return self.value * (mass_factor / mole_factor)

@attr.s(auto_attribs=True)
class Temperature(PhysicalProperty):
    """Temperature property (base unit: Kelvin)."""
    DEFAULT_BOUNDS: ClassVar[Tuple[float, float]] = (0, 800)
    def convert(self, to_unit: str) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        return self.converter.temperature(self.value, self.unit, to_unit)

@attr.s(auto_attribs=True)
class Pressure(PhysicalProperty):
    """Pressure property (base unit: bar)."""
    DEFAULT_BOUNDS: ClassVar[Tuple[float, float]] = (0, 3500)
    def convert(self, to_unit: str) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        return self.converter.pressure(self.value, self.unit, to_unit)

@attr.s(auto_attribs=True)
class Density(PhysicalProperty):
    """Density property (base unit: kg/m^3)."""
    DEFAULT_BOUNDS: ClassVar[Tuple[float, float]] = (0, 2000)
    def convert(self, to_unit: str) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        return self.converter.density(self.value, self.unit, to_unit)

@attr.s(auto_attribs=True)
class Volume(PhysicalProperty):
    """Volume property (base unit: m^3)."""
    DEFAULT_BOUNDS: ClassVar[Tuple[float, float]] = (0, None)
    def convert(self, to_unit: str) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        return self.converter.volume(self.value, self.unit, to_unit)

@attr.s(auto_attribs=True)
class SolubilityParameter(PhysicalProperty):
    """SolubilityParameter property (base unit: MPa**0.5)."""
    DEFAULT_BOUNDS: ClassVar[Tuple[float, float]] = (0, 50.)
    
