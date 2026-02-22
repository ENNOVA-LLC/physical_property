""" `properties.transport`

Contains classes for transport properties (e.g., viscosity, shear stress).
"""
from attrs import define
from typing import Optional
import numpy as np

from ..base import PhysicalProperty
from .thermodynamic import Density

@define
class Flow(PhysicalProperty):
    """Flow rate property (base unit: kg/s for mass, m³/s for volume)."""
    def convert(self, to_unit: Optional[str], density: Optional[Density] = None) -> np.ndarray:
        """
        Convert flow rate to a new unit, handling mass and volume flows.

        Parameters
        ----------
        to_unit : str, optional
            The target unit.
        density : Density, optional
            Density property for conversion from mass to volume flow and vice versa.

        Returns
        -------
        np.ndarray
            The converted flow rate.
        """
        if self.unit == to_unit:
            return self.value
        if to_unit is None and self.unit is not None:
            raise ValueError("Cannot convert between dimensionless and dimensional units.")
        if density is None:
            density = Density(name="density", value=0.0, unit="kg/m3")
        return self.converter.flow(self.value, self.unit, to_unit, dens=density.value, dens_unit=density.unit)

    def to(self, to_unit: str, density: Optional[Density] = None) -> 'Flow':
        """
        Return a new Flow instance with flow rate converted to the new unit.

        Parameters
        ----------
        to_unit : str
            The target unit.
        density : Density, optional
            Density property for conversion.
        
        Returns
        -------
        Flow
            A new Flow instance with converted value.
        """
        return self.__class__(name=self.name, value=self.convert(to_unit, density), unit=to_unit, doc=self.doc)

@define
class Viscosity(PhysicalProperty):
    """Viscosity property (base unit: Pa*s)."""
    def convert(self, to_unit: Optional[str]) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        if to_unit is None and self.unit is not None:
            raise ValueError("Cannot convert between dimensionless and dimensional units.")
        return self.converter.viscosity(self.value, self.unit, to_unit)
    
@define
class ShearStress(PhysicalProperty):
    """Shear stress property (base unit: bar)."""
    def convert(self, to_unit: Optional[str]) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        if to_unit is None and self.unit is not None:
            raise ValueError("Cannot convert between dimensionless and dimensional units.")
        return self.converter.pressure(self.value, self.unit, to_unit)

@define
class PressureGradient(PhysicalProperty):
    """Pressure gradient property (base unit: bar/m)."""
    def convert(self, to_unit: Optional[str]) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        if to_unit is None and self.unit is not None:
            raise ValueError("Cannot convert between dimensionless and dimensional units.")
        
        try:
            P_from, L_from = self.unit.split('/')
            P_to, L_to = to_unit.split('/')
        except Exception as e:
            raise ValueError("Pressure gradient units must be in 'pressure/length' format (e.g., 'bar/m').") from e

        pressure_factor = self.converter.pressure(1, P_from, P_to)
        length_factor = self.converter.length(1, L_from, L_to)

        # Adjust the pressure gradient: new_value = original_value * (pressure conversion factor / length conversion factor)
        return self.value * (pressure_factor / length_factor)

@define
class PressureDrop(PhysicalProperty):
    """Pressure drop property (base unit: bar)."""
    def convert(self, to_unit: Optional[str]) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        if to_unit is None and self.unit is not None:
            raise ValueError("Cannot convert between dimensionless and dimensional units.")
        
        pressure_factor = self.converter.convert("pressure", 1, self.unit, to_unit)
        return self.value * (pressure_factor)
    
@define
class SurfaceTension(PhysicalProperty):
    """SurfaceTension property (base unit: N/m)."""
    def convert(self, to_unit: Optional[str]) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        if to_unit is None and self.unit is not None:
            raise ValueError("Cannot convert between dimensionless and dimensional units.")
        return self.converter.viscosity(self.value, self.unit, to_unit)
        
    @classmethod
    def from_parachor(cls, parachor, density_liq: Density, density_vap: Density) -> 'SurfaceTension':
        """Create instance of parachor-based surface tension.

        Parameters
        ----------
        parachor : np.ndarray | float
            Parachor parameter.
        density_liq : Density | float
            Density of the liquid phase (kg/m3).
        density_vap : Density | float
            Density of the vapor phase (kg/m3).
        """
        instance = cls(name="Surface tension", value=0.0, unit='N/m')
        instance.update_from_parachor(parachor, density_liq, density_vap)
        return instance

    def update_from_parachor(self, parachor, density_liq: Density, density_vap: Density) -> None:
        """Update the surface tension from the parachor, liquid density, and vapor density.
        
        Parameters
        ----------
        parachor : np.ndarray | float
            Parachor parameter.
        density_liq : Density | float
            Density of the liquid phase (kg/m3).
        density_vap : Density | float
            Density of the vapor phase (kg/m3).
        """
        rho_liq = density_liq.convert("kg/m3") if isinstance(density_liq, Density) else density_liq
        rho_vap = density_vap.convert("kg/m3") if isinstance(density_vap, Density) else density_vap
        surface_tension = parachor * ((rho_liq - rho_vap) * 1e-3) ** 4
        self.update_value(surface_tension * 0.001)  # Convert from mN/m to N/m

@define
class MassFlux(PhysicalProperty):
    """Mass flux property (base unit: kg/m²·s)."""
    def convert(self, to_unit: Optional[str]) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        return self.converter.convert("mass_flux", self.value, self.unit, to_unit)

@define
class Diffusivity(PhysicalProperty):
    """Diffusivity property (base unit: m²/s)."""
    def convert(self, to_unit: Optional[str]) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        return self.converter.convert("diffusivity", self.value, self.unit, to_unit)

@define
class Dispersion(PhysicalProperty):
    """Dispersion property (base unit: m²/s)."""
    def convert(self, to_unit: Optional[str]) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        return self.converter.convert("dispersion", self.value, self.unit, to_unit)

@define
class ThermalConductivity(PhysicalProperty):
    """Thermal conductivity property (base unit: k=W/m·K)."""
    def convert(self, to_unit: Optional[str]) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        return self.converter.convert("thermal_conductivity", self.value, self.unit, to_unit)

@define
class HeatTransferCoefficient(PhysicalProperty):
    """Heat transfer coefficient property (base unit: h=W/m²·K)."""
    def convert(self, to_unit: Optional[str]) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        return self.converter.convert("heat_transfer_coefficient", self.value, self.unit, to_unit)

@define
class HeatCapacity(PhysicalProperty):
    """Heat capacity property (base unit: Cp=J/kg·K)."""
    def convert(self, to_unit: Optional[str]) -> np.ndarray:
        if self.unit == to_unit:
            return self.value
        return self.converter.convert("heat_capacity", self.value, self.unit, to_unit)
