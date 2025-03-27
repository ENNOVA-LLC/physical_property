""" `properties.specialty.gas_oil_ratio`

Gas-Oil Ratio property supporting conversion among volumetric, mass, and molar ratios.
"""
import attr
from typing import Union, Dict, Any, Optional, ClassVar
import numpy as np

from ...base import PhysicalProperty
from ..thermodynamic import Density, MolarMass
from ...units import UnitConverter
unit_converter = UnitConverter(unit_set='standard')
UNITS_DICT = unit_converter.UNITS_DICT

@attr.s(auto_attribs=True)
class GasOilRatio(PhysicalProperty):
    """
    Gas-Oil Ratio property supporting conversion among volumetric, mass, and molar ratios.

    Recognized unit formats (case-insensitive) are expressed as "gas_unit/oil_unit". For example:
    - Volumetric: "ft3/bbl", "m3/m3"
    - Mass: "kg/kg" (or similar variants like "kg gas/kg oil")
    - Molar: "mol/mol" (or similar variants)

    Conversions between these representations require additional data:
    - Volumetric ⇄ Mass: provide gas_density and oil_density (in consistent units, e.g. kg/m³)
    - Mass ⇄ Molar: provide gas_MW and oil_MW (molecular weights)
    """
    UNITS: ClassVar[Dict[str, Any]] = {
        'volume': UNITS_DICT["volume"],
        'mass':   UNITS_DICT["mass"],
        'mole':   UNITS_DICT["mol"],
    }

    def _get_ratio_type(self, unit: str) -> str:
        """Utility to determine the type of GasOilRatio (volume, mass, or mole) from its unit."""
        u = unit.lower()
        gas_unit, oil_unit = u.split("/")
        if gas_unit in self.UNITS['volume'] and oil_unit in self.UNITS['volume']:
            return "volume"
        elif gas_unit in self.UNITS['mass'] and oil_unit in self.UNITS['mass']:
            return "mass"
        elif gas_unit in self.UNITS['mole'] and oil_unit in self.UNITS['mole']:
            return "mole"
        else:
            raise ValueError(f"Unrecognized unit type for GasOilRatio: {unit}")

    def _convert_within_volumetric(self, value: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
        gas_from, oil_from = from_unit.lower().split("/")
        gas_to, oil_to = to_unit.lower().split("/")
        factor_gas = unit_converter.volume(1.0, gas_from, gas_to)
        factor_oil = unit_converter.volume(1.0, oil_from, oil_to)
        factor = factor_gas / factor_oil
        return value * factor

    def _convert_within_mass(self, value: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
        gas_from, oil_from = from_unit.lower().split("/")
        gas_to, oil_to = to_unit.lower().split("/")
        factor_gas = unit_converter.mass(1.0, gas_from, gas_to)
        factor_oil = unit_converter.mass(1.0, oil_from, oil_to)
        factor = factor_gas / factor_oil
        return value * factor

    def _convert_within_molar(self, value: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
        gas_from, oil_from = from_unit.lower().split("/")
        gas_to, oil_to = to_unit.lower().split("/")
        factor_gas = unit_converter.mol(1.0, gas_from, gas_to)
        factor_oil = unit_converter.mol(1.0, oil_from, oil_to)
        factor = factor_gas / factor_oil
        return value * factor

    def convert(self, to_unit: str, *,
                gas_density: Optional[Union[Density, float, np.ndarray]] = None,
                oil_density: Optional[Union[Density, float, np.ndarray]] = None,
                gas_MW: Optional[Union[MolarMass, float]] = None,
                oil_MW: Optional[Union[MolarMass, float]] = None) -> np.ndarray:
        """
        Convert the GasOilRatio value from `self.unit` to `to_unit`.

        Parameters
        ----------
        to_unit : str
            Target unit for conversion, specified in the format "gas_unit/oil_unit".
            Supported options include:
            - Volumetric: e.g., "ft3/bbl", "m3/m3", "mL/mL"
            - Mass: e.g., "kg/kg", "lb/lb"
            - Molar: e.g., "mol/mol"
        gas_density : Density or float or np.ndarray, optional
            Gas density required for volumetric ⇄ mass conversion (e.g., in kg/m³).
        oil_density : Density or float or np.ndarray, optional
            Oil density required for volumetric ⇄ mass conversion (e.g., in kg/m³).
        gas_MW : MolarMass or float, optional
            Gas molecular weight required for mass ⇄ molar conversion.
        oil_MW : MolarMass or float, optional
            Oil molecular weight required for mass ⇄ molar conversion.

        Returns
        -------
        np.ndarray
            Converted GasOilRatio value in the target unit.

        Raises
        ------
        ValueError
            If conversion cannot be performed due to missing parameters or unsupported unit types.
        """
        # Check if no conversion is required
        if self.unit.lower() == to_unit.lower():
            return self.value
        
        # Check for correct units
        from_type = self._get_ratio_type(self.unit)
        to_type = self._get_ratio_type(to_unit)
        
        # Convert density and molar mass inputs to appropriate types if provided
        if isinstance(gas_density, Density):
            gas_density = gas_density.convert("kg/m3")
        if isinstance(oil_density, Density):
            oil_density = oil_density.convert("kg/m3")
        if isinstance(gas_MW, MolarMass):
            gas_MW = gas_MW.convert("g/mol")
        if isinstance(oil_MW, MolarMass):
            oil_MW = oil_MW.convert("g/mol")
        
        # Perform the conversion
        intermediate = self.value
        # -- Volumetric ⇄ Volumetric, Mass ⇄ Mass, Molar ⇄ Molar
        if from_type == to_type:
            if from_type == "volume":
                return self._convert_within_volumetric(intermediate, self.unit, to_unit)
            elif from_type == "mass":
                return self._convert_within_mass(intermediate, self.unit, to_unit)
            elif from_type == "mole":
                return self._convert_within_molar(intermediate, self.unit, to_unit)
        
        # -- Volumetric ⇄ Mass, Molar ⇄ Mass
        if from_type == "volume":
            if gas_density is None or oil_density is None:
                raise ValueError("Gas and oil densities must be provided for volumetric to mass conversion.")
            intermediate = intermediate * (gas_density / oil_density)
        elif from_type == "mole":
            if gas_MW is None or oil_MW is None:
                raise ValueError("Gas and oil molecular weights must be provided for molar to mass conversion.")
            intermediate = intermediate * (gas_MW / oil_MW)
        
        # -- Mass ⇄ Volumetric, Mass ⇄ Molar
        if to_type == "mass":
            result = intermediate
            result = self._convert_within_mass(result, "kg/kg", to_unit)
        elif to_type == "volume":
            if gas_density is None or oil_density is None:
                raise ValueError("Gas and oil densities must be provided for mass to volumetric conversion.")
            
            result = intermediate * (oil_density / gas_density)
            result = self._convert_within_volumetric(result, "m3/m3", to_unit)
        elif to_type == "mole":
            if gas_MW is None or oil_MW is None:
                raise ValueError("Gas and oil molecular weights must be provided for mass to molar conversion.")
            result = intermediate * (oil_MW / gas_MW)
            result = self._convert_within_molar(result, "mol/mol", to_unit)
        else:
            raise ValueError(f"Unsupported target ratio type: {to_type}")
        return result

    def to(self, to_unit: str, **kwargs) -> 'GasOilRatio':
        """
        Convert the GasOilRatio value from `self.unit` to `to_unit`.

        Parameters
        ----------
        to_unit : str
            Target unit for conversion, specified in the format "gas_unit/oil_unit".
            Supported options include:
            - Volumetric: e.g., "ft3/bbl", "m3/m3", "mL/mL"
            - Mass: e.g., "kg/kg", "lb/lb"
            - Molar: e.g., "mol/mol"
        gas_density : Density or float or np.ndarray, optional
            Gas density required for volumetric ⇄ mass conversion (e.g., in kg/m³).
        oil_density : Density or float or np.ndarray, optional
            Oil density required for volumetric ⇄ mass conversion (e.g., in kg/m³).
        gas_MW : MolarMass or float, optional
            Gas molecular weight required for mass ⇄ molar conversion.
        oil_MW : MolarMass or float, optional
            Oil molecular weight required for mass ⇄ molar conversion.

        Returns
        -------
        GasOilRatio
            New GasOilRatio instance with value converted to the target unit.

        Raises
        ------
        ValueError
            If conversion cannot be performed due to missing parameters or unsupported unit types.
        """
        new_value = self.convert(to_unit, **kwargs)
        return GasOilRatio(name=self.name, value=new_value, unit=to_unit, doc=self.doc, bounds=self.bounds)

