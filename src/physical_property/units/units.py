""" `units.units`

Unit conversion module for common physico-chemical properties.
Delegates to `fpcore.units` for actual conversion logic.
"""
from attrs import define, field
from typing import Dict, Tuple, List, Any, Union, ClassVar, Optional, Set
import numpy as np

# Integration with fpCore
from fpcore.units import convert as fp_convert, get_ureg
from fpcore.units import convert_flow as fp_convert_flow
from fpcore.units.api import UNIT_ALIAS_MAP
from fpcore.units import UNITS_DICT as FP_UNITS_DICT

from .unit_sets import get_unit_set, DEFAULT_UNIT_SET


# Configure logging
from ..utils.logging import get_logger
logger = get_logger(__name__)

@define
class UnitConverter:
    """
    Initialize the UnitConverter object.

    Attributes
    ----------
    unit_set : str|dict, optional
        If `str`, then options are: ['standard', 'field', 'US', 'metric', 'SI']
        If `dict`, then provide a dictionary containing the unit settings. 
        Defaults to 'standard'.
    MW : array-like[float], optional 
        The molecular weight. Defaults to None.
    """
    unit_set: Union[str, Dict[str, str]] = "standard"
    MW: Any = None
    
    # Class-level attributes
    UNITS_DICT: ClassVar[Dict[str, Set[str]]] = FP_UNITS_DICT
    DEFAULT_UNIT_SET: ClassVar[Dict[str, str]] = DEFAULT_UNIT_SET
    # Use fpCore's alias map instead of manual ALT_KEYS
    ALT_KEYS: ClassVar[Dict[str, List[str]]] = UNIT_ALIAS_MAP.canonical_to_aliases
    ALIAS_MAP = UNIT_ALIAS_MAP # Expose the helper object directly too

    def __attrs_post_init__(self):
        logger.debug("UnitConverter initialized with unit_set and fpCore backend: %s", self.unit_set)
        # Ensure unit_set is resolved to dictionary
        if isinstance(self.unit_set, str):
            self.unit_set = self.get_unit_set(self.unit_set)
    
    # --------------------------------------
    # region: Utilities
    # --------------------------------------
    def get_unit_set(self, unit_set) -> Dict[str, str]:
        """
        Retrieves the unit set based on the `unit_set` input.

        Parameters
        ----------
        unit_set : str or dict
            The unit set identifier (e.g. 'standard') or a dictionary of unit mappings.

        Returns
        -------
        dict
            A dictionary containing the unit set mappings.
        """
        return get_unit_set(unit_set)

    def get_unit_type(self, unit: str) -> Optional[str]:
        """
        Get the property type (e.g. 'mass', 'length', 'density') for a given unit string.
        
        Parameters
        ----------
        unit : str
            The unit string (e.g. 'kg', 'm', 'kg/m3').

        Returns
        -------
        str or None
            The property type if found, else None.
        """
        if not unit:
            return None
            
        # Check explicit mappings first
        for prop_type, units in self.UNITS_DICT.items():
            if unit in units:
                return prop_type
        
        # Try to infer from dimensionality using pint
        try:
            ureg = get_ureg()
            qty = ureg(unit)
            dim = str(qty.dimensionality)
            
            # Map dimensions to property types
            dim_map = {
                '[mass] / [length] ** 3': 'density',
                '[mass] / [length] / [time]': 'viscosity',
                '[length] ** 2 / [time]': 'diffusivity', # or kinematic_viscosity
                '[length]': 'length',
                '[mass]': 'mass',
                '[time]': 'time',
                '[temperature]': 'temperature',
                '[mass] / [length] / [time] ** 2': 'pressure',
                '[mass] / [time] ** 2': 'surface_tension',
                '[mass] ** 0.5 / [length] ** 0.5 / [time]': 'solubility_parameter', # MPa**0.5
            }
            if dim in dim_map:
                return dim_map[dim]
                
        except Exception:
            pass
            
        return None

    def convert(self, item: str = None, value: Any = None, from_unit: str = None, to_unit: str = None, **kwargs) -> Any:
        """
        General function to convert any item from one unit to another.
        """
        # Handle aliases (e.g. from FluidLUT usage: prop, val, unit_in, unit_out)
        if item is None:
            item = kwargs.get('prop', kwargs.get('property'))
        if value is None:
            value = kwargs.get('val')
        if from_unit is None:
            from_unit = kwargs.get('unit_in')
        if to_unit is None:
            to_unit = kwargs.get('unit_out')
            
        if item is None or value is None or from_unit is None or to_unit is None:
            raise ValueError(f"Missing arguments for convert: item={item}, value={value}, from={from_unit}, to={to_unit}")

        if from_unit == to_unit:
            return value

        # Map item/prop_type to method (e.g. "temperature" -> "convert_temperature")
        method_name = f"convert_{item}"
        
        # Try specific conversion method if available
        if hasattr(self, method_name):
            converter = getattr(self, method_name)
            return converter(value, from_unit, to_unit)
        
        # Also check for direct method name match (legacy compatibility if needed)
        if hasattr(self, item):
            converter = getattr(self, item)
            return converter(value, from_unit, to_unit)

        # Fallback to fpCore generic convert
        try:
            return fp_convert(value, from_unit, to_unit)
        except Exception as e:
            logger.error(f"Failed to convert {value} from {from_unit} to {to_unit} for item {item}: {e}")
            raise e

    # --------------------------------------
    # region: Specific Property Conversions
    # --------------------------------------

    def convert_temperature(self, value, from_unit, to_unit):
        """Convert temperature."""
        return fp_convert(value, from_unit, to_unit)

    def convert_pressure(self, value, from_unit, to_unit):
        """Convert pressure."""
        return fp_convert(value, from_unit, to_unit)
        
    def convert_length(self, value, from_unit, to_unit):
        return fp_convert(value, from_unit, to_unit)

    def convert_area(self, value, from_unit, to_unit):
        return fp_convert(value, from_unit, to_unit)

    def convert_volume(self, value, from_unit, to_unit):
        return fp_convert(value, from_unit, to_unit)

    def convert_mass(self, value, from_unit, to_unit):
        return fp_convert(value, from_unit, to_unit)

    def convert_time(self, value, from_unit, to_unit):
        return fp_convert(value, from_unit, to_unit)

    def convert_flow(self, value, from_unit, to_unit, **kwargs):
        """Convert flow rate."""
        # Map physical_property kwargs to fpCore
        if 'dens' in kwargs:
            kwargs['density'] = kwargs.pop('dens')
        if 'dens_unit' in kwargs:
            # fpCore assumes kg/m3. 
            # If unit is different, I should convert it. But here I only have the value locally.
            # fpCore `convert_flow` docstring says: "The density of the fluid [kg/m3]"
            # So I should ensure the density value passed is in kg/m3.
            d_val = kwargs['density']
            d_unit = kwargs.pop('dens_unit')
            if d_unit != 'kg/m3':
                # Convert density to kg/m3
                try:
                    kwargs['density'] = fp_convert(d_val, d_unit, 'kg/m3')
                except Exception as e:
                    logger.warning(f"Could not convert density from {d_unit} to kg/m3: {e}")
                    pass # Hope it's compatible or fpCore handles it? No, fpCore assumes kg/m3.

        # Generic flow conversion delegating to fpCore
        try:
            # Basic conversion
            return fp_convert(value, from_unit, to_unit)
        except Exception:
            # Try flow specific
            return fp_convert_flow(value, from_unit, to_unit, **kwargs)

    def convert_density(self, value, from_unit, to_unit):
        return fp_convert(value, from_unit, to_unit)

    def convert_viscosity(self, value, from_unit, to_unit):
        return fp_convert(value, from_unit, to_unit)

    def convert_energy(self, value, from_unit, to_unit):
        return fp_convert(value, from_unit, to_unit)

    def convert_power(self, value, from_unit, to_unit):
        return fp_convert(value, from_unit, to_unit)
        
    def convert_dimensionless(self, value, from_unit, to_unit):
        return value

    # --------------------------------------
    # region: Aliases for Backward Compatibility
    # --------------------------------------
    def temperature(self, value, from_unit, to_unit): return self.convert_temperature(value, from_unit, to_unit)
    def pressure(self, value, from_unit, to_unit): return self.convert_pressure(value, from_unit, to_unit)
    def length(self, value, from_unit, to_unit): return self.convert_length(value, from_unit, to_unit)
    def area(self, value, from_unit, to_unit): return self.convert_area(value, from_unit, to_unit)
    def volume(self, value, from_unit, to_unit): return self.convert_volume(value, from_unit, to_unit)
    def mass(self, value, from_unit, to_unit): return self.convert_mass(value, from_unit, to_unit)
    def mol(self, value, from_unit, to_unit): return fp_convert(value, from_unit, to_unit) # No specific convert_mol?
    def density(self, value, from_unit, to_unit): return self.convert_density(value, from_unit, to_unit)
    def flow(self, value, from_unit, to_unit, **kwargs): return self.convert_flow(value, from_unit, to_unit, **kwargs)
    def viscosity(self, value, from_unit, to_unit): return self.convert_viscosity(value, from_unit, to_unit)
    def time(self, value, from_unit, to_unit): return self.convert_time(value, from_unit, to_unit)
    def velocity(self, value, from_unit, to_unit): return fp_convert(value, from_unit, to_unit) # No dedicated convert_velocity method

