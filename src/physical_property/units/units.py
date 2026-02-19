""" `units.units`

Unit conversion module for common physico-chemical properties.
Delegates to `fpcore.units` for actual conversion logic.
"""
import attrs
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

@attrs.define
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

    def convert(self, item: str, value: Any, from_unit: str, to_unit: str) -> Any:
        """
        General function to convert any item from one unit to another.
        """
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

    def convert_flow(self, value, from_unit, to_unit):
        """Convert flow rate."""
        # Generic flow conversion delegating to fpCore
        # fpCore handles complex units like 'm3/d', 'bbl/d'
        try:
            # Basic conversion
            return fp_convert(value, from_unit, to_unit)
        except Exception:
            # Try flow specific
            return fp_convert_flow(value, from_unit, to_unit)

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
