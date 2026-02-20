""" `units.unit_sets`

For the unit conversion functionality in `__str__` methods.
Integration with fpCore for unit definitions.
"""
from typing import Dict, Any, Union
from attrs import asdict

# Integration with fpCore for unit definitions
from fpcore.units import SI, US, US_OIL, EU, EU_OIL

# Configure logging
from ..utils.logging import get_logger
logger = get_logger(__name__)

# Map legacy names to fpCore profiles
_PROFILE_MAP = {
    'si': SI,
    'metric': SI,
    'us': US,
    'us_oil': US_OIL,
    'field': US_OIL, # Mapping 'field' to US Oilfield
    'eu': EU,
    'eu_oil': EU_OIL,
}

# Define legacy DEFAULT dictionary using legacy values
# This is crucial for backward compatibility (e.g. pressure=bar instead of Pa).
# Original DEFAULT_UNIT_SET was in unit_utils.py.
DEFAULT_UNIT_SET = {
    "temperature":      "K",
    "pressure":         "bar",
    "length":           "meter",
    "mass":             "kg",
    "mol":              "mol",
    "time":             "s",
    "volume":           "m^3",
    "viscosity":        "Pa*s",
    "kinematic_viscosity": "St",
    "velocity":         "m/s",
    "composition":      "mol",
}

def get_unit_set(unit_set: Union[str, Dict[str, Any]] = 'standard', flat_dict: bool = False) -> Dict[str, Any]:
    """
    Get the unit conversion mapping for the specified unit set.

    Parameters
    ----------
    unit_set : str or dict
        The unit set to get the unit conversion mapping for.
        Valid 'str' options include: ['standard', 'field', 'US', 'SI', 'metric', 'eu', 'us_oil']
        If 'dict' is provided, it is returned as is.
    flat_dict : bool, optional
        Ignored. Kept for signature compatibility.
        
    Returns
    -------
    dict
        The unit conversion mapping for the specified unit set (using canonical keys).
    """
    if isinstance(unit_set, dict):
        return unit_set

    # 1. Handle 'standard' explicitly (legacy default)
    if unit_set.lower() == 'standard':
        return DEFAULT_UNIT_SET.copy()
        
    # 2. Handle fpCore profiles
    profile = _PROFILE_MAP.get(unit_set.lower())
    if profile:
        return asdict(profile)
        
    # 3. Fallback / Error
    valid_options = ['standard'] + list(_PROFILE_MAP.keys())
    raise ValueError(f"Unsupported unit set: {unit_set}. Valid options are: {valid_options}")

if __name__ == "__main__":
    # Test
    print("Standard:", get_unit_set("standard"))
    print("US:", get_unit_set("us"))
