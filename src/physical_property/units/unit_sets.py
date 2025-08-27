""" `units.unit_sets`

For the unit conversion functionality in `__str__` methods.
"""
#from ..utils.units import convert_x

# Configure logging
from ..utils.logging import get_logger
logger = get_logger(__name__)

unit_sets = {
    'standard': {
        'T'     : 'K',
        'P'     : 'bar',
        'V'     : 'm3',
        't'     : 's',
        'L'     : 'm',
        'R'     : 'm',
        'angle' : 'rad',
        'm'     : 'kg',
        'mol'   : 'mol',
        'mflow' : {'V': 'kg/s', 'L': 'kg/s'},
        'vflow' : {'V': 'm3/s', 'L': 'm3/s'},
        'dens'  : 'kg/m3',
        'visco' : 'Pa.s',
        "kinematic_visco": "st",
        'velo'  : 'm/s',
    },
    'field': {
        'T'     : 'F',
        'P'     : 'psia',
        'V'     : 'ft3',
        't'     : 'day',
        'L'     : 'ft',
        'R'     : 'in',
        'angle' : 'rad',
        'm'     : 'kg',
        'mol'   : 'mol',
        'mflow' : {'V': 'kg/s', 'L': 'kg/s'},
        'vflow' : {'V': 'Mcf/d', 'L': 'bbl/d'},
        'dens'  : 'g/cm3',
        'visco' : 'cP',
        "kinematic_visco": "st",
        'velo'  : 'ft/s',
    },
    'us': {
        'T'     : 'F',
        'P'     : 'psia',
        'V'     : 'ft3',
        't'     : 'day',
        'L'     : 'ft',
        'R'     : 'in',
        'angle' : 'rad',
        'm'     : 'lb',
        'mol'   : 'mol',
        'mflow' : {'V': 'lb/s', 'L': 'lb/s'},
        'vflow' : {'V': 'Mcf/d', 'L': 'bbl/d'},
        'dens'  : 'lb/ft3',
        'visco' : 'cP',
        "kinematic_visco": "st",
        'velo'  : 'ft/s',
    },
    'si': {
        'T'     : 'K',
        'P'     : 'Pa',
        'V'     : 'm3',
        't'     : 's',
        'L'     : 'm',
        'R'     : 'm',
        'angle' : 'rad',
        'm'     : 'kg',
        'mol'   : 'mol',
        'mflow' : {'V': 'kg/s', 'L': 'kg/s'},
        'vflow' : {'V': 'm3/s', 'L': 'm3/s'},
        'dens'  : 'kg/m3',
        'visco' : 'Pa.s',
        "kinematic_visco": "st",
        'velo'  : 'm/s',
    },
    'metric': {
        'T'     : 'C',
        'P'     : 'bar',
        'V'     : 'L',
        't'     : 'day',
        'L'     : 'm',
        'R'     : 'cm',
        'angle' : 'rad',
        'm'     : 'kg',
        'mol'   : 'mol',
        'mflow' : {'V': 'kg/s', 'L': 'kg/s'},
        'vflow' : {'V': 'L/d', 'L': 'L/d'},
        'dens'  : 'g/mL',
        'visco' : 'cP',
        "kinematic_visco": "st",
        'velo'  : 'm/s',
    }
}
unit_sets['default'] = unit_sets.get('standard')

def get_unit_set(unit_set='standard', flat_dict=False):
    """
    Get the unit conversion mapping for the specified unit set.

    Parameters
    ----------
    unit_set : str
        The unit set to get the unit conversion mapping for.
        Valid options are: ['standard', 'field', 'US', 'SI', 'metric']
    flat_dict : bool, optional
        If True, extract the key inside nested dictionaries.
        
    Returns
    -------
    dict
        The unit conversion mapping for the specified unit set.
    """
    unit_set = unit_set.lower()
    if unit_set not in unit_sets:
        raise ValueError(f"Unsupported unit set. Valid options are: {list(unit_sets.keys())}")

    units = unit_sets[unit_set]
    if not flat_dict:
        return units

    flat_units = {}
    for key, value in units.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flat_units[key] = nested_value
        else:
            flat_units[key] = value
    return flat_units


# For debugging purposes
if __name__ == "__main__":
    """
    Print out the unit conversion mapping for each unit set.
    """

    # Loop over unit sets and print out the unit conversion mapping for each
    for unit_set in unit_sets:
        print(f"{unit_set} units:")
        for key, unit in unit_sets[unit_set].items():
            print(f"  {key}: {unit}")

