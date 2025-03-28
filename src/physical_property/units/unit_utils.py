""" `units.unit_utils`

Utility functions for converting between different units of various physical properties.
"""
from typing import Dict, List

# Full units dictionary for various physical properties
# This dictionary maps each physical property to its corresponding units and conversion factors.
UNITS_DICT: Dict[str, Dict[str, float]] = {
    "temperature": {
        "k": 1, "kelvin": 1, # standard
        "c": 1, "celsius": 1,
        "f": 1, "fahrenheit": 1,
        "r": 1, "rankine": 1,
    },
    "pressure": {
        "bar": 1, "bara": 1, # standard
        "kpa": 0.01, "kilopascal": 0.01,
        "mbar": 0.001, "millibar": 0.001,
        "pa": 1e-5, "pascal": 1e-5,
        "mpa": 10, "megapascal": 10, 
        "gpa": 1.e4, "gigapascal": 1.e4, 
        "atm": 1.01325, "atmosphere": 1.01325,
        "psi": 0.06894757, "psia": 0.06894757,
        "ksi": 68.947572932,
        "torr": 1.01325 / 760,
        "inh2o": 0.0024908891
    },
    "length": {
        "m": 1., "meter": 1., # standard
        "km": 1000., "kilometer": 1000.,
        "cm": 0.01, "centimeter": 0.01,
        "mm": 0.001, "millimeter": 0.001,
        "micron": 1e-6, "micrometer": 1e-6,
        "angstrom": 1e-10,
        "mile": 1/1609,
        "yd": 0.9144, "yard": 0.9144,
        "ft": 0.3048, "feet": 0.3048,
        "in": 0.0254, "inch": 0.0254,
    },
    "mass": {
        "g": 1,         "gram": 1, # standard
        "kg": 1000,     "kilogram": 1000,
        "mg": 1e-3,     "milligram": 1e-3,
        "lb": 453.592,  "lbm": 453.592,
        "oz": 28.3495,  "ounce": 28.3495,
        "slug": 14593.9
    },
    "mol": {
        "mol": 1., "mole": 1.,
        "kmol": 1000., "kilomole": 1000.,
        "mmol": 1e-3, "millimole": 1e-3,
    },
    "time": {
        "s": 1, "sec": 1, # standard
        "year": 86400 * 365., "yr": 86400*365.,
        "month": 86400 * 30.42,
        "week": 86400 * 7,
        "day": 86400, "d": 86400,
        "hr": 3600, "h": 3600,
        "min": 60,
    },
    "volume": {
        "m3": 1.,  "cubic_meter": 1., "m³": 1.,# standard
        "l": 0.001, "liter": 0.001,
        "ml": 1e-6, "milliliter": 1e-6,
        "cm3": 1e-6, "cc": 1e-6, "cm³": 1., 
        "gal": 0.00378541, "gallon": 0.00378541,
        "ft3": 0.02831684661, "cf": 0.02831684661, "ft³": 1.,
        "bbl": 0.15898730, "barrel": 0.15898730,
        "us_bbl": 0.11924636299, "us_barrel": 0.11924636299,
        "fl_oz": 2.95735e-5, "fluid_ounce": 2.95735e-5,
    },
    "viscosity": {
        "pa.s": 1., "pa*s": 1.,  # standard
        "mpa.s": 0.001, "mpa*s": 0.001,
        "cp": 0.001,
        "poise": 10,
        "dynes/cm2": 10, "dynes/cm²": 10,
        "lbf.s/in2": 6894.75, "lbf.s/in²": 6894.75, "lbf*s/in2": 6894.75, "lbf*s/in²": 6894.75,
    },
    "kinematic_viscosity": {
        "st": 1., "stoke": 1.,    # standard
        "cst": 0.01, "centistoke": 0.01, 
        "m2/s": 0.0001, "m²/s": 0.0001,
        "cm2/s": 1., "cm²/s": 1.,
        "mm2/s": 100., "mm²/s": 100.,
        "ft2/h": 3.874467, "ft²/h": 3.874467,
        "ft2/s": 0.00107639, "ft²/s": 0.00107639,
        "in2/s": 0.155, "in²/s": 0.155,
    },
    "velocity": {
        "m/s": 1.,  # standard
        "cm/s": 0.01,
        "mm/s": 0.001,
        "ft/s": 0.3048,
        "in/s": 0.0254,
        "km/h": 1 / 3.6, "kph": 1 / 3.6,
        "mi/h": 0.44704, "mph": 0.44704,
        "knot": 0.514444,
    },
    "composition": {
        "units_mass": {"kg", "g", "mg", "wtf", "wt%"},
        "units_mols": {"kmol", "mol", "mmol", "molf", "mol%"},
    }
}

# Default unit set for various physical properties
DEFAULT_UNIT_SET: Dict[str, str] = {
    "temperature":      "kelvin",
    "pressure":         "bar",
    "length":           "meter",
    "mass":             "kg",
    "mol":              "mol",
    "time":             "s",
    "volume":           "m3",
    "viscosity":        "pa.s",
    "kinematic_viscosity": "st",
    "velocity":         "m/s",
    "composition":      "mol",
}

# Unit type aliases
ALT_KEYS: Dict[str, List[str]] = {
    "temperature":      ["T", "temp"],
    "pressure":         ["P"],
    "length":           ["L", "distance"],
    "mass":             ["m"],
    "mol":              ["mole", "molar"],
    "time":             ["t"],
    "volume":           ["V", "vol"],
    "density":          ["D", "dens", "mass_density", "density_mass"],
    "density_mol":      ["D_mol", "dens_mol", "molar_density"],
    "viscosity":        ["visco", "dynamic_visco", "dynamic_viscosity"],
    "kinematic_viscosity": ["kinematic_visco"],
    "volume_flow":      ["vflow", "volflow", "vol_flow", "volume_flow"],
    "mass_flow":        ["mflow", "massflow"],
    "mol_flow":         ["molar_flow", "molflow", "molarflow"],
    "velocity":         ["velo", "u", "speed"],
    "composition":      ["compo", "amount", "amounts", "n", "ni", "nik"],
}
