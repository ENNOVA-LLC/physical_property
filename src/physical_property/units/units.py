""" `units.units`

Unit conversion module for common physico-chemical properties.

This package contains unit conversion functions for various
physico-chemical properties such as temperature, pressure, mass, etc.
"""
import attrs
from typing import Dict, Tuple, List, Any, Union, ClassVar
import numpy as np
from loguru import logger

from .unit_sets import get_unit_set

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
    UNITS_DICT: ClassVar[Dict[str, Dict[str, float]]] = {
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
    
    DEFAULT_UNIT_SET: ClassVar[Dict[str, str]] = {
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
    
    ALT_KEYS: ClassVar[Dict[str, List[str]]] = {
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
        "volume_flow":      ["vflow", "volflow", "vol_flow"],
        "mass_flow":        ["mflow", "massflow"],
        "mol_flow":         ["molar_flow", "molflow"],
        "velocity":         ["velo", "u", "speed"],
        "composition":      ["compo", "amount", "amounts", "n", "ni", "nik"],
    }
    
    def __attrs_post_init__(self):
        logger.info(f"UnitConverter initialized with unit_set: {self.unit_set}")
        self.unit_set = self.get_unit_set(self.unit_set)
        
    def get_unit_set(self, unit_set) -> Dict[str, str]:
        """
        Retrieves the unit set based on the `unit_set` input.

        Parameters
        ----------
        unit_set : str|dict
            If `str`, the options are: ['standard', 'field', 'US', 'metric', 'SI']
            If `dict`, provide a dictionary containing the unit settings.

        Returns
        -------
        Dict[str, str]
            The unit conversion mapping for the specified unit set.
        """
        if unit_set is None:
            unit_set = self.DEFAULT_UNIT_SET
        else:
            if isinstance(unit_set, str):
                unit_set = get_unit_set(unit_set, flat_dict=True)

        if not isinstance(unit_set, dict):
            raise ValueError("Invalid input: `unit_set`.")
        
        # Translate keys using ALT_KEYS
        unit_set_standard = {}
        for key, value in unit_set.items():
            for std_key, alt_keys in self.ALT_KEYS.items():
                if key in alt_keys or key == std_key:
                    unit_set_standard[std_key] = value
                    break
            else:
                unit_set_standard[key] = value
        
        return unit_set_standard
    
    @staticmethod
    def _unit_error_check(unit: str, unit_type: str) -> None:
        """Check if user provided unsupported unit."""
        if "/" in unit:
            units = unit.split("/")
            for u in units:
                if u not in UnitConverter.UNITS_DICT.get(unit_type, {}):
                    raise ValueError(f"Unsupported unit: {u} for unit type: {unit_type}")
        else:
            if unit not in UnitConverter.UNITS_DICT.get(unit_type, {}):
                raise ValueError(f"Unsupported unit: {unit} for unit type: {unit_type}")


    @staticmethod
    def _handle_unit_prefix(unit: str, unit_type: str) -> Tuple[str, float]:
        """Handle unit prefix like 'MM' and 'M' for volume units."""
        if unit_type in ["volume", "flow"]:
            if unit.startswith("MM"):
                return unit[2:], 1e6
            elif unit.startswith("M") and unit not in {"m3"}:
                return unit[1:], 1e3
        return unit, 1

    @staticmethod
    def _preproc_inputs(val, unit_in:str, unit_out:str, unit_type:str) -> Tuple[np.ndarray, str, str]:
        """ `unit` string pre-processor."""
        def standard_string(s:str) -> str:
            """Removes the following elements and returns lowercase string.
            elements: ['°', 'S', '^']
            """
            if s.startswith("S"):
                s = s[1:]  # Remove the leading "S" if it's uppercase (denotes "standard" volume)
            return s.replace("°", "").replace("^", "").lower()
        
        # Handle prefixes and standardize units
        unit_in, prefix_in = UnitConverter._handle_unit_prefix(unit_in.lower(), unit_type)
        unit_out, prefix_out = UnitConverter._handle_unit_prefix(unit_out.lower(), unit_type)

        # Apply prefix multipliers
        val = np.array(val, dtype=np.float64) * prefix_in / prefix_out
        
        # standardize inputs
        unit_in = standard_string(unit_in)      # unit strings
        unit_out = standard_string(unit_out)
        
        # check if unit strings are supported
        UnitConverter._unit_error_check(unit_in, unit_type)
        UnitConverter._unit_error_check(unit_out, unit_type)
        return val, unit_in, unit_out

    @staticmethod
    def _master_convert(val, unit_in:str, unit_out:str, unit_type:str) -> np.ndarray:
        """Master converter for units that follow a simple multiplicative conversion."""
        val, unit_in, unit_out = UnitConverter._preproc_inputs(val, unit_in, unit_out, unit_type)
        return val * UnitConverter.UNITS_DICT[unit_type][unit_in] / UnitConverter.UNITS_DICT[unit_type][unit_out]

    def get_unit_type(self, unit: str) -> Tuple[str, List]:
        """Determine the unit type for a given unit string, including composite units."""
        unit = unit.replace(" ", "").replace("%", "")  # Remove some unwanted characters for consistency
        if "/" in unit:
            unit_parts = unit.split("/")
            if len(unit_parts) == 2:
                unit_type_1 = self.get_unit_type(unit_parts[0])
                unit_type_2 = self.get_unit_type(unit_parts[1])
                if unit_type_1 == "volume" and unit_type_2 == "time":
                    return "volume_flow"
                elif unit_type_1 == "mass" and unit_type_2 == "time":
                    return "mass_flow"
                elif unit_type_1 == "length" and unit_type_2 == "time":
                    return "velocity"
                elif unit_type_1 == "mass" and unit_type_2 == "volume":
                    return "density"
                elif unit_type_1 == "mol" and unit_type_2 == "volume":
                    return 'density_mol'
            raise ValueError(f"Unsupported composite unit: {unit}")
        else:
            for key, units in self.UNITS_DICT.items():
                if unit.lower() in units:
                    return key
            raise ValueError(f"Unsupported unit: {unit}")
        
    # conversion methods
    def convert(self, data: Dict[str, Any], unit_set=None) -> Dict[str, Any]:
        """
        Converts `data` (containing keys {'unit', 'value'}) to the specification in `unit_set`.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Contains keys {'unit', 'value'} specifying the unit and value to be converted.
        unit_set : str|dict, optional
            Defaults to instance attribute value(s).
            
        Returns
        -------
        dict
            Contains the converted unit and value.
        
        Notes
        -----
        `data` may be of either formats below:
        ```python
        data = {'unit': 'C', 'value': 25}
        data = {'T': {'unit': 'C', 'value': 25}, 'P': {'unit': 'atm', 'value': 1}}
        ```
        """
        def convert_single_entry(details: dict, unit_set: dict) -> dict:
            unit_in = details['unit']
            val = details['value']
            
            # Extract unit_type (if provided) or set it based on `unit_in`
            unit_type = details.get('unit_type', details.get('property'))
            unit_type = unit_type or self.get_unit_type(unit_in)
            
            # Set `unit_out`
            if unit_type == "composite":
                unit_in_parts = unit_in.split("/")
                unit_out = "/".join([unit_set.get(self.get_unit_type(part), part) for part in unit_in_parts])
            else:
                unit_out = unit_set.get(unit_type, unit_in)

            # Convert
            val_out = self.convert_x(unit_type, val, unit_in, unit_out)
            return {'unit': unit_out, 'value': val_out}

        # Retrieve unit_set
        if unit_set:
            unit_set = self.get_unit_set(unit_set)
        else:
            unit_set = self.unit_set

        # Handle single-entry format
        if 'unit' in data and 'value' in data:
            return convert_single_entry(data, unit_set)
        
        # Handle multi-entry format
        return {key: convert_single_entry(val, unit_set) for key, val in data.items()}
    
    def convert_x(self, prop: str, val, unit_in: str, unit_out: str, return_unit=False, **kwargs) -> float | np.ndarray | Tuple:
        """Generic unit converter.
        
        Parameters
        ----------
        prop : str
            Property to convert.
        val : float
            Value to convert.
        unit_in : str
            Unit to convert from.
        unit_out : str
            Unit to convert to.
        return_unit : bool, optional
            Whether to return the unit in addition to the value. Defaults to False.
        """
        if prop == "composite":
            unit_in_parts = unit_in.split("/")
            unit_out_parts = unit_out.split("/")
            converted_val_parts = []
            for i, unit_part in enumerate(unit_in_parts):
                unit_type = self.get_unit_type(unit_part)
                converted_val_part = self.convert_x(unit_type, val, unit_part, unit_out_parts[i], **kwargs)
                converted_val_parts.append(converted_val_part)
            converted_val = np.prod(converted_val_parts)
        else:
            dict_prop = {}
            for std_key, alt_keys in self.ALT_KEYS.items():
                for alt_key in alt_keys:
                    dict_prop[alt_key] = getattr(self, std_key, None)
                dict_prop[std_key] = getattr(self, std_key, None)

            func_convert = dict_prop.get(prop)
            if func_convert is None:
                raise ValueError(f"Unsupported property: `{prop}`.")

            converted_val = func_convert(val, unit_in, unit_out, **kwargs)
            converted_val = float(converted_val) if converted_val.shape == () else converted_val

        if return_unit:
            return converted_val, unit_out
        else:
            return converted_val

    # region Conversion Methods
    def temperature(self, val, unit_in:str, unit_out:str) -> np.ndarray:
        """Temperature unit converter."""
        unit_in = unit_in.replace("°", "").lower()
        unit_out = unit_out.replace("°", "").lower()
        def T_to_K(T:float, unit:str):
            if unit in {"c", "°c", "celsius"}:
                return T + 273.15
            elif unit in {"k", "kelvin"}:
                return T * 1.
            elif unit in {"f", "°f", "fahrenheit"}:
                return (T + 459.67) * (5/9)
            elif unit in {"r", "°r", "rankine"}:
                return T * (5/9)

        def K_to_T(T:float, unit:str):
            if unit in {"c", "°c"}:
                return T - 273.15
            elif unit == "k":
                return T * 1.
            elif unit in {"f", "°f"}:
                return (T - 273.15) / (5/9) + 32
            elif unit in {"r", "°r"}:
                return T / (5/9)

        val, unit_in, unit_out = self._preproc_inputs(val, unit_in, unit_out, "temperature")
        return K_to_T(T_to_K(val, unit_in), unit_out)

    def pressure(self, val, unit_in:str, unit_out:str) -> np.ndarray:
        """Pressure unit converter."""
        gauge_in = unit_in[-1] == "g"
        gauge_out = unit_out[-1] == "g"

        if gauge_in:
            unit_in = unit_in[:-1]
        if gauge_out:
            unit_out = unit_out[:-1]

        if gauge_in:
            val += self.UNITS_DICT['pressure']['atm'] / self.UNITS_DICT['pressure'][unit_in]
            
        P_out = self._master_convert(val, unit_in, unit_out, 'pressure')
        
        if gauge_out:
            P_out -= self.UNITS_DICT['pressure']['atm'] / self.UNITS_DICT['pressure'][unit_out]

        return P_out

    def length(self, val, unit_in:str, unit_out:str):
        """Length unit converter."""
        return self._master_convert(val, unit_in, unit_out, 'length')

    def mass(self, val, unit_in: str, unit_out: str) -> float:
        """Mass unit converter."""
        return self._master_convert(val, unit_in, unit_out, 'mass')

    def mol(self, val, unit_in: str, unit_out: str) -> float:
        """Molar unit converter."""
        return self._master_convert(val, unit_in, unit_out, 'mol')

    def amount(self, val, unit_in: str, unit_out: str, MW: float = None) -> float | np.ndarray:
        """
        Converts between mass and mole units. If converting between mass and moles,
        MW (molecular weight in g/mol) must be provided.

        Parameters
        ----------
        val : float or np.ndarray
            The numerical value(s) to convert.
        unit_in : str
            The input unit (e.g., 'kg', 'mol').
        unit_out : str
            The desired output unit (e.g., 'g', 'mol').
        MW : float, optional
            Molecular weight in g/mol. Required for cross conversion between mass and moles.

        Returns
        -------
        float or np.ndarray
            The converted value(s).

        Raises
        ------
        ValueError
            If MW is required but not provided, or if the units are unsupported.
        """
        u_in = unit_in.lower()
        u_out = unit_out.lower()
        mass_units = set(self.UNITS_DICT["mass"].keys())
        mol_units = set(self.UNITS_DICT["mol"].keys())

        # Same type conversion: mass to mass or mole to mole
        if u_in in mass_units and u_out in mass_units:
            return self.mass(val, u_in, u_out)
        elif u_in in mol_units and u_out in mol_units:
            return self.mol(val, u_in, u_out)
        
        # Cross conversion: mass <-> mole requires MW
        if MW is None:
            raise ValueError("MW (molecular weight in g/mol) is required to convert between mass and moles.")

        if u_in in mass_units and u_out in mol_units:
            # Convert mass (input) to standard mass (grams)
            mass_in_grams = self.mass(val, u_in, "g")
            # Convert grams to moles using MW
            moles = mass_in_grams / MW
            # Convert standard mole ('mol') to desired mole unit
            return self.mol(moles, "mol", u_out)
        elif u_in in mol_units and u_out in mass_units:
            # Convert moles (input) to standard mole ('mol')
            moles_standard = self.mol(val, u_in, "mol")
            # Convert moles to mass in grams using MW
            mass_in_grams = moles_standard * MW
            # Convert grams to desired mass unit
            return self.mass(mass_in_grams, "g", u_out)
        else:
            raise ValueError(f"Unsupported conversion from {unit_in} to {unit_out}.")
                                    
    def time(self, val, unit_in: str, unit_out: str) -> float:
        """Time unit converter."""
        return self._master_convert(val, unit_in, unit_out, 'time')

    def volume(self, val, unit_in:str, unit_out:str):
        """Volume unit converter."""
        return self._master_convert(val, unit_in, unit_out, 'volume')

    def density(self, val, unit_in:str, unit_out:str):
        """Density (mass/vol) unit converter."""
        unit_in_m, unit_in_V = unit_in.split("/")
        unit_out_m, unit_out_V = unit_out.split("/")
        return self.mass(self.volume(val, unit_out_V, unit_in_V), unit_in_m, unit_out_m) 

    def density_mol(self, val, unit_in:str, unit_out:str):
        """Density (mol/vol) unit converter."""
        unit_in_m, unit_in_V = unit_in.split("/")
        unit_out_m, unit_out_V = unit_out.split("/")
        return self.mol(self.volume(val, unit_out_V, unit_in_V), unit_in_m, unit_out_m)

    def viscosity(self, val, unit_in:str, unit_out:str):
        """Viscosity (dynamic) unit converter."""
        return self._master_convert(val, unit_in, unit_out, 'viscosity')

    def kinematic_viscosity(self, val, unit_in:str, unit_out:str):
        """Viscosity (kinematic) unit converter."""
        return self._master_convert(val, unit_in, unit_out, 'kinematic_viscosity')

    def velocity(self, val, unit_in:str, unit_out:str) -> np.ndarray:
        """Velocity (length/time) unit converter."""
        unit_in_L, unit_in_t = unit_in.split("/")
        unit_out_L, unit_out_t = unit_out.split("/")
        return self.length(self.time(val, unit_out_t, unit_in_t), unit_in_L, unit_out_L) 

    def volume_flow(self, val, unit_in:str, unit_out:str):
        """Flow rate (volume) unit converter."""
        unit_in_V, unit_in_t = unit_in.split("/")
        unit_out_V, unit_out_t = unit_out.split("/")
        return self.volume(self.time(val, unit_out_t, unit_in_t), unit_in_V, unit_out_V)

    def mass_flow(self, val, unit_in:str, unit_out:str):
        """Flow rate (mass) unit converter."""
        unit_in_m, unit_in_t = unit_in.split("/")
        unit_out_m, unit_out_t = unit_out.split("/")
        return self.mass(self.time(val, unit_out_t, unit_in_t), unit_in_m, unit_out_m)

    def mol_flow(self, val, unit_in:str, unit_out:str):
        """Flow rate (mol) unit converter."""
        unit_in_m, unit_in_t = unit_in.split("/")
        unit_out_m, unit_out_t = unit_out.split("/")
        return self.mol(self.time(val, unit_out_t, unit_in_t), unit_in_m, unit_out_m)

    def flow(self, val, unit_in:str, unit_out:str, dens=None, dens_unit:str="kg/m3"):
        """Flow rate unit converter."""
        def standard_string(unit):
            if unit.startswith("S"):
                unit = unit[1:]  # Remove the leading "S" if it's uppercase
            return unit.lower().split("/")
                
        def determine_unit_type(unit: str):
            if unit in V_units:
                return 'volume'
            elif unit in m_units:
                return 'mass'
            elif unit in mol_units:
                return 'mol'
            else:
                raise ValueError(f"Unsupported unit type: {unit}")
        
        unit_in, prefix_in = self._handle_unit_prefix(unit_in, "flow")
        unit_out, prefix_out = self._handle_unit_prefix(unit_out, "flow")

        val = np.array(val, dtype=np.float64) * prefix_in / prefix_out
        
        V_units = set(self.UNITS_DICT['volume'].keys())
        m_units = set(self.UNITS_DICT['mass'].keys())
        mol_units = set(self.UNITS_DICT['mol'].keys())
            
        prop_in, time_in = standard_string(unit_in)
        prop_out, time_out = standard_string(unit_out)

        type_in = determine_unit_type(prop_in)
        type_out = determine_unit_type(prop_out)
        
        if type_in == type_out:
            if type_in == 'volume':
                return self.volume_flow(val, unit_in, unit_out)
            elif type_in == 'mass':
                return self.mass_flow(val, unit_in, unit_out)
            else:
                return self.mol_flow(val, unit_in, unit_out)

        if dens is None or dens_unit is None:
            raise ValueError("`dens` and `dens_unit` are required for the given conversion.")
        
        if type_in == 'volume' and type_out == 'mass':
            density = self.density(dens, dens_unit, "kg/m3")
            return self.mass_flow(self.volume_flow(val, unit_in, "m3/s") * density, "kg/s", unit_out)

        elif type_in == 'volume' and type_out == 'mol':
            density = self.density_mol(dens, dens_unit, "mol/m3")
            return self.mol_flow(self.volume_flow(val, unit_in, "m3/s") * density, "mol/s", unit_out)
        
        elif type_in == 'mass' and type_out == 'volume':
            density = self.density(dens, dens_unit, "kg/m3")
            return self.volume_flow(self.mass_flow(val, unit_in, "kg/s") / density, "m3/s", unit_out)

        elif type_in == 'mol' and type_out == 'volume':
            density = self.density_mol(dens, dens_unit, "mol/m3")
            return self.volume_flow(self.mol_flow(val, unit_in, "mol/s") / density, "m3/s", unit_out)

        else:
            raise ValueError("Unsupported conversion type.")

    def composition(self, compo, unit_in:str, unit_out:str, MW=None):
        """Composition unit converter."""
        def mass_to_moles(mass, MW):
            return mass / MW

        def moles_to_mass(mols, MW):
            return mols * MW

        def normalize(vals, basis:float=1):
            return vals / np.sum(vals) * basis

        ni_dict = dict(kg=1000, g=1, mg=1e-3, kmol=1000, mol=1, mmol=1e-3,)
        
        unit_in = unit_in.lower()
        unit_out = unit_out.lower()
        
        units_mass = self.UNITS_DICT["compo"]["units_mass"]
        units_mols = self.UNITS_DICT["compo"]["units_mols"]
        unit_in_mols = unit_in in units_mols
        unit_out_mols = unit_out in units_mols
        
        if unit_in in {"wtf", "wt%"}:
            unit_in = "g"
        elif unit_in in {"molf", "mol%"}:
            unit_in = "mol"
        
        compo = np.array(compo, dtype=np.float64)
        if unit_in in ni_dict:
            compo = compo * ni_dict[unit_in]
            
        if unit_in_mols != unit_out_mols:
            if MW is None:
                raise ValueError("`MW` is required to convert between mass and moles.")
            
            MW = np.array(MW, dtype=np.float64)
            if unit_in in units_mass and unit_out in units_mols:
                compo = mass_to_moles(compo, MW)
            elif unit_in in units_mols and unit_out in units_mass:
                compo = moles_to_mass(compo, MW)
        
        if unit_out in ni_dict:
            compo = compo / ni_dict[unit_out]
            
        if "f" in unit_out:
            compo = normalize(compo, 1)
        elif "%" in unit_out:
            compo = normalize(compo, 100)

        return compo

if __name__ == "__main__":
    logger.info("The `units` module is a utility to be imported.")
