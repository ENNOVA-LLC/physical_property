""" `base`

This module contains the base class `PhysicalProperty` for physical properties with unit information.
`PhysicalProperty` is inherited by specific property classes (e.g., `Time`, `Temperature`, `Pressure`)
that define the conversion logic for each property type.

The `UnitConverter` class is used to handle unit conversions between different unit sets.
"""
from typing import Tuple, Union, Dict, Any, Optional, ClassVar
import numpy as np
import attr
from plotly import graph_objects as go
from loguru import logger

from .units import UnitConverter  # Import the UnitConverter class

logger.add("logs/property.log", rotation="10 MB")

# Instantiate a UnitConverter object for use across all classes
unit_converter = UnitConverter(unit_set='standard')
UNITS_DICT = unit_converter.UNITS_DICT

    
@attr.s(auto_attribs=True)
class PhysicalProperty:
    """
    Base class for physical properties with unit information.

    Attributes
    ----------
    name : str
        The name of the property. Defaults to the lowercase class name.
    unit : str, optional
        The unit of the property. Defaults to None.
    value : float or np.ndarray
        The property value(s). Defaults to an empty array.
    doc : str, optional
        A description of the property. Defaults to "".
    bounds : Optional[Tuple[Optional[float], Optional[float]]]
        Allowed bounds (lower, upper) expressed in the same unit as the value.
        If not provided and the subclass defines DEFAULT_BOUNDS (in standard units),
        these will be automatically converted to the user-specified unit.
    """
    name: str = attr.ib(default=attr.Factory(lambda self: self.__class__.__name__.lower(), takes_self=True))
    unit: str = attr.ib(default=None)  # Allow None for dimensionless
    doc: str = ""
    _value: np.ndarray = attr.ib(
        default=attr.Factory(lambda: np.array([], dtype=float)),
        converter=lambda v: np.array(v, dtype=float) if v is not None else np.array([], dtype=float)
    )
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = attr.ib(default=None)

    @bounds.validator
    def check_bounds(self, attribute, value):
        """
        Validator for bounds attribute.

        Checks that bounds is either None or a tuple of two numbers (or None) in the form (lower, upper),
        where lower and upper are either None or numbers, and lower is less than or equal to upper.
        """
        if value is not None:
            if not (isinstance(value, tuple) and len(value) == 2):
                raise ValueError("bounds must be a tuple of two numbers (or None) (lower, upper)")
            lower, upper = value
            if lower is not None and not isinstance(lower, (int, float)):
                raise ValueError("Lower bound must be a number or None")
            if upper is not None and not isinstance(upper, (int, float)):
                raise ValueError("Upper bound must be a number or None")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("bounds: lower bound must be less than or equal to upper bound")

    def _convert_default_bound(self, bound_value: float, standard_unit: str) -> float:
        """
        Create a temporary instance (without calling __init__) to convert a default bound
        from the standard unit to the current unit.
        """
        temp = object.__new__(self.__class__)
        # Manually set the necessary attributes.
        temp.name = self.name
        temp._value = np.array([bound_value], dtype=float)
        temp.unit = standard_unit
        temp.doc = self.doc
        temp.bounds = None  # Skip bounds conversion on the temporary instance.
        # Use the instance's convert() method.
        converted = temp.convert(self.unit)
        return float(converted[0])

    def __attrs_post_init__(self):
        """
        Post-initialization logic.

        If no bounds were provided but the subclass defines DEFAULT_BOUNDS (in standard units),
        convert these default bounds to the instance unit.
        """
        # If no bounds are provided and DEFAULT_BOUNDS exists, convert them.
        if self.bounds is None and hasattr(self.__class__, "DEFAULT_BOUNDS"):
            std_bounds = self.__class__.DEFAULT_BOUNDS  # in standard units
            std_unit = unit_converter.DEFAULT_UNIT_SET.get(self._get_property_type(), self.unit)
            if std_unit == self.unit:
                object.__setattr__(self, 'bounds', std_bounds)
            else:
                lower_bound = None
                upper_bound = None
                if std_bounds[0] is not None:
                    lower_bound = self._convert_default_bound(std_bounds[0], std_unit)
                if std_bounds[1] is not None:
                    upper_bound = self._convert_default_bound(std_bounds[1], std_unit)
                object.__setattr__(self, 'bounds', (lower_bound, upper_bound))

        if self.bounds is not None:
            lower, upper = self.bounds
            if lower is not None and np.any(self._value < lower):
                raise ValueError(f"Initial value is below the lower bound: lower bound = {lower}; min value = {np.min(self._value)}")
            if upper is not None and np.any(self._value > upper):
                raise ValueError(f"Initial value is above the upper bound: upper bound = {upper}; max value = {np.max(self._value)}")
        #stack = "".join(traceback.format_stack(limit=10))
        logger.info(f"Created {self.__class__.__name__} instance") #: {self}\nStack trace:\n{stack}")

    @property
    def value(self) -> np.ndarray:
        """
        The value of the physical property.

        Returns
        -------
        np.ndarray
            The value array.
        """
        return self._value

    def empty_value(self) -> np.ndarray:
        """
        Update the value attribute with an empty array.
        """
        object.__setattr__(self, '_value', np.array([], dtype=float)) 

    def _check_and_clip_bounds(self, new_value: np.ndarray, tol: float = 0.005) -> np.ndarray:
        """
        Check new_value against self.bounds. If any element is slightly out-of-bounds (by < 0.5%
        relative to the bound), then clip that element to the bound. Otherwise, raise an error.
        
        Parameters
        ----------
        new_value : np.ndarray
            The array of new values to be checked.
        tol : float, optional
            The tolerance for checking bounds. Defaults to 0.005 (0.5%). The value is clipped if it is within tol of the bound.
            
        Returns
        -------
        np.ndarray
            The possibly clipped new_value array.
        """
        if self.bounds is None:
            return new_value
        lower, upper = self.bounds
        clipped = new_value.copy()
        
        if lower is not None:
            # For values below lower
            diff = lower - clipped  # positive if value is below lower
            mask = diff > 0
            if np.any(mask):
                # For lower == 0, use absolute tolerance (e.g. 1e-8)
                rel_diff = np.where(lower != 0, diff / abs(lower), diff)
                if np.all(rel_diff[mask] < tol):
                    clipped[mask] = lower
                else:
                    raise ValueError(
                        f"Some values ({clipped[mask]}) are below the lower bound {lower} by more than 0.5%."
                    )
        if upper is not None:
            # For values above upper
            diff = clipped - upper  # positive if value is above upper
            mask = diff > 0
            if np.any(mask):
                rel_diff = np.where(upper != 0, diff / abs(upper), diff)
                if np.all(rel_diff[mask] < tol):
                    clipped[mask] = upper
                else:
                    raise ValueError(
                        f"Some values ({clipped[mask]}) are above the upper bound ({upper}) by more than 0.5%."
                    )
        return clipped

    def add_to_value(self, new_value: Union[np.ndarray, float]) -> None:
        """
        Add a new value or array to the current value array.

        Parameters
        ----------
        new_value : np.ndarray or float
            The value(s) to add.
        """
        if isinstance(new_value, PhysicalProperty):
            new_value = new_value.value
        new_value = self.value + new_value
        self.update_value(new_value)

    def update_value(self, new_value: Union[np.ndarray, float]) -> None:
        """
        Update the value attribute with a new array or value.

        Parameters
        ----------
        new_value : np.ndarray or float
            The new value(s).
        """
        if isinstance(new_value, PhysicalProperty):
            new_value = new_value.value
        new_val_array = np.array(new_value, dtype=float)
        new_val_array = self._check_and_clip_bounds(new_val_array)
        if self.bounds is not None:
            lower, upper = self.bounds
            if lower is not None and np.any(new_val_array < lower):
                raise ValueError(f"Updated value is below the lower bound: lower bound = {lower}; min value = {np.min(new_val_array)}.")
            if upper is not None and np.any(new_val_array > upper):
                raise ValueError(f"Updated value is above the upper bound: upper bound = {upper}; max value = {np.max(new_val_array)}.")
        object.__setattr__(self, '_value', new_val_array)
        logger.info(f"Updated {self.__class__.__name__} instance: {self}")

    def append_value(self, new_value: Union[np.ndarray, float], prepend: bool = False) -> None:
        """
        Append a new value or array to the current value array.

        Parameters
        ----------
        new_value : np.ndarray or float
            The value(s) to append.
        prepend : bool, optional
            If True, append the new values to the front of the array. Defaults to False.
        """
        if isinstance(new_value, PhysicalProperty):
            new_value = new_value.value
        new_val_array = np.array(new_value, dtype=float)
        new_val_array = self._check_and_clip_bounds(new_val_array)
        current_value = self._value if self._value.size > 0 else np.array([], dtype=float)
        if prepend:
            updated_value = np.append(new_val_array, current_value)
        else:
            updated_value = np.append(current_value, new_val_array)
        self.update_value(updated_value)
        logger.info(f"Appended {'to the front' if prepend else ''} {self.__class__.__name__} instance: {self}")

    def copy(self) -> 'PhysicalProperty':
        """
        Return a deep copy of the object.
        """
        return self.__class__(name=self.name, value=np.copy(self.value), unit=self.unit, doc=self.doc, bounds=self.bounds)

    def _convert_bounds(self, to_unit: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
        """
        Convert the bounds of the physical property to a new unit.

        Parameters
        ----------
        to_unit : str
            The target unit for conversion.

        Returns
        -------
        Optional[Tuple[Optional[float], Optional[float]]]
            The converted bounds as a tuple (lower, upper) in the new unit.
        """
        if self.bounds is None:
            return None
        lower, upper = self.bounds
        new_lower = None
        new_upper = None
        if lower is not None:
            lower_conv = self.__class__(name=self.name, value=[lower], unit=self.unit, doc=self.doc, bounds=None).convert(to_unit)
            new_lower = float(lower_conv[0])
        if upper is not None:
            upper_conv = self.__class__(name=self.name, value=[upper], unit=self.unit, doc=self.doc, bounds=None).convert(to_unit)
            new_upper = float(upper_conv[0])
        return (new_lower, new_upper)

    def convert(self, to_unit: str) -> np.ndarray:
        """
        Convert the value to a new unit (default: no conversion).

        Parameters
        ----------
        to_unit : str
            The target unit for conversion.

        Returns
        -------
        np.ndarray
            The converted value.

        Raises
        ------
        ValueError
            If conversion is not supported.
        """
        if self.unit == to_unit or (self.unit is None and to_unit is None):
            return self.value
        if self.unit is None or to_unit is None:
            raise ValueError("Cannot convert between dimensionless and dimensional units.")
        raise ValueError(f"Conversion from {self.unit} to {to_unit} not supported in base class.")

    def to(self, to_unit: str) -> 'PhysicalProperty':
        """
        Return a new instance with the value converted to the specified unit.

        Parameters
        ----------
        to_unit : str
            The target unit.

        Returns
        -------
        PhysicalProperty
            A new instance with the converted value and bounds.
        """
        if self.unit == to_unit or (self.unit is None and to_unit is None):
            return self.copy()
        if self.unit is None or to_unit is None:
            raise ValueError("Cannot convert between dimensionless and dimensional units.")
        new_value = self.convert(to_unit)
        new_bounds = self._convert_bounds(to_unit)
        return self.__class__(name=self.name, value=new_value, unit=to_unit, doc=self.doc, bounds=new_bounds)

    def to_standard(self) -> 'PhysicalProperty':
        """
        Convert to the standard unit defined by UnitConverter.DEFAULT_UNIT_SET.

        Returns
        -------
        PhysicalProperty
            A new instance in the standard unit.
        """
        standard_unit = unit_converter.DEFAULT_UNIT_SET.get(self._get_property_type(), self.unit)
        return self.to(standard_unit)

    @classmethod
    def from_unit(cls, name: str, value: Union[float, np.ndarray], unit: str, doc: str = "") -> 'PhysicalProperty':
        """
        Factory method to instantiate the correct PhysicalProperty subclass based on unit.

        Parameters
        ----------
        name : str
            The property name.
        value : float or np.ndarray
            The property value(s).
        unit : str
            The unit of the property.
        doc : str, optional
            A description of the property.

        Returns
        -------
        PhysicalProperty
            An instance of the appropriate subclass.
        """
        unit_key = unit.lower() if unit else None
        prop_type = unit_converter.get_unit_type(unit_key)
        if not prop_type:
            logger.warning(f"Unknown unit '{unit}', defaulting to generic PhysicalProperty.")
            return cls(name=name, value=value, unit=unit, doc=doc)
        if 'flow' in prop_type:
            class_name = 'Flow'
        else:
            class_name = prop_type.capitalize()
        prop_class = globals().get(class_name)
        if not prop_class or not issubclass(prop_class, PhysicalProperty):
            logger.warning(f"Class '{class_name}' not found or not a PhysicalProperty subclass, using PhysicalProperty.")
            return cls(name=name, value=value, unit=unit, doc=doc)
        return prop_class(name=name, value=value, unit=unit, doc=doc)

    # region NUMPY METHODS
    def interpolate(self, new_size: int) -> 'PhysicalProperty':
        """
        Interpolate the value array to a new size.

        Parameters
        ----------
        new_size : int
            The desired number of points.

        Returns
        -------
        PhysicalProperty
            A new instance with the interpolated value.
        """
        original_points = np.linspace(0, 1, len(self.value))
        new_points = np.linspace(0, 1, new_size)
        interpolated_value = np.interp(new_points, original_points, self.value)
        return self.__class__(name=self.name, value=interpolated_value, unit=self.unit, doc=self.doc)

    # basic mathematical operations
    def mean(self) -> float:
        """
        Compute the mean of the value array.

        Returns
        -------
        float
            The mean value.
        """
        return np.mean(self.value)
    
    def min(self) -> float:
        """
        Compute the min of the value array.

        Returns
        -------
        float
            The min value.
        """
        return np.min(self.value)
    
    def max(self) -> float:
        """
        Compute the max of the value array.

        Returns
        -------
        float
            The max value.
        """
        return np.max(self.value)
    
    def std(self) -> float:
        """
        Compute the standard deviation of the value array.

        Returns
        -------
        float
            The standard deviation.
        """
        return np.std(self.value)
    
    def sum(self) -> float:
        """
        Compute the sum of the value array.

        Returns
        -------
        float
            The sum.
        """
        return np.sum(self.value)
    
    def isin(self, value: Union[float, np.ndarray]) -> bool:
        """Check if the value is contained in the `value` attribute.
        
        Parameters
        ----------
        value : float or np.ndarray
            The value to check.
        
        Returns
        -------
        bool
            True if the value is contained in the `value` attribute, False otherwise.
        """
        return np.any(self.value == value) if isinstance(value, np.ndarray) else value in self.value
        
    # array creation methods
    def ones_like(self, constant=1., **kwargs) -> np.ndarray:
        """
        Create a new instance with a value array filled with ones.

        Parameters
        ----------
        constant : float, optional
            The value to fill the array with. Defaults to 1.
        **kwargs : Any, optional
            Additional keyword arguments to pass to np.ones_like().
        
        Returns
        -------
        np.ndarray
            Array filled with constant.
        """
        return np.ones_like(self.value, **kwargs) * constant

    def zeros_like(self, **kwargs) -> np.ndarray:
        """
        Create a new instance with a value array filled with zeros.

        Parameters
        ----------
        **kwargs : Any, optional
            Additional keyword arguments to pass to np.zeros_like().
        
        Returns
        -------
        np.ndarray
            Array filled with zeros.
        """
        return np.zeros_like(self.value, **kwargs)
    
    def empty_like(self, **kwargs) -> np.ndarray:
        """
        Create a new instance with an empty value array.
        
        Parameters
        ----------
        **kwargs : Any, optional
            Additional keyword arguments to pass to np.empty_like().
        
        Returns
        -------
        np.ndarray
            Empty array.
        """        
        return np.empty_like(self.value, **kwargs)
    
    def full_like(self, fill_value: float=0.0, **kwargs) -> np.ndarray:
        """
        Create a new instance with a value array filled with a specific value.
        
        Parameters
        ----------
        fill_value : float
            The value to fill the array with.
        **kwargs : Any, optional
            Additional keyword arguments to pass to np.full_like().
        
        Returns
        -------
        np.ndarray
            Array filled with fill_value.
        """
        return np.full_like(self.value, fill_value, **kwargs)
    # endregion
    
    # region SERIALIZATION
    def to_dict(self, tolist=True) -> Dict[str, Any]:
        """
        Convert the PhysicalProperty instance to a dictionary.

        Parameters
        ----------
        tolist : bool, optional
            Whether to convert the value array to a list.

        Returns
        -------
        dict
            A dictionary representation of the instance.
        """
        return {
            "type": self.__class__.__name__,  # Add subclass name (e.g., "Pressure")
            "name": self.name,
            "value": self.value.tolist() if tolist else self.value,
            "unit": self.unit,
            "doc": self.doc,
            "bounds": self.bounds,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PhysicalProperty':
        """
        Create an instance from a dictionary.

        Parameters
        ----------
        d : dict
            A dictionary containing the property data.

        Returns
        -------
        PhysicalProperty
            A new instance of the property.
        """
        d_copy = d.copy()
        default_name = cls.__name__.lower()
        d_copy.setdefault("name", default_name)
        d_copy["value"] = np.array(d_copy["value"], dtype=float)
        d_copy.setdefault("unit", None)
        d_copy.setdefault("doc", "")
        
        # Check for type and instantiate the correct subclass
        prop_type = d_copy.pop("type", None)
        if prop_type:
            prop_class = globals().get(prop_type, PhysicalProperty)
            if not issubclass(prop_class, PhysicalProperty):
                logger.warning(f"Type '{prop_type}' not a PhysicalProperty subclass, using PhysicalProperty.")
                prop_class = PhysicalProperty
            return prop_class(**d_copy)
        # Fallback to `from_unit` if no type is provided
        return PhysicalProperty.from_unit(
            name=d_copy["name"], value=d_copy["value"], unit=d_copy["unit"], doc=d_copy["doc"]
        )

    def _get_property_type(self) -> str:
        """
        Return the property type based on the subclass name.

        Returns
        -------
        str
            The property type (e.g. "time", "temperature").
        """
        return self.__class__.__name__.lower()
    # endregion
    
    # region PLOT
    def plot(self, x: Any, y: Any = None, title: str = None,
             xaxis_title: str = None, yaxis_title: str = None,
             mode: str = "lines+markers", **kwargs) -> go.Figure:
        """
        Plot data using Plotly.

        Parameters
        ----------
        x : numpy.ndarray or PhysicalProperty
            The x-axis data. If a PhysicalProperty is supplied, its value is used.
        y : None, PhysicalProperty, or list of PhysicalProperty
            The y-axis data. If None, self.value is used.
            If a list is supplied, each PhysicalProperty in the list is plotted as a separate trace.
        title : str, optional
            Title for the plot. Defaults to "y vs x" if not provided.
        xaxis_title : str, optional
            Label for the x-axis. If not provided and x is a PhysicalProperty, defaults to "name (unit)".
        yaxis_title : str, optional
            Label for the y-axis. If not provided and y is a PhysicalProperty,
            defaults to "name (unit)". If y is a list, the names are concatenated.
        mode : str, optional
            Plotly mode for each trace (default: "lines+markers").
        **kwargs
            Additional keyword arguments passed to fig.update_layout.

        Returns
        -------
        go.Figure
            A Plotly figure with the plotted data.
        """
        if isinstance(x, PhysicalProperty):
            x_vals = x.value
            if xaxis_title is None:
                xaxis_title = f"{x.name} ({x.unit})"
        else:
            x_vals = x
            if xaxis_title is None:
                xaxis_title = "x"
        traces = []
        y_vals = self.value
        if yaxis_title is None:
            yaxis_title = f"{self.name} ({self.unit})"
        traces.append(go.Scatter(x=x_vals, y=y_vals, mode=mode, name=self.name))
        if y is None:
            pass
        elif isinstance(y, list):
            if yaxis_title is None:
                yaxis_title = ", ".join([f"{prop.name} ({prop.unit})" for prop in y])
            for prop in y:
                if isinstance(prop, PhysicalProperty):
                    traces.append(go.Scatter(x=x_vals, y=prop.value, mode=mode, name=f"{prop.name} ({prop.unit})"))
                else:
                    raise ValueError("All elements in y must be instances of PhysicalProperty")
        elif isinstance(y, PhysicalProperty):
            y_vals = y.value
            if yaxis_title is None:
                yaxis_title = f"{y.name} ({y.unit})"
            traces.append(go.Scatter(x=x_vals, y=y_vals, mode=mode, name=f"{y.name}"))
        else:
            raise ValueError("y must be None, a PhysicalProperty, or a list of PhysicalProperty instances")
        if title is None:
            title = f"{yaxis_title} vs {xaxis_title}"
        fig = go.Figure(data=traces)
        fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                          showlegend=True, **kwargs)
        return fig
    # endregion
    
    # region MAGIC METHODS
    def __str__(self):
        #v_str = f"{self.value[:5]}..." if self.value.size > 5 else f"{self.value}"
        v_str = self.value.shape
        return f"{self.__class__.__name__}(name='{self.name}', unit='{self.unit}', value={v_str})"
    
    def __repr__(self):
        return str(self)
    
    def __len__(self):
        return len(self.value)
    
    def __getitem__(self, key) -> Union['PhysicalProperty', np.ndarray]:
        """
        Return a new instance with a sliced value array, or the value only if key is a tuple with 'val' or 'value'.
        
        Examples
        --------
        >>> prop = PhysicalProperty(value=np.array([[1, 2], [3, 4]]), unit="kg")
        >>> prop[0, 1]  # Returns PhysicalProperty(value=2, unit="kg")
        >>> prop[0, 1, 'val']  # Returns 2 (scalar)
        """
        if isinstance(key, tuple) and ('val' in key or 'value' in key):
            # Extract the indexing part, excluding 'val'
            idx = tuple(k for k in key if k != 'val' and k != 'value')
            return self.value[idx] if idx else self.value
        return self.__class__(name=self.name, value=self.value[key], unit=self.unit, doc=self.doc)
    
    def __format__(self, format_spec: str) -> str:
        """
        Format the PhysicalProperty instance using the provided format specification.

        Parameters
        ----------
        format_spec : str
            The format specification (e.g., '.2e', '.3f'). If empty, falls back to str(self).

        Returns
        -------
        str
            The formatted string representation of the value if scalar, or full object if not.

        Raises
        ------
        ValueError
            If the value array does not contain exactly one element when a numeric format is requested.
        """
        if not format_spec:  # No format spec, use default string representation
            return str(self)
        if self.value.size != 1:
            raise ValueError(
                f"Cannot apply format '{format_spec}' to {self.__class__.__name__} with {self.value.size} elements; "
                "numeric formatting requires a scalar value (size=1)."
            )
        # Handle both 0D (scalar) and 1D (single-element array) cases
        scalar_value = float(self.value.item() if self.value.ndim == 0 else self.value[0])
        return format(scalar_value, format_spec)
    
    # region ARITHMETIC OPERATORS
    # division
    # def __truediv__(self, other) -> 'PhysicalProperty':
    #     if isinstance(other, PhysicalProperty):
    #         result_value = self.value / other.value
    #         # Simple unit inference for common cases
    #         if self.unit == "m" and other.unit == "m/s":
    #             return Time(name="residence time", value=result_value, unit="s", doc=self.doc)
    #         return PhysicalProperty(name=f"{self.name}/{other.name}", value=result_value, unit=None, doc=self.doc)
    #     elif isinstance(other, (int, float, np.ndarray)):
    #         result_value = self.value / other
    #         return self.__class__(name=self.name, value=result_value, unit=self.unit, doc=self.doc)
    #     else:
    #         raise TypeError(f"Division not supported between {type(self)} and {type(other)}")

    # def __rtruediv__(self, other) -> 'PhysicalProperty':
    #     if isinstance(other, (int, float, np.ndarray)):
    #         result_value = other / self.value
    #         if isinstance(other, PhysicalProperty) and other.unit == "m" and self.unit == "m/s":
    #             return Time(name="residence time", value=result_value, unit="s", doc=self.doc)
    #         return PhysicalProperty(name=f"scalar/{self.name}", value=result_value, unit=None, doc=self.doc)
    #     else:
    #         raise TypeError(f"Right division not supported between {type(other)} and {type(self)}")

    # Addition
    def __add__(self, other) -> 'PhysicalProperty':
        """Addition of this PhysicalProperty with another value or PhysicalProperty."""
        if isinstance(other, PhysicalProperty):
            if self.unit != other.unit and self.unit is not None and other.unit is not None:
                raise ValueError(f"Cannot add {self.unit} and {other.unit} without unit conversion.")
            result_value = self.value + other.value
            return self.__class__(name=self.name, value=result_value, unit=self.unit, doc=self.doc)
        elif isinstance(other, (int, float, np.ndarray)):
            result_value = self.value + other
            return self.__class__(name=self.name, value=result_value, unit=self.unit, doc=self.doc)
        else:
            raise TypeError(f"Addition not supported between {type(self)} and {type(other)}")

    def __radd__(self, other) -> 'PhysicalProperty':
        """Right addition (other + self)."""
        return self.__add__(other)  # Addition is commutative

    # Subtraction
    def __sub__(self, other) -> 'PhysicalProperty':
        """Subtraction of this PhysicalProperty by another value or PhysicalProperty."""
        if isinstance(other, PhysicalProperty):
            if self.unit != other.unit and self.unit is not None and other.unit is not None:
                raise ValueError(f"Cannot subtract {other.unit} from {self.unit} without unit conversion.")
            result_value = self.value - other.value
            return self.__class__(name=self.name, value=result_value, unit=self.unit, doc=self.doc)
        elif isinstance(other, (int, float, np.ndarray)):
            result_value = self.value - other
            return self.__class__(name=self.name, value=result_value, unit=self.unit, doc=self.doc)
        else:
            raise TypeError(f"Subtraction not supported between {type(self)} and {type(other)}")

    def __rsub__(self, other) -> 'PhysicalProperty':
        """Right subtraction (other - self)."""
        if isinstance(other, (int, float, np.ndarray)):
            result_value = other - self.value
            return self.__class__(name=self.name, value=result_value, unit=self.unit, doc=self.doc)
        else:
            raise TypeError(f"Right subtraction not supported between {type(other)} and {type(self)}")

    # Multiplication
    # def __mul__(self, other) -> 'PhysicalProperty':
    #     """Multiplication of this PhysicalProperty with another value or PhysicalProperty."""
    #     if isinstance(other, PhysicalProperty):
    #         result_value = self.value * other.value
    #         # Simple unit inference for common cases
    #         if self.unit == "m/s" and other.unit == "s":
    #             return Length(name="distance", value=result_value, unit="m", doc=self.doc)
    #         return PhysicalProperty(name=f"{self.name}*{other.name}", value=result_value, unit=None, doc=self.doc)
    #     elif isinstance(other, (int, float, np.ndarray)):
    #         result_value = self.value * other
    #         return self.__class__(name=self.name, value=result_value, unit=self.unit, doc=self.doc)
    #     else:
    #         raise TypeError(f"Multiplication not supported between {type(self)} and {type(other)}")

    # def __rmul__(self, other) -> 'PhysicalProperty':
    #     """Right multiplication (other * self)."""
    #     return self.__mul__(other)  # Multiplication is commutative

    # endregion
    