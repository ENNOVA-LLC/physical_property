""" `properties.specialty.composition`

Contains classes and functions for working with chemical compositions.

Classes
-------
Species: 
    Represents a single chemical species in a composition.
Composition: 
    Represents a mixture of chemical species.
Fluids: 
    A collection of fluid compositions.
SolventBlendSpec: 
    A specification for solvent blending, producing new compositions.
"""
import attr
import numpy as np
from typing import List, Dict, Union

from physical_property.units import UnitConverter

# Configure logging
from ...utils.logging import get_logger
logger = get_logger(__name__)

# Default UnitConverter instance
DEFAULT_CONVERTER = UnitConverter(unit_set="standard")

@attr.s(auto_attribs=True)
class Species:
    """Represents a single chemical species in a composition.

    Attributes
    ----------
    name : str
        Name of the species (e.g., "H2O", "CO2").
    amount : float
        Amount of the species (must be non-negative).
    unit : str
        Unit of the amount (e.g., "kg", "mol").
    molar_mass : float
        Molar mass of the species (in g/mol).

    Examples
    --------
    >>> water = Species(name="H2O", amount=10, unit="kg", molar_mass=18.015)
    >>> print(water)
    Species(name='H2O', amount=10.0, unit='kg', molar_mass=18.015)
    """
    name: str
    amount: float = attr.ib(converter=float)
    unit: str
    molar_mass: float = attr.ib(converter=float)  # g/mol
    _converter: UnitConverter = attr.ib(default=DEFAULT_CONVERTER, repr=False)

    @amount.validator
    def _check_amount(self, attribute, value):
        """Ensure amount is non-negative."""
        if value < 0:
            raise ValueError("Species amount must be non-negative.")

    def to(self, to_unit: str) -> "Species":
        """Convert amount to a new unit, handling mass-to-mole conversions using the UnitConverter."""
        if self.unit == to_unit:
            return Species(name=self.name, amount=self.amount, unit=self.unit, molar_mass=self.molar_mass)
        
        from_type = self._converter.get_unit_type(self.unit.lower())
        to_type = self._converter.get_unit_type(to_unit.lower())
        
        if from_type == to_type:
            converted_amount = self._converter.convert(from_type, self.amount, self.unit, to_unit)
        elif from_type == "mass" and to_type == "moles":
            amount_g = self._converter.convert("mass", self.amount, self.unit, "g")
            converted_amount = amount_g / self.molar_mass
            if to_unit != "mol":
                converted_amount = self._converter.convert("moles", converted_amount, "mol", to_unit)
        elif from_type == "moles" and to_type == "mass":
            amount_mol = self._converter.convert("moles", self.amount, self.unit, "mol")
            converted_amount = amount_mol * self.molar_mass
            if to_unit != "g":
                converted_amount = self._converter.convert("mass", converted_amount, "g", to_unit)
        else:
            raise ValueError(f"Cannot convert from {self.unit} ({from_type}) to {to_unit} ({to_type}).")
        
        return Species(name=self.name, amount=converted_amount, unit=to_unit, molar_mass=self.molar_mass)

    def __str__(self):
        """Return a string representation of the `Species` instance."""
        return f"Species(name='{self.name}', amount={self.amount}, unit='{self.unit}', molar_mass={self.molar_mass})"
    
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Dump instance to a dict."""
        return {
            "name": self.name,
            "amount": self.amount,
            "unit": self.unit,
            "molar_mass": self.molar_mass
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, float]]):
        """Load a `Species` instance from a dict.
        
        Parameters
        ----------
        data : Dict[str, Union[str, float]]
            Dictionary representation of a Species instance.
            Requires the keys `"name": str`, `"amount": float`, `"unit": str`, and `"molar_mass": float`.
        
        Returns
        -------
        Species
            A Species instance created from the dictionary representation.
        """
        return cls(name=data["name"], amount=data["amount"], unit=data["unit"], molar_mass=data["molar_mass"])


@attr.s(auto_attribs=True)
class SolventBlendSpec:
    """A specification for solvent blending: Base + x Solvent -> Composition.

    Attributes
    ----------
    base : str
        Name of the base fluid.
    solvent : str
        Name of the solvent fluid.
    unit : str
        Unit for blending (e.g., "molf" or "mol%" for moles, "wtf" or "wt%" for mass, "volf" or "vol%" for volume).
    value : float or List[float]
        Amount of solvent in the final blend (in the specified unit).
        Each element corresponds to a new Composition specification.

    Examples
    --------
    >>> spec = SolventBlendSpec(base="H2O", solvent="Ethanol", value=0.5, unit="wtf")
    >>> print(spec)
    SolventBlendSpec(base='H2O', solvent='Ethanol', value=0.5, unit='wtf')
    >>> spec_multi = SolventBlendSpec(base="H2O", solvent="Ethanol", value=[0.5, 1.0], unit="mol%")
    >>> print(spec_multi.value)
    [0.5 1. ]
    """
    base: str
    solvent: str
    unit: str
    value: Union[float, List[float]]

    def __attrs_post_init__(self):
        """Validate the unit and value."""
        valid_units = {"wtf", "wt%", "molf", "mol%", "volf", "vol%"}
        if self.unit not in valid_units:
            raise ValueError(f"Unit must be one of {valid_units}, got '{self.unit}'.")
        if isinstance(self.value, (list, tuple)):
            if not all(isinstance(x, (int, float)) for x in self.value):
                raise ValueError("value list must contain numbers.")
        elif not isinstance(self.value, (int, float)):
            raise ValueError("value must be a number or list of numbers.")
        logger.debug(f"Created SolventBlendSpec: {self}")

    @property
    def value(self) -> np.ndarray:
        """Return value as an array for compatibility with plotting."""
        return np.array(self.value if isinstance(self.value, (list, tuple)) else [self.value])

    def __str__(self):
        """Return a string representation."""
        return f"SolventBlendSpec(base='{self.base}', solvent='{self.solvent}', unit='{self.unit}', value={self.value})"
    
    def __getattr__(self, attr):
        # Allow access to base and solvent as attributes
        if attr == "name":
            return self.solvent
        if attr == "base":
            return self.base
        raise AttributeError(attr)

    def to_dict(self, tolist=True) -> Dict[str, Union[str, float]]:
        """Dump instance to a dict.
        
        Parameters
        ----------
        tolist : bool, optional
            Whether to convert the value array to a list.
        """
        return {
            "base": self.base,
            "solvent": self.solvent,
            "unit": self.unit,
            "value": self.tolist() if tolist else self.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, float]]):
        """Load a `SolventBlendSpec` instance from a dict.
        
        Parameters
        ----------
        data : Dict[str, Union[str, float]]
            Dictionary representation of a SolventBlendSpec instance.
            Requires the keys `"base": str`, `"solvent": str`, `"value": float`, and `"unit": str`.
        
        Returns
        -------
        SolventBlendSpec
            A SolventBlendSpec instance created from the dictionary representation.
        """
        return cls(base=data["base"], solvent=data["solvent"], unit=data["unit"], value=data["value"])

    def tolist(self) -> List[float]:
        """Convert value to a list."""
        return self.value.tolist() if isinstance(self.value, np.ndarray) else list(self.value)

@attr.s(auto_attribs=True)
class ChemicalComposition:
    """Represents a mixture of chemical species.

    Attributes
    ----------
    species : List[Species]
        List of species in the composition.
    name : str
        Name of the composition (defaults to "composition").

    Examples
    --------
    >>> water = Species(name="H2O", amount=10, unit="kg", molar_mass=18.015)
    >>> co2 = Species(name="CO2", amount=5, unit="kg", molar_mass=44.01)
    >>> comp = ChemicalComposition(species=[water, co2], name="Mixture")
    >>> print(comp.total_amount("kg"))
    15.0
    >>> comp_moles = comp.convert("mol")
    >>> print(f"{comp_moles.total_amount('mol'):.1f} mol")
    668.9 mol
    """
    species: List[Species] = attr.ib(default=attr.Factory(list))
    name: str = "composition"

    def __attrs_post_init__(self):
        """Validate species."""
        if not self.species:
            raise ValueError("Composition must contain at least one species.")
        logger.debug(f"Created ChemicalComposition: {self}")

    def convert(self, to_unit: str) -> "ChemicalComposition":
        """Convert all species to the specified unit."""
        converted_species = [s.to(to_unit) for s in self.species]
        return ChemicalComposition(species=converted_species, name=self.name)

    def total_amount(self, unit: str = None) -> float:
        """Return the total amount in the specified unit (or first species' unit if None)."""
        target_unit = unit or self.species[0].unit
        converted = self.convert(target_unit)
        return sum(s.amount for s in converted.species)

    def blend(self, other: "ChemicalComposition", proportion: float, unit: str) -> "ChemicalComposition":
        """Blend with another Composition based on a proportion in the specified unit.

        Combines all species from both compositions, summing amounts for overlapping species.

        Parameters
        ----------
        other : Composition
            The other composition to blend with.
        proportion : float
            Fraction of self (0 to 1) in the specified unit.
        unit : str
            Unit for blending (e.g., "kg" for mass, "mol" for moles).

        Returns
        -------
        ChemicalComposition
            The blended composition containing all species from both inputs.
        """
        if not isinstance(other, ChemicalComposition):
            raise TypeError("Can only blend with another Composition.")
        if not 0 <= proportion <= 1:
            raise ValueError("Proportion must be between 0 and 1.")

        self_conv = self.convert(unit)
        other_conv = other.convert(unit)

        # Build a molar mass lookup from both compositions
        molar_mass_dict = {s.name: s.molar_mass for s in self_conv.species}
        molar_mass_dict.update({s.name: s.molar_mass for s in other_conv.species})

        # Combine species amounts
        species_dict = {s.name: s.amount * proportion for s in self_conv.species}
        for s in other_conv.species:
            species_dict[s.name] = species_dict.get(s.name, 0) + s.amount * (1 - proportion)

        # Create blended species list with molar masses from the lookup
        blended_species = [
            Species(name=name, amount=value, unit=unit, molar_mass=molar_mass_dict[name])
            for name, value in species_dict.items()
        ]
        return ChemicalComposition(species=blended_species, name=f"{self.name}+{other.name}")


    def to_dict(self) -> Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]:
        """Dump instance to a dict."""
        return {"name": self.name, "species": [s.to_dict() for s in self.species]}

    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]):
        """Load a `ChemicalComposition` instance from a dict.
        
        Parameters
        ----------
        data : Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]
            Dictionary representation of a Composition instance.
            Requires the keys `"name": str` and `"species": List[Dict[str, Union[str, float]]]`.
        
        Returns
        -------
        Composition
            A Composition instance created from the dictionary representation.
        """
        return cls(species=[Species.from_dict(s) for s in data["species"]], name=data["name"])
    
    def __str__(self):
        """Return a string representation of the instance."""
        return f"Composition(name='{self.name}', species={self.species})"
    

@attr.s(auto_attribs=True)
class Fluids:
    """A collection of fluid compositions.

    Attributes
    ----------
    compositions : List[Composition]
        List of Composition instances.

    Examples
    --------
    >>> water = Species(name="H2O", amount=10, unit="kg", molar_mass=18.015)
    >>> co2 = Species(name="CO2", amount=5, unit="kg", molar_mass=44.01)
    >>> comp1 = Composition(species=[water], name="Water Stream")
    >>> comp2 = Composition(species=[co2], name="CO2 Stream")
    >>> fluids = Fluids(compositions=[comp1])
    >>> fluids.add(comp2)
    >>> blended = fluids.blend(indices=[0, 1], proportions=[0.2, 0.8], unit="mol")
    >>> print(f"{blended.total_amount('mol'):.1f} mol")
    227.8 mol
    """
    compositions: List[ChemicalComposition] = attr.ib(default=attr.Factory(list))

    def add(self, composition: ChemicalComposition) -> None:
        """Add a Composition instance to the Fluids instance."""
        if not isinstance(composition, ChemicalComposition):
            raise TypeError("Can only add Composition instances.")
        self.compositions.append(composition)

    def convert_all(self, to_unit: str) -> "Fluids":
        """Convert all compositions to the specified unit."""
        converted_comps = [comp.convert(to_unit) for comp in self.compositions]
        return Fluids(compositions=converted_comps)

    def blend(self, indices: List[int], proportions: List[float], unit: str) -> ChemicalComposition:
        """Blend specific compositions in specified proportions using the given unit."""
        if len(indices) != len(proportions):
            raise ValueError("Number of indices must match number of proportions.")
        if not np.isclose(sum(proportions), 1):
            raise ValueError("Proportions must sum to 1.")
        if not all(0 <= p <= 1 for p in proportions):
            raise ValueError("Proportions must be between 0 and 1.")
        if not all(0 <= i < len(self.compositions) for i in indices):
            raise IndexError("Invalid composition indices.")

        selected_comps = [self.compositions[i] for i in indices]
        result = selected_comps[0].convert(unit)
        proportion_used = proportions[0]

        for comp, prop in zip(selected_comps[1:], proportions[1:]):
            result = result.blend(comp, proportion=prop / (1 - proportion_used), unit=unit)
            proportion_used += prop

        return result

    def from_spec(self, spec: SolventBlendSpec, exclude_zero_comps: bool = False) -> List[ChemicalComposition]:
        """Construct new compositions from a SolventBlendSpec.

        The spec's value denotes the solvent amount in the final blend, with the base amount
        derived to satisfy the unit (e.g., "wtf" for weight fraction, "mol%" for mole percent).
        Total blend amount is normalized to 1 unit (e.g., 1 kg or 1 mole).

        Parameters
        ----------
        spec : SolventBlendSpec
            The specification for blending a base fluid with a solvent.

        Returns
        -------
        List[Composition]
            A list of new Composition instances based on the spec.

        Examples
        --------
        >>> water = Species(name="H2O", amount=1, unit="kg", molar_mass=18.015)
        >>> ethanol = Species(name="Ethanol", amount=0, unit="kg", molar_mass=46.07)
        >>> comp1 = Composition(species=[water], name="H2O")
        >>> comp2 = Composition(species=[ethanol], name="Ethanol")
        >>> fluids = Fluids(compositions=[comp1, comp2])
        >>> spec = SolventBlendSpec(base="H2O", solvent="Ethanol", value=[0.5, 1.0], unit="wtf")
        >>> new_comps = fluids.from_spec(spec)
        >>> print([comp.total_amount("kg") for comp in new_comps])
        [1.0, 1.0]
        """
        if not isinstance(spec, SolventBlendSpec):
            raise TypeError("Spec must be a SolventBlendSpec instance.")

        base_comp = next((c for c in self.compositions if c.name == spec.base), None)
        solvent_comp = next((c for c in self.compositions if c.name == spec.solvent), None)
        if not base_comp or not solvent_comp:
            raise ValueError(f"Base '{spec.base}' or solvent '{spec.solvent}' not found in Fluids.")

        unit_map = {"wtf": "kg", "wt%": "kg", "molf": "mol", "mol%": "mol", "volf": "m^3", "vol%": "m^3"}
        unit = unit_map.get(spec.unit)
        if not unit:
            raise ValueError(f"Unsupported unit '{spec.unit}' in SolventBlendSpec. Use 'wtf', 'wt%', 'molf', 'mol%', 'volf', or 'vol%'.")

        base_conv = base_comp.convert(unit)
        solvent_conv = solvent_comp.convert(unit)
        base_total = base_conv.total_amount(unit)

        values = spec.value if isinstance(spec.value, (list, tuple)) else [spec.value]
        new_compositions = []
        for value in values:
            solvent_fraction = value / 100 if spec.unit.endswith("%") else value
            # Total blend normalized to 1 unit
            solvent_amount = solvent_fraction  # Solvent amount in final blend
            base_amount = 1 - solvent_fraction  # Base amount to reach total of 1

            # Scale base composition to base_amount
            base_scale = base_amount / base_total if base_total > 0 else 0
            base_species = [
                Species(name=s.name, amount=s.amount * base_scale, unit=unit, molar_mass=s.molar_mass)
                for s in base_conv.species
            ]

            # Set solvent composition to solvent_amount (replace initial amount)
            solvent_species = [
                Species(name=s.name, amount=solvent_amount if s.name == solvent_comp.species[0].name else 0, unit=unit, molar_mass=s.molar_mass)
                for s in solvent_conv.species
            ]

            # Combine species, summing overlapping ones
            species_dict = {s.name: s.amount for s in base_species}
            for s in solvent_species:
                species_dict[s.name] = species_dict.get(s.name, 0) + s.amount

            molar_mass_dict = {s.name: s.molar_mass for s in base_conv.species} | {
                s.name: s.molar_mass for s in solvent_conv.species
            }
            # Create blended species list
            amount_tol = 0 if exclude_zero_comps else -1e-6
            blended_species = [
                Species(name=name, amount=amount, unit=unit, molar_mass=molar_mass_dict[name])
                for name, amount in species_dict.items() if amount > amount_tol  # Exclude zero amounts
            ]
            new_compositions.append(ChemicalComposition(species=blended_species, name=f"{base_comp.name}+{solvent_comp.name}"))

        return new_compositions

    def to_dict(self):
        """Dump instance to a dict."""
        return {"compositions": [c.to_dict() for c in self.compositions]}

    @classmethod
    def from_dict(cls, data):
        """Load a `Fluids` instance from a dict.
        
        Parameters
        ----------
        data : dict
            Dictionary representation of a Fluids instance.
            Requires the key `"compositions": List[Composition]`.
        
        Returns
        -------
        Fluids
            A Fluids instance created from the dictionary representation.
        """
        return cls(compositions=[ChemicalComposition.from_dict(c) for c in data["compositions"]])
    
    def __str__(self):
        """String representation of the `Fluids` instance."""
        return f"Fluids(compositions={self.compositions})"
