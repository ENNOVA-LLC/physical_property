""" `physical_property`

Top-level module for the physical_property package.
Provides access to physical property classes and utilities.
"""
# Quiet the package by default (apps can enable explicitly)
from loguru import logger as _logger
_logger.disable("physical_property")

from importlib.metadata import PackageNotFoundError, version

from .base import PhysicalProperty
from .properties import (
    Time, Length, Rate, Area, Angle, Velocity,
    ReynoldsNumber, PecletNumber, FroudeNumber, DamkohlerNumber, PrandtlNumber, NusseltNumber, SchmidtNumber, MachNumber,
    Mass, Moles, Composition, MolarMass, Temperature, Pressure, Density, Volume, SolubilityParameter,
    Flow, Viscosity, ShearStress, PressureGradient, PressureDrop, SurfaceTension, 
    MassFlux, Diffusivity, Dispersion, ThermalConductivity, HeatTransferCoefficient, HeatCapacity,
    GasOilRatio, Species, ChemicalComposition, Fluids, SolventBlendSpec
)
from .xy_data import XYData
from .units.units import UnitConverter, get_unit_set

# Get version number
try:
    __version__ = version("physical_property")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    'PhysicalProperty',
    # properties
    'Time', 'Length', 'Rate', 'Area', 'Angle', 'Velocity',
    'ReynoldsNumber', 'PecletNumber', 'FroudeNumber', 'DamkohlerNumber', 'PrandtlNumber', 'NusseltNumber', 'SchmidtNumber', 'MachNumber',
    'Mass', 'Moles', 'Composition', 'MolarMass', 'Temperature', 'Pressure', 'Density', 'Volume', 'SolubilityParameter',
    'Flow', 'Viscosity', 'ShearStress', 'PressureGradient', 'PressureDrop', 
    'SurfaceTension', 'MassFlux', 'Diffusivity', 'Dispersion', 'ThermalConductivity', 'HeatTransferCoefficient', 'HeatCapacity',
    'GasOilRatio', 'Species', 'ChemicalComposition', 'Fluids', 'SolventBlendSpec',
    # xy-data
    'XYData',
    # units
    'UnitConverter', 'get_unit_set',
]
