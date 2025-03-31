""" `properties.specialty.__init__`

Contains classes for specialty physical properties (e.g., gas-oil ratio).
"""
from .gas_oil_ratio import GasOilRatio
from .composition import Species, ChemicalComposition, Fluids, SolventBlendSpec

__all__ = [
    "GasOilRatio",
    "Species", "ChemicalComposition", "Fluids", "SolventBlendSpec"
]
