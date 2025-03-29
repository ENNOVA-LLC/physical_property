""" `properties.specialty.__init__`

Contains classes for specialty physical properties (e.g., gas-oil ratio).
"""
from .gas_oil_ratio import GasOilRatio
from .composition import Species, Composition, Fluids, SolventBlendSpec

__all__ = [
    "GasOilRatio",
    "Species", "Composition", "Fluids", "SolventBlendSpec"
]
