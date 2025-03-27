""" `properties.dimensionless`

Contains classes for dimensionless numbers (e.g., Reynolds number, Prandtl number, Damkohler number, etc.).
"""
import attr
import numpy as np
from scipy.optimize import fsolve

from ..base import PhysicalProperty
from .basic import Length, Velocity, Time, Rate
from .thermodynamic import Density
from .transport import Viscosity, Diffusivity, ThermalConductivity, HeatTransferCoefficient

# region DIMENSIONLESS NUMBERS
@attr.s(auto_attribs=True)
class ReynoldsNumber(PhysicalProperty):
    """Reynolds number (dimensionless). Ratio of inertial to viscous forces acting on a liquid
    
    Reference
    ---------
    https://en.wikipedia.org/wiki/Reynolds_number
    """
    def convert(self, to_unit):
        """Dimensionless numbers do not support unit conversion. Returns the original value."""
        return self.value

    @classmethod
    def from_definition(cls, length: Length, velocity: Velocity, density: Density, viscosity: Viscosity) -> 'ReynoldsNumber':
        """Create a Reynolds number from its definition.
        
        Parameters
        ----------
        length : Length
            Characteristic length.
        velocity : Velocity
            Fluid velocity.
        density : Density
            Fluid density.
        viscosity : Viscosity
            Fluid viscosity.
        """
        instance = cls(name="Reynolds number", value=0.0, unit=None)
        instance.update_from_definition(length, velocity, density, viscosity)
        return instance
    
    def update_from_definition(self, length: Length, velocity: Velocity, density: Density, viscosity: Viscosity) -> None:
        """Update the Reynolds number from its definition.
        
        Parameters
        ----------
        length : Length
            Characteristic length.
        velocity : Velocity
            Fluid velocity.
        density : Density
            Fluid density.
        viscosity : Viscosity
            Fluid viscosity.
        """
        velocity_val = velocity.convert('m/s')
        length_val = length.convert('m')
        density_val = density.convert('kg/m3')
        viscosity_val = viscosity.convert('Pa*s')
        Re = (velocity_val * length_val * density_val) / viscosity_val
        self.update_value(Re)
    
    def friction_factor(self, radius: Length, epsilon: Length=None) -> float:
        """Friction factor using Colebrook-White equation

        Parameters
        ----------
        radius : Length
            Pipe radius.
        epsilon : Length
            Roughness of the pipe.
        
        Reference
        ---------
        1. https://www.wikiwand.com/en/articles/Darcy_friction_factor
        2. https://www.wikiwand.com/en/articles/Colebrook-White_equation
        """
        #TODO: Check Fernando's formula.
        Re_val = self.value
        f = np.where(Re_val < 2300, 64 / Re_val, 0.0056 + 0.5 / Re_val ** 0.32)

        if epsilon is None:
            return f
        
        # Convert diameter and pipe roughness
        d = radius.convert('m') * 2
        e = epsilon.convert('m')
        
        # Solve Colebrook-White equation (iterative)
        def friction_factor_fun(f):
            # Fernando's formula
            return (1. / np.sqrt(f)) - 1.74 + 2. * np.log10(2 * e / d + 18.7 / (Re_val * np.sqrt(f)))
        return fsolve(friction_factor_fun, x0=f)

@attr.s(auto_attribs=True)
class PecletNumber(PhysicalProperty):
    """Peclet number (dimensionless) for mass transfer. Ratio of a fluid's advective and diffusive transport rates.
    
    Reference
    ---------
    https://en.wikipedia.org/wiki/Péclet_number
    """
    def convert(self, to_unit):
        """Dimensionless numbers do not support unit conversion. Returns the original value."""
        return self.value
    
    @classmethod
    def from_definition(cls, length: Length, velocity: Velocity, diffusivity: Diffusivity) -> 'PecletNumber':
        """Create a Peclet number from its definition.
        
        Parameters
        ----------
        length : Length
            Characteristic length.
        velocity : Velocity
            Fluid velocity.
        diffusion_coefficient : DiffusionCoefficient
            Mass diffusion coefficient.
        """
        instance = cls(name="Peclet number", value=0.0, unit=None)
        instance.update_from_definition(length, velocity, diffusivity)
        return instance

    def update_from_definition(self, length: Length, velocity: Velocity, diffusivity: Diffusivity) -> None:
        """Update the Peclet number from its definition.
        
        Parameters
        ----------
        length : Length
            Characteristic length.
        velocity : Velocity
            Fluid velocity.
        diffusivity : Diffusivity
            Mass diffusivity.
        """
        velocity_val = velocity.convert('m/s')
        length_val = length.convert('m')
        diffusivity_val = diffusivity.convert('m2/s')
        Pe = (velocity_val * length_val) / diffusivity_val
        self.update_value(Pe)

@attr.s(auto_attribs=True)
class FroudeNumber(PhysicalProperty):
    """Froude number property (dimensionless). Ratio of a fluid's flow inertia to the external field.
    
    Reference
    ---------
    https://en.wikipedia.org/wiki/Froude_number
    """
    def convert(self, to_unit):
        """Dimensionless numbers do not support unit conversion. Returns the original value."""
        return self.value
    
    @classmethod
    def from_definition(cls, velocity: Velocity, length: Length, gravity: float=9.81) -> 'FroudeNumber':
        """Create a Froude number from its definition.
        
        Parameters
        ----------
        velocity : Velocity
            Fluid velocity.
        length : Length
            Characteristic length.
        gravity : float
            Acceleration due to gravity (m/s²).
        """
        instance = cls(name="Froude number", value=0.0, unit=None)
        instance.update_from_definition(velocity, length, gravity)
        return instance
    
    def update_from_definition(self, velocity: Velocity, length: Length, gravity: float=9.81) -> None:
        """Update the Froude number from its definition.
        
        Parameters
        ----------
        velocity : Velocity
            Fluid velocity.
        length : Length
            Characteristic length.
        gravity : float
            Acceleration due to gravity (m/s²).
        """
        velocity_val = velocity.convert('m/s')
        length_val = length.convert('m')
        Fr = velocity_val / np.sqrt(gravity * length_val)
        self.update_value(Fr)

@attr.s(auto_attribs=True)
class DamkohlerNumber(PhysicalProperty):
    """Damköhler number (dimensionless). Ratio of reaction rate to transport rate.
    
    Reference
    ---------
    https://en.wikipedia.org/wiki/Damköhler_numbers
    """
    def convert(self, to_unit):
        """Dimensionless numbers do not support unit conversion. Returns the original value."""
        return self.value
    
    @classmethod
    def from_definition(cls, rate: Rate, t_residence: Time) -> 'DamkohlerNumber':
        """Create a Damköhler number from its definition.
        
        Parameters
        ----------
        rate : Rate
            Reaction rate (1/t).
        t_residence : Time
            Residence time (t).
        """
        instance = cls(name="Damköhler number", value=0.0, unit=None)
        instance.update_from_definition(rate, t_residence)
        return instance
    
    def update_from_definition(self, rate: Rate, t_residence: Time) -> None:
        """Update the Damköhler number from its definition.
        
        Parameters
        ----------
        rate : Rate
            Reaction rate (1/t).
        t_residence : Time
            Residence time (t).
        """
        rate_val = rate.convert('1/s')
        res_time_val = t_residence.convert('s')
        Da = rate_val * res_time_val
        self.update_value(Da)

@attr.s(auto_attribs=True)
class PrandtlNumber(PhysicalProperty):
    """Prandtl number (dimensionless). Ratio of kinematic to thermal diffusivity.
    
    Reference
    ---------
    https://en.wikipedia.org/wiki/Prandtl_number
    """
    def convert(self, to_unit):
        """Dimensionless numbers do not support unit conversion. Returns the original value."""
        return self.value
    

@attr.s(auto_attribs=True)
class NusseltNumber(PhysicalProperty):
    """Nusselt number (dimensionless). Ratio of convective to conductive heat transfer.
    
    Reference
    ---------
    1. https://en.wikipedia.org/wiki/Nusselt_number
    2. https://resources.wolframcloud.com/FormulaRepository/resources/DittusBoelter-Equation

    Notes
    -----
    The Nusselt number is a dimensionless number that represents the ratio of convective to conductive heat transfer.
    """
    def convert(self, to_unit):
        """Dimensionless numbers do not support unit conversion. Returns the original value."""
        return self.value
    
    @classmethod
    def from_definition(cls, h: HeatTransferCoefficient, L: Length, k: ThermalConductivity) -> 'NusseltNumber':
        """Create a Nusselt number from its definition.
        
        Parameters
        ----------
        h : HeatTransferCoefficient
            Convective heat transfer coefficient.
        L : Length
            Characteristic length.
        k : ThermalConductivity
            Thermal conductivity.
        """
        instance = cls(name="Nusselt number", value=0.0, unit=None)
        instance.update_from_definition(h, L, k)
        return instance
    
    def update_from_definition(self, h: HeatTransferCoefficient, L: Length, k: ThermalConductivity) -> None:
        """Update the Nusselt number from its definition.
        
        Parameters
        ----------
        h : HeatTransferCoefficient
            Convective heat transfer coefficient.
        L : Length
            Characteristic length.
        k : ThermalConductivity
            Thermal conductivity.
        """
        h_val = h.convert('W/m²*K')
        L_val = L.convert('m')
        k_val = k.convert('W/m*K')
        Nu = h_val * L_val / k_val
        self.update_value(Nu)
    
    def update_from_dittus_boelter(self, Re: ReynoldsNumber, Pr: PrandtlNumber, n: float = 0.33) -> None:
        """Update the Nusselt number using the Dittus-Boelter correlation.
        
        Parameters
        ----------
        Re : ReynoldsNumber
            Reynolds number.
        Pr : PrandtlNumber
            Prandtl number.
        n : float, optional
            "Heating" exponent for the Dittus-Boelter correlation (default: 0.33).
        
        Reference
        ---------
        https://resources.wolframcloud.com/FormulaRepository/resources/DittusBoelter-Equation
        """
        Re_val = Re.value
        Pr_val = Pr.value
        Nu = 0.023 * Re_val ** 0.8 * Pr_val ** n
        self.update_value(Nu)

@attr.s(auto_attribs=True)
class SchmidtNumber(PhysicalProperty):
    """Schmidt number (dimensionless). Ratio of a fluid's kinematic viscosity to mass diffusivity.
    
    Reference
    ---------
    https://en.wikipedia.org/wiki/Schmidt_number
    """
    def convert(self, to_unit):
        """Dimensionless numbers do not support unit conversion. Returns the original value."""
        return self.value
    
    @classmethod
    def from_definition(cls, viscosity: Viscosity, density: Density, diffusivity: Diffusivity) -> 'SchmidtNumber':
        """Create a Schmidt number from its definition.
        
        Parameters
        ----------
        viscosity : Viscosity
            Fluid viscosity.
        density : Density
            Fluid density.
        diffusivity : Diffusivity
            Mass diffusivity.
        """
        instance = cls(name="Schmidt number", value=0.0, unit=None)
        instance.update_from_definition(viscosity, density, diffusivity)
        return instance
    
    def update_from_definition(self, viscosity: Viscosity, density: Density, diffusivity: Diffusivity) -> None:
        """Update the Schmidt number from its definition.
        
        Parameters
        ----------
        viscosity : Viscosity
            Fluid viscosity.
        density : Density
            Fluid density.
        diffusivity : Diffusivity
            Mass diffusivity.
        """
        viscosity_val = viscosity.convert('Pa*s')
        density_val = density.convert('kg/m3')
        diffusivity_val = diffusivity.convert('m2/s')
        Sc = viscosity_val / (density_val * diffusivity_val)
        self.update_value(Sc)
    
    def update_from_Pe_Re(self, peclet: PecletNumber, reynolds: ReynoldsNumber) -> None:
        """Update the Schmidt number from Peclet and Reynolds numbers.
        
        Parameters
        ----------
        peclet : PecletNumber
            Peclet number.
        reynolds : ReynoldsNumber
            Reynolds number.
        """
        Sc = peclet.value / reynolds.value
        self.update_value(Sc)

@attr.s(auto_attribs=True)
class MachNumber(PhysicalProperty):
    """Mach number (dimensionless). Ratio of the fluid velocity to the speed of sound in the fluid.
    
    Reference
    ---------
    https://en.wikipedia.org/wiki/Mach_number
    """
    def convert(self, to_unit):
        """Dimensionless numbers do not support unit conversion. Returns the original value."""
        return self.value
    
    @classmethod
    def from_definition(cls, flow_velocity: Velocity, sound_velocity: Velocity) -> 'MachNumber':
        """Create a Schmidt number from its definition.
        
        Parameters
        ----------
        flow_velocity : Velocity
            Fluid velocity.
        sound_velocity : Velocity
            Velocity of sound in the medium.
        """
        instance = cls(name="Mach number", value=0.0, unit=None, doc="Mach: (<0.8: subsonic, 0.8-1.2: transonic, >1.2-5.0: supersonic, >5.0: hypersonic)")
        instance.update_from_definition(flow_velocity, sound_velocity)
        return instance
    
    def update_from_definition(self, flow_velocity: Velocity, sound_velocity: Velocity) -> None:
        """Update the Schmidt number from its definition.
        
        Parameters
        ----------
        flow_velocity : Velocity
            Fluid velocity.
        sound_velocity : Velocity
            Velocity of sound in the medium.
        """
        u_val = flow_velocity.convert('m/s')
        c_val = sound_velocity.convert('m/s')
        mach = u_val / c_val
        self.update_value(mach)
