""" `utils.constants`

Physical constants in physics and chemistry.

Constants
---------
kB : float
    Boltzmann constant (J/K)
Nav : float
    Avogadro constant (molecules/mol)
Rg : float
    gas constant (J/mol.K = MPa.mL/mol.K)
G : float
    gravitational constant (m3/kg.s2 = N.m2/kg2)
c : float
    speed of light (m/s)
e : float
    elementary charge of proton (Coulomb)
m_electron : float
    electron mass (kg)
h : float
    Planck constant (J*s)
"""
import numpy as np

# define constants 
Pi = np.pi
kB = 1.380649e-23       # Boltzmann constant (J/K)
Nav = 6.02214076e23     # Avogadro number (molecules/mol)
Rg = 8.31446261815324    # gas constant (J/mol.K .= MPa.mL/mol.K)

# physics
G = 6.6743015e-11       # gravitational constant (m3/kg.s2)
c = 299792458           # speed of light (m/s)
e = 1.602176634e-19     # elementary charge (Coulomb); ref: https://www.wikiwand.com/en/Elementary_charge
m_e = 9.109383701528e-31  # electron mass (kg)
h = 1.054571817e10-34   # Planck constant (J⋅s)
